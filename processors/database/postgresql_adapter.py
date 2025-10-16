#!/usr/bin/env python3
"""
PostgreSQL Database Adapter
Implements BaseDatabaseAdapter for PostgreSQL backend
"""

import logging
from typing import Dict, List, Any, Optional
try:
    import psycopg2
    import psycopg2.extras
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

from .base_adapter import (
    BaseDatabaseAdapter, DatabaseBackend,
    TimelineEvent, CausalLink, CharacterData
)

logger = logging.getLogger(__name__)


class PostgreSQLAdapter(BaseDatabaseAdapter):
    """PostgreSQL implementation of database adapter"""

    def __init__(self, config: Any):
        """Initialize PostgreSQL adapter"""
        super().__init__(config)
        self._backend_type = DatabaseBackend.POSTGRESQL
        self.conn = None
        self.cursor = None

        if not POSTGRES_AVAILABLE:
            raise ImportError("psycopg2 not installed. Install with: pip install psycopg2-binary")

    # ========================================================================
    # CONNECTION MANAGEMENT
    # ========================================================================

    def connect(self) -> bool:
        """Establish PostgreSQL connection"""
        try:
            self.conn = psycopg2.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_db,
                user=self.config.postgres_user,
                password=self.config.postgres_password
            )
            self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            self.connected = True
            logger.info(f"✅ PostgreSQL connected: {self.config.postgres_host}:{self.config.postgres_port}")
            return True
        except Exception as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Close PostgreSQL connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        self.connected = False
        logger.info("PostgreSQL disconnected")

    def is_connected(self) -> bool:
        """Check PostgreSQL connection status"""
        return self.connected and self.conn and not self.conn.closed

    # ========================================================================
    # SCHEMA MANAGEMENT
    # ========================================================================

    def initialize_schema(self):
        """Create PostgreSQL tables for timeline system"""
        if not self.is_connected():
            raise ConnectionError("Not connected to PostgreSQL")

        schema_sql = """
        -- Timeline Events Table
        CREATE TABLE IF NOT EXISTS timeline_events (
            event_id VARCHAR PRIMARY KEY,
            volume_id INTEGER NOT NULL,
            batch_id INTEGER NOT NULL,
            chronological_order INTEGER,
            description TEXT NOT NULL,
            event_type VARCHAR NOT NULL,
            importance_score REAL DEFAULT 0.5,
            confidence_level REAL DEFAULT 0.5,
            primary_actors TEXT[],
            affected_characters TEXT[],
            caused_by_events TEXT[],
            causes_events TEXT[],
            temporal_markers TEXT[],
            created_at TIMESTAMP DEFAULT NOW()
        );

        -- Event Causality Table
        CREATE TABLE IF NOT EXISTS event_causality (
            id SERIAL PRIMARY KEY,
            cause_event_id VARCHAR NOT NULL,
            effect_event_id VARCHAR NOT NULL,
            causality_type VARCHAR NOT NULL,
            strength REAL DEFAULT 0.5,
            reasoning TEXT,
            confidence REAL DEFAULT 0.5,
            created_at TIMESTAMP DEFAULT NOW(),
            FOREIGN KEY (cause_event_id) REFERENCES timeline_events(event_id),
            FOREIGN KEY (effect_event_id) REFERENCES timeline_events(event_id),
            UNIQUE(cause_event_id, effect_event_id)
        );

        -- Character Registry Table
        CREATE TABLE IF NOT EXISTS character_registry (
            id SERIAL PRIMARY KEY,
            character_id VARCHAR UNIQUE NOT NULL,
            name VARCHAR NOT NULL,
            volume_id INTEGER NOT NULL,
            batch_id INTEGER NOT NULL,
            character_type VARCHAR,
            aliases TEXT[],
            personality_traits TEXT[],
            first_appearance INTEGER,
            confidence_score REAL DEFAULT 0.5,
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(name, volume_id)
        );

        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_timeline_events_volume ON timeline_events(volume_id);
        CREATE INDEX IF NOT EXISTS idx_timeline_events_chronological ON timeline_events(chronological_order);
        CREATE INDEX IF NOT EXISTS idx_timeline_events_type ON timeline_events(event_type);
        CREATE INDEX IF NOT EXISTS idx_event_causality_cause ON event_causality(cause_event_id);
        CREATE INDEX IF NOT EXISTS idx_event_causality_effect ON event_causality(effect_event_id);
        CREATE INDEX IF NOT EXISTS idx_character_registry_name ON character_registry(name);
        CREATE INDEX IF NOT EXISTS idx_character_registry_volume ON character_registry(volume_id);
        """

        try:
            self.cursor.execute(schema_sql)
            self.conn.commit()
            logger.info("✅ PostgreSQL schema initialized")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ Schema initialization failed: {e}")
            raise

    def verify_schema(self) -> bool:
        """Verify PostgreSQL schema exists"""
        try:
            self.cursor.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('timeline_events', 'event_causality', 'character_registry')
            """)
            tables = [row['table_name'] for row in self.cursor.fetchall()]
            return len(tables) == 3
        except Exception as e:
            logger.error(f"Schema verification failed: {e}")
            return False

    # ========================================================================
    # EVENT OPERATIONS
    # ========================================================================

    async def store_event(self, event: TimelineEvent) -> bool:
        """Store timeline event to PostgreSQL"""
        try:
            self.cursor.execute("""
                INSERT INTO timeline_events (
                    event_id, volume_id, batch_id, chronological_order, description,
                    event_type, importance_score, confidence_level,
                    primary_actors, affected_characters, caused_by_events, causes_events,
                    temporal_markers, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (event_id) DO UPDATE SET
                    description = EXCLUDED.description,
                    importance_score = EXCLUDED.importance_score,
                    chronological_order = EXCLUDED.chronological_order
            """, (
                event.event_id, event.volume_id, event.batch_id, event.chronological_order,
                event.description, event.event_type, event.importance_score, event.confidence_level,
                event.primary_actors or [], event.affected_characters or [],
                event.caused_by_events or [], event.causes_events or [],
                event.temporal_markers or [], event.created_at
            ))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to store event {event.event_id}: {e}")
            return False

    async def get_event(self, event_id: str) -> Optional[TimelineEvent]:
        """Retrieve event from PostgreSQL"""
        try:
            self.cursor.execute("SELECT * FROM timeline_events WHERE event_id = %s", (event_id,))
            row = self.cursor.fetchone()
            if row:
                return TimelineEvent(
                    event_id=row['event_id'],
                    volume_id=row['volume_id'],
                    batch_id=row['batch_id'],
                    description=row['description'],
                    event_type=row['event_type'],
                    importance_score=row['importance_score'],
                    chronological_order=row['chronological_order'],
                    primary_actors=row['primary_actors'],
                    affected_characters=row['affected_characters'],
                    caused_by_events=row['caused_by_events'],
                    causes_events=row['causes_events'],
                    temporal_markers=row['temporal_markers'],
                    confidence_level=row['confidence_level'],
                    created_at=row['created_at']
                )
            return None
        except Exception as e:
            logger.error(f"Failed to get event {event_id}: {e}")
            return None

    async def query_events(self, filters: Dict[str, Any]) -> List[TimelineEvent]:
        """Query events with filters"""
        conditions = []
        params = []

        if 'volume_id' in filters:
            conditions.append("volume_id = %s")
            params.append(filters['volume_id'])

        if 'event_type' in filters:
            conditions.append("event_type = %s")
            params.append(filters['event_type'])

        if 'importance_min' in filters:
            conditions.append("importance_score >= %s")
            params.append(filters['importance_min'])

        if 'character' in filters:
            conditions.append("%s = ANY(primary_actors) OR %s = ANY(affected_characters)")
            params.extend([filters['character'], filters['character']])

        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        query = f"SELECT * FROM timeline_events WHERE {where_clause} ORDER BY chronological_order"

        try:
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            return [TimelineEvent(**dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Query events failed: {e}")
            return []

    # ========================================================================
    # CAUSALITY OPERATIONS
    # ========================================================================

    async def store_causal_link(self, link: CausalLink) -> bool:
        """Store causal relationship"""
        try:
            self.cursor.execute("""
                INSERT INTO event_causality (
                    cause_event_id, effect_event_id, causality_type, strength, reasoning, confidence
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (cause_event_id, effect_event_id) DO UPDATE SET
                    strength = EXCLUDED.strength,
                    reasoning = EXCLUDED.reasoning
            """, (link.from_event, link.to_event, link.causality_type, link.strength, link.reasoning, link.confidence))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to store causal link: {e}")
            return False

    async def query_causality_chain(self, start_event: str, end_event: str, max_depth: int = 5) -> List[Dict[str, Any]]:
        """
        Find causality chain in PostgreSQL (LIMITED - use Neo4j for better performance)
        NOTE: This is O(n) performance. Neo4j is O(log n) - 50-100x faster!
        """
        logger.warning("⚠️ PostgreSQL causality queries are slow. Consider using Neo4j for 50-100x speedup!")

        try:
            # Simple 1-hop causality check (multi-hop is too expensive in PostgreSQL)
            self.cursor.execute("""
                SELECT cause_event_id, effect_event_id, causality_type, strength, reasoning
                FROM event_causality
                WHERE cause_event_id = %s AND effect_event_id = %s
            """, (start_event, end_event))

            result = self.cursor.fetchone()
            if result:
                return [dict(result)]
            return []
        except Exception as e:
            logger.error(f"Causality query failed: {e}")
            return []

    async def get_event_causes(self, event_id: str) -> List[str]:
        """Get events that caused this event"""
        try:
            self.cursor.execute("SELECT caused_by_events FROM timeline_events WHERE event_id = %s", (event_id,))
            row = self.cursor.fetchone()
            return row['caused_by_events'] if row and row['caused_by_events'] else []
        except Exception as e:
            logger.error(f"Failed to get causes for {event_id}: {e}")
            return []

    async def get_event_consequences(self, event_id: str) -> List[str]:
        """Get events caused by this event"""
        try:
            self.cursor.execute("SELECT causes_events FROM timeline_events WHERE event_id = %s", (event_id,))
            row = self.cursor.fetchone()
            return row['causes_events'] if row and row['causes_events'] else []
        except Exception as e:
            logger.error(f"Failed to get consequences for {event_id}: {e}")
            return []

    # ========================================================================
    # CHARACTER OPERATIONS
    # ========================================================================

    async def store_character(self, character: CharacterData) -> bool:
        """Store character to PostgreSQL"""
        try:
            self.cursor.execute("""
                INSERT INTO character_registry (
                    character_id, name, volume_id, batch_id, character_type,
                    aliases, personality_traits, first_appearance, confidence_score
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (character_id) DO UPDATE SET
                    aliases = EXCLUDED.aliases,
                    personality_traits = EXCLUDED.personality_traits,
                    confidence_score = GREATEST(character_registry.confidence_score, EXCLUDED.confidence_score)
            """, (
                character.character_id, character.name, character.volume_id, character.batch_id,
                character.character_type, character.aliases or [], character.personality_traits or [],
                character.first_appearance, character.confidence_score
            ))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to store character {character.character_id}: {e}")
            return False

    async def get_character(self, character_id: str) -> Optional[CharacterData]:
        """Retrieve character from PostgreSQL"""
        try:
            self.cursor.execute("SELECT * FROM character_registry WHERE character_id = %s", (character_id,))
            row = self.cursor.fetchone()
            if row:
                return CharacterData(**dict(row))
            return None
        except Exception as e:
            logger.error(f"Failed to get character {character_id}: {e}")
            return None

    async def query_character_events(self, character_name: str, volume_id: Optional[int] = None) -> List[TimelineEvent]:
        """Get all events involving a character"""
        query = """
            SELECT * FROM timeline_events
            WHERE %s = ANY(primary_actors) OR %s = ANY(affected_characters)
        """
        params = [character_name, character_name]

        if volume_id is not None:
            query += " AND volume_id = %s"
            params.append(volume_id)

        query += " ORDER BY chronological_order"

        try:
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            return [TimelineEvent(**dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to query character events: {e}")
            return []

    async def get_character_relationships(self, character_id: str) -> List[Dict[str, Any]]:
        """Get character relationships (basic implementation)"""
        # TODO: Implement proper relationship tracking
        logger.warning("Character relationships not fully implemented in PostgreSQL adapter")
        return []

    # ========================================================================
    # BATCH OPERATIONS
    # ========================================================================

    async def store_events_batch(self, events: List[TimelineEvent]) -> int:
        """Store multiple events in batch"""
        success_count = 0
        for event in events:
            if await self.store_event(event):
                success_count += 1
        return success_count

    async def store_causal_links_batch(self, links: List[CausalLink]) -> int:
        """Store multiple causal links in batch"""
        success_count = 0
        for link in links:
            if await self.store_causal_link(link):
                success_count += 1
        return success_count
