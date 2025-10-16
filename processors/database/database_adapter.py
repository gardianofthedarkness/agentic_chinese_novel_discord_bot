#!/usr/bin/env python3
"""
Database Adapter Coordinator
Manages multiple database backends and smart query routing
Implements hybrid PostgreSQL + Neo4j architecture from PERFORMANCE_OPTIMIZATION.md
"""

import logging
from typing import Dict, List, Any, Optional
from enum import Enum

from .base_adapter import (
    BaseDatabaseAdapter, DatabaseBackend,
    TimelineEvent, CausalLink, CharacterData
)
from .postgresql_adapter import PostgreSQLAdapter
from .neo4j_adapter import Neo4jAdapter

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of database queries for smart routing"""
    CAUSALITY_CHAINS = "causality_chains"           # → Neo4j (50x speedup)
    CHARACTER_RELATIONSHIPS = "character_relationships"  # → Neo4j (25x speedup)
    TIMELINE_TRAVERSAL = "timeline_traversal"       # → Neo4j (15x speedup)
    EVENT_STORAGE = "event_storage"                 # → Both (dual write)
    CHARACTER_STATS = "character_stats"             # → PostgreSQL (aggregations)
    FULL_TEXT_SEARCH = "full_text_search"           # → PostgreSQL
    BATCH_OPERATIONS = "batch_operations"           # → Both (optimized)


class DatabaseAdapter:
    """
    Multi-backend database coordinator with smart query routing

    Features:
    - Automatic routing to optimal database per query type
    - Dual-write mode for data consistency
    - Graceful fallback to PostgreSQL if Neo4j unavailable
    - Performance monitoring and metrics
    """

    # Query routing matrix from PERFORMANCE_OPTIMIZATION.md
    ROUTING_MATRIX = {
        QueryType.CAUSALITY_CHAINS: ('neo4j', 'postgresql', 50),
        QueryType.CHARACTER_RELATIONSHIPS: ('neo4j', 'postgresql', 25),
        QueryType.TIMELINE_TRAVERSAL: ('neo4j', 'postgresql', 15),
        QueryType.CHARACTER_STATS: ('postgresql', 'neo4j', 1),
        QueryType.FULL_TEXT_SEARCH: ('postgresql', 'neo4j', 1),
        QueryType.EVENT_STORAGE: ('both', 'postgresql', 1),
        QueryType.BATCH_OPERATIONS: ('both', 'postgresql', 1),
    }

    def __init__(self, config: Any):
        """
        Initialize database adapter with multi-backend support

        Args:
            config: ProcessingConfig with database settings
        """
        self.config = config
        self.postgres = None
        self.neo4j = None
        self.active_backends = []

        # Performance metrics
        self.query_counts = {qt: 0 for qt in QueryType}
        self.fallback_counts = 0

        # Initialize backends based on config
        self._initialize_backends()

    def _initialize_backends(self):
        """Initialize database backends based on configuration"""
        # Always initialize PostgreSQL (baseline)
        if self.config.use_postgres:
            try:
                self.postgres = PostgreSQLAdapter(self.config)
                if self.postgres.connect():
                    self.postgres.initialize_schema()
                    self.active_backends.append('postgresql')
                    logger.info("✅ PostgreSQL backend active")
            except Exception as e:
                logger.error(f"❌ PostgreSQL initialization failed: {e}")

        # Initialize Neo4j if enabled
        if self.config.use_neo4j:
            try:
                self.neo4j = Neo4jAdapter(self.config)
                if self.neo4j.connect():
                    self.neo4j.initialize_schema()
                    self.active_backends.append('neo4j')
                    logger.info("✅ Neo4j backend active")
            except Exception as e:
                logger.error(f"❌ Neo4j initialization failed: {e}")
                if not self.config.fallback_to_postgres:
                    raise

        if not self.active_backends:
            raise RuntimeError("No database backends available!")

        logger.info(f"🗄️ Active backends: {', '.join(self.active_backends)}")

    def _route_query(self, query_type: QueryType) -> BaseDatabaseAdapter:
        """
        Smart query routing based on ROUTING_MATRIX

        Args:
            query_type: Type of query to route

        Returns:
            Optimal database adapter for this query
        """
        primary_db, fallback_db, expected_speedup = self.ROUTING_MATRIX[query_type]

        # Track query type
        self.query_counts[query_type] += 1

        # Route to primary database
        if primary_db == 'neo4j' and self.neo4j:
            if expected_speedup > 1:
                logger.debug(f"🚀 Routing {query_type.value} to Neo4j (expected {expected_speedup}x speedup)")
            return self.neo4j
        elif primary_db == 'postgresql' and self.postgres:
            return self.postgres
        elif primary_db == 'both':
            # Prefer Neo4j for dual-write when available
            return self.neo4j if self.neo4j else self.postgres

        # Fallback
        self.fallback_counts += 1
        logger.warning(f"⚠️ Primary DB '{primary_db}' unavailable, using fallback '{fallback_db}'")

        if fallback_db == 'postgresql' and self.postgres:
            return self.postgres
        elif fallback_db == 'neo4j' and self.neo4j:
            return self.neo4j
        else:
            raise RuntimeError(f"No database available for {query_type}")

    # ========================================================================
    # EVENT OPERATIONS (with smart routing)
    # ========================================================================

    async def store_event(self, event: TimelineEvent) -> bool:
        """
        Store event with dual-write strategy

        Strategy:
        - Write to both PostgreSQL and Neo4j for data consistency
        - Neo4j provides fast causality queries
        - PostgreSQL provides backup and aggregation capabilities
        """
        success = True

        # Write to Neo4j if available (primary)
        if self.neo4j:
            if not await self.neo4j.store_event(event):
                logger.warning(f"⚠️ Neo4j write failed for event {event.event_id}")
                success = False

        # Write to PostgreSQL (backup/fallback)
        if self.postgres and self.config.fallback_to_postgres:
            if not await self.postgres.store_event(event):
                logger.warning(f"⚠️ PostgreSQL write failed for event {event.event_id}")
                success = False

        return success

    async def get_event(self, event_id: str) -> Optional[TimelineEvent]:
        """Get event (try Neo4j first, fallback to PostgreSQL)"""
        # Try Neo4j first (faster)
        if self.neo4j:
            event = await self.neo4j.get_event(event_id)
            if event:
                return event

        # Fallback to PostgreSQL
        if self.postgres:
            return await self.postgres.get_event(event_id)

        return None

    async def query_events(self, filters: Dict[str, Any]) -> List[TimelineEvent]:
        """Query events using optimal backend"""
        adapter = self._route_query(QueryType.TIMELINE_TRAVERSAL)
        return await adapter.query_events(filters)

    # ========================================================================
    # CAUSALITY OPERATIONS (⭐ Neo4j优势!)
    # ========================================================================

    async def store_causal_link(self, link: CausalLink) -> bool:
        """Store causal link with dual-write"""
        success = True

        if self.neo4j:
            if not await self.neo4j.store_causal_link(link):
                success = False

        if self.postgres and self.config.fallback_to_postgres:
            if not await self.postgres.store_causal_link(link):
                success = False

        return success

    async def query_causality_chain(self, start_event: str, end_event: str,
                                   max_depth: int = 5) -> List[Dict[str, Any]]:
        """
        🚀 Query causality chain - ROUTED TO NEO4J for 50-100x speedup!

        This is the killer feature of Neo4j integration.
        PostgreSQL: O(n²) complexity, 5-30 seconds
        Neo4j: O(log n) complexity, 0.05-0.15 seconds
        """
        adapter = self._route_query(QueryType.CAUSALITY_CHAINS)

        if adapter == self.neo4j:
            logger.info(f"🚀 Using Neo4j for causality query (50-100x faster than PostgreSQL)")
        else:
            logger.warning(f"⚠️ Using PostgreSQL for causality (slow! Consider enabling Neo4j)")

        return await adapter.query_causality_chain(start_event, end_event, max_depth)

    async def get_event_causes(self, event_id: str) -> List[str]:
        """Get event causes (prefer Neo4j)"""
        adapter = self._route_query(QueryType.CAUSALITY_CHAINS)
        return await adapter.get_event_causes(event_id)

    async def get_event_consequences(self, event_id: str) -> List[str]:
        """Get event consequences (prefer Neo4j)"""
        adapter = self._route_query(QueryType.CAUSALITY_CHAINS)
        return await adapter.get_event_consequences(event_id)

    # ========================================================================
    # CHARACTER OPERATIONS
    # ========================================================================

    async def store_character(self, character: CharacterData) -> bool:
        """Store character with dual-write"""
        success = True

        if self.neo4j:
            if not await self.neo4j.store_character(character):
                success = False

        if self.postgres:
            if not await self.postgres.store_character(character):
                success = False

        return success

    async def get_character(self, character_id: str) -> Optional[CharacterData]:
        """Get character (try Neo4j first)"""
        if self.neo4j:
            character = await self.neo4j.get_character(character_id)
            if character:
                return character

        if self.postgres:
            return await self.postgres.get_character(character_id)

        return None

    async def query_character_events(self, character_name: str,
                                    volume_id: Optional[int] = None) -> List[TimelineEvent]:
        """Query character events (prefer Neo4j for graph traversal)"""
        adapter = self._route_query(QueryType.TIMELINE_TRAVERSAL)
        return await adapter.query_character_events(character_name, volume_id)

    async def get_character_relationships(self, character_id: str) -> List[Dict[str, Any]]:
        """
        🚀 Get character relationships - ROUTED TO NEO4J for 25x speedup!

        This query is 25x faster in Neo4j due to native graph traversal
        """
        adapter = self._route_query(QueryType.CHARACTER_RELATIONSHIPS)

        if adapter == self.neo4j:
            logger.info(f"🚀 Using Neo4j for character relationships (25x faster)")

        return await adapter.get_character_relationships(character_id)

    # ========================================================================
    # BATCH OPERATIONS
    # ========================================================================

    async def store_events_batch(self, events: List[TimelineEvent]) -> int:
        """Store events in batch (dual-write optimized)"""
        total_stored = 0

        if self.neo4j:
            neo4j_count = await self.neo4j.store_events_batch(events)
            logger.info(f"✅ Neo4j stored {neo4j_count}/{len(events)} events")
            total_stored = max(total_stored, neo4j_count)

        if self.postgres and self.config.fallback_to_postgres:
            pg_count = await self.postgres.store_events_batch(events)
            logger.info(f"✅ PostgreSQL stored {pg_count}/{len(events)} events")
            total_stored = max(total_stored, pg_count)

        return total_stored

    async def store_causal_links_batch(self, links: List[CausalLink]) -> int:
        """Store causal links in batch (dual-write optimized)"""
        total_stored = 0

        if self.neo4j:
            neo4j_count = await self.neo4j.store_causal_links_batch(links)
            logger.info(f"✅ Neo4j stored {neo4j_count}/{len(links)} causal links")
            total_stored = max(total_stored, neo4j_count)

        if self.postgres and self.config.fallback_to_postgres:
            pg_count = await self.postgres.store_causal_links_batch(links)
            logger.info(f"✅ PostgreSQL stored {pg_count}/{len(links)} causal links")
            total_stored = max(total_stored, pg_count)

        return total_stored

    # ========================================================================
    # ADVANCED NEO4J-ONLY QUERIES
    # ========================================================================

    async def get_character_influence_network(self, character_name: str,
                                             depth: int = 2) -> Dict[str, Any]:
        """
        🚀 Advanced graph query (Neo4j only)
        Returns character's sphere of influence using graph algorithms
        """
        if not self.neo4j:
            logger.warning("⚠️ Neo4j required for influence network queries")
            return {}

        return await self.neo4j.get_character_influence_network(character_name, depth)

    async def find_plot_bottlenecks(self, volume_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        🚀 Advanced graph query (Neo4j only)
        Identifies key plot points using centrality analysis
        """
        if not self.neo4j:
            logger.warning("⚠️ Neo4j required for plot bottleneck analysis")
            return []

        return await self.neo4j.find_plot_bottlenecks(volume_id, limit)

    # ========================================================================
    # UTILITY & MONITORING
    # ========================================================================

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'active_backends': self.active_backends,
            'query_counts': {qt.value: count for qt, count in self.query_counts.items()},
            'fallback_count': self.fallback_counts,
            'neo4j_available': self.neo4j is not None,
            'postgres_available': self.postgres is not None
        }

    def close(self):
        """Close all database connections"""
        if self.neo4j:
            self.neo4j.disconnect()
        if self.postgres:
            self.postgres.disconnect()
        logger.info("All database connections closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
