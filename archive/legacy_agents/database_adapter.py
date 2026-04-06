#!/usr/bin/env python3
"""
Database Adapter
Provides unified interface for both SQLite (development) and PostgreSQL (production)
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

# Database-specific imports
try:
    import asyncpg
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

import sqlite3
import aiosqlite

from timeline_models import TimelineNode, CharacterState, StoryEvent, UserCharacter

logger = logging.getLogger(__name__)


class DatabaseAdapter(ABC):
    """Abstract base class for database adapters"""
    
    @abstractmethod
    async def initialize(self):
        """Initialize database connection and schema"""
        pass
    
    @abstractmethod
    async def save_timeline_node(self, node: TimelineNode) -> bool:
        """Save timeline node to database"""
        pass
    
    @abstractmethod
    async def load_timeline_node(self, node_id: str) -> Optional[TimelineNode]:
        """Load timeline node from database"""
        pass
    
    @abstractmethod
    async def close(self):
        """Close database connections"""
        pass


class SQLiteAdapter(DatabaseAdapter):
    """SQLite adapter for development and single-instance deployments"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.initialized = False
    
    async def initialize(self):
        """Initialize SQLite database"""
        if self.initialized:
            return
        
        # Create directory if needed
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        async with aiosqlite.connect(self.db_path) as db:
            # Create tables (same as timeline_database.py)
            await db.execute('''
                CREATE TABLE IF NOT EXISTS timeline_nodes (
                    id TEXT PRIMARY KEY,
                    parent_id TEXT,
                    channel_id TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    data TEXT NOT NULL  -- JSON blob with all node data
                )
            ''')
            
            await db.execute('''
                CREATE TABLE IF NOT EXISTS character_states (
                    id TEXT PRIMARY KEY,
                    node_id TEXT NOT NULL,
                    character_name TEXT NOT NULL,
                    data TEXT NOT NULL,  -- JSON blob with character state
                    FOREIGN KEY (node_id) REFERENCES timeline_nodes(id)
                )
            ''')
            
            await db.execute('''
                CREATE TABLE IF NOT EXISTS user_characters (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    character_name TEXT NOT NULL,
                    data TEXT NOT NULL,  -- JSON blob with user character data
                    UNIQUE(user_id, character_name)
                )
            ''')
            
            # Create indexes
            await db.execute('CREATE INDEX IF NOT EXISTS idx_nodes_channel ON timeline_nodes(channel_id)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_nodes_parent ON timeline_nodes(parent_id)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_characters_node ON character_states(node_id)')
            
            await db.commit()
        
        self.initialized = True
        logger.info(f"✅ SQLite database initialized at {self.db_path}")
    
    async def save_timeline_node(self, node: TimelineNode) -> bool:
        """Save timeline node to SQLite"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Save node as JSON blob for simplicity
                node_data = node.json()
                
                await db.execute('''
                    INSERT OR REPLACE INTO timeline_nodes (id, parent_id, channel_id, created_at, data)
                    VALUES (?, ?, ?, ?, ?)
                ''', (node.id, node.parent_id, node.channel_id, node.created_at, node_data))
                
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"❌ Error saving timeline node to SQLite: {e}")
            return False
    
    async def load_timeline_node(self, node_id: str) -> Optional[TimelineNode]:
        """Load timeline node from SQLite"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute('SELECT data FROM timeline_nodes WHERE id = ?', (node_id,))
                row = await cursor.fetchone()
                
                if not row:
                    return None
                
                # Parse JSON data back to TimelineNode
                return TimelineNode.parse_raw(row[0])
                
        except Exception as e:
            logger.error(f"❌ Error loading timeline node from SQLite: {e}")
            return None
    
    async def close(self):
        """Close SQLite connections (no persistent connections)"""
        pass


class PostgreSQLAdapter(DatabaseAdapter):
    """PostgreSQL adapter for production deployments"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool: Optional[asyncpg.Pool] = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize PostgreSQL connection pool"""
        if not POSTGRES_AVAILABLE:
            raise ImportError("asyncpg not available. Install with: pip install asyncpg")
        
        if self.initialized:
            return
        
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=2,
                max_size=10,
                command_timeout=30
            )
            
            # Create tables
            async with self.pool.acquire() as conn:
                # Timeline nodes table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS timeline_nodes (
                        id TEXT PRIMARY KEY,
                        parent_id TEXT,
                        channel_id TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        data JSONB NOT NULL
                    )
                ''')
                
                # Character states table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS character_states (
                        id TEXT PRIMARY KEY,
                        node_id TEXT NOT NULL,
                        character_name TEXT NOT NULL,
                        data JSONB NOT NULL,
                        FOREIGN KEY (node_id) REFERENCES timeline_nodes(id) ON DELETE CASCADE
                    )
                ''')
                
                # User characters table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS user_characters (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        character_name TEXT NOT NULL,
                        data JSONB NOT NULL,
                        UNIQUE(user_id, character_name)
                    )
                ''')
                
                # Create indexes
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_nodes_channel ON timeline_nodes(channel_id)')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_nodes_parent ON timeline_nodes(parent_id)')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_nodes_created ON timeline_nodes(created_at)')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_characters_node ON character_states(node_id)')
                
                # JSON indexes for better query performance
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_nodes_data_universe ON timeline_nodes USING GIN ((data->\'universe_fingerprint\'))')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_nodes_data_location ON timeline_nodes USING GIN ((data->\'chapter_context\'->\'setting\'))')
            
            self.initialized = True
            logger.info("✅ PostgreSQL database initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing PostgreSQL: {e}")
            raise
    
    async def save_timeline_node(self, node: TimelineNode) -> bool:
        """Save timeline node to PostgreSQL"""
        try:
            if not self.pool:
                await self.initialize()
            
            async with self.pool.acquire() as conn:
                node_data = node.dict()
                
                await conn.execute('''
                    INSERT INTO timeline_nodes (id, parent_id, channel_id, created_at, data)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (id) DO UPDATE SET
                        parent_id = EXCLUDED.parent_id,
                        channel_id = EXCLUDED.channel_id,
                        created_at = EXCLUDED.created_at,
                        data = EXCLUDED.data
                ''', node.id, node.parent_id, node.channel_id, node.created_at, node_data)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving timeline node to PostgreSQL: {e}")
            return False
    
    async def load_timeline_node(self, node_id: str) -> Optional[TimelineNode]:
        """Load timeline node from PostgreSQL"""
        try:
            if not self.pool:
                await self.initialize()
            
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow('SELECT data FROM timeline_nodes WHERE id = $1', node_id)
                
                if not row:
                    return None
                
                return TimelineNode(**row['data'])
                
        except Exception as e:
            logger.error(f"❌ Error loading timeline node from PostgreSQL: {e}")
            return None
    
    async def find_similar_timelines(self, channel_id: str, universe_data: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar timelines using PostgreSQL JSON queries"""
        try:
            if not self.pool:
                await self.initialize()
            
            async with self.pool.acquire() as conn:
                # Use PostgreSQL JSON operators for similarity search
                rows = await conn.fetch('''
                    SELECT id, data->>'universe_fingerprint' as universe_data, created_at
                    FROM timeline_nodes 
                    WHERE channel_id = $1 
                    AND data->'universe_fingerprint'->>'current_location' = $2
                    ORDER BY created_at DESC 
                    LIMIT $3
                ''', channel_id, universe_data.get('current_location', ''), limit)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Error finding similar timelines: {e}")
            return []
    
    async def close(self):
        """Close PostgreSQL connection pool"""
        if self.pool:
            await self.pool.close()
            self.pool = None


def create_database_adapter(config: Optional[Dict[str, Any]] = None) -> DatabaseAdapter:
    """Factory function to create appropriate database adapter"""
    if config is None:
        config = {}
    
    # Check environment variables
    database_url = os.getenv('DATABASE_URL') or os.getenv('TIMELINE_DATABASE_URL')
    
    if database_url and database_url.startswith('postgresql'):
        # Use PostgreSQL
        logger.info("🐘 Using PostgreSQL database adapter")
        return PostgreSQLAdapter(database_url)
    else:
        # Use SQLite (default)
        db_path = config.get('db_path', 'character_data/timeline.db')
        logger.info(f"📁 Using SQLite database adapter: {db_path}")
        return SQLiteAdapter(db_path)


# Enhanced timeline database class that uses the adapter
class EnhancedTimelineDatabase:
    """Enhanced timeline database that works with both SQLite and PostgreSQL"""
    
    def __init__(self, adapter: Optional[DatabaseAdapter] = None):
        self.adapter = adapter or create_database_adapter()
    
    async def initialize(self):
        """Initialize the database adapter"""
        await self.adapter.initialize()
    
    async def save_timeline_node(self, node: TimelineNode) -> bool:
        """Save timeline node using the adapter"""
        return await self.adapter.save_timeline_node(node)
    
    async def load_timeline_node(self, node_id: str) -> Optional[TimelineNode]:
        """Load timeline node using the adapter"""
        return await self.adapter.load_timeline_node(node_id)
    
    async def close(self):
        """Close database connections"""
        await self.adapter.close()
    
    # Additional convenience methods
    async def save_user_character(self, user_character: UserCharacter) -> bool:
        """Save user character (implement based on adapter type)"""
        # This would delegate to adapter-specific implementation
        pass
    
    async def find_similar_timelines(self, channel_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar timelines (implement based on adapter type)"""
        if hasattr(self.adapter, 'find_similar_timelines'):
            return await self.adapter.find_similar_timelines(channel_id, context)
        else:
            # Fallback implementation for SQLite
            return []


# Example usage
async def test_adapters():
    """Test both database adapters"""
    from timeline_models import create_example_timeline
    
    # Test SQLite adapter
    sqlite_adapter = SQLiteAdapter("test_sqlite.db")
    await sqlite_adapter.initialize()
    
    example_node = create_example_timeline()
    success = await sqlite_adapter.save_timeline_node(example_node)
    print(f"SQLite save success: {success}")
    
    loaded_node = await sqlite_adapter.load_timeline_node(example_node.id)
    print(f"SQLite load success: {loaded_node is not None}")
    
    await sqlite_adapter.close()
    
    # Test PostgreSQL adapter (if available)
    if POSTGRES_AVAILABLE:
        try:
            pg_url = "postgresql://timeline_user:timeline_pass@localhost:5432/test_db"
            pg_adapter = PostgreSQLAdapter(pg_url)
            await pg_adapter.initialize()
            
            success = await pg_adapter.save_timeline_node(example_node)
            print(f"PostgreSQL save success: {success}")
            
            loaded_node = await pg_adapter.load_timeline_node(example_node.id)
            print(f"PostgreSQL load success: {loaded_node is not None}")
            
            await pg_adapter.close()
        except Exception as e:
            print(f"PostgreSQL test failed (expected): {e}")


if __name__ == "__main__":
    asyncio.run(test_adapters())