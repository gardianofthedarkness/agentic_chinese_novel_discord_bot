"""
Database Abstraction Layer for Unified Novel Processor
Supports multiple backends: PostgreSQL, Neo4j, SQLite
"""

from .base_adapter import BaseDatabaseAdapter, DatabaseBackend
from .postgresql_adapter import PostgreSQLAdapter
from .neo4j_adapter import Neo4jAdapter
from .database_adapter import DatabaseAdapter

__all__ = [
    'BaseDatabaseAdapter',
    'DatabaseBackend',
    'PostgreSQLAdapter',
    'Neo4jAdapter',
    'DatabaseAdapter'
]
