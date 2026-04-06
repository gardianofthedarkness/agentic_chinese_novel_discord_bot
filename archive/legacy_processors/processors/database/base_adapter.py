#!/usr/bin/env python3
"""
Base Database Adapter - Abstract interface for all database backends
Provides unified API for PostgreSQL, Neo4j, and SQLite
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class DatabaseBackend(Enum):
    """Available database backends"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    NEO4J = "neo4j"
    HYBRID = "hybrid"  # Smart routing across multiple backends


@dataclass
class TimelineEvent:
    """Timeline event data structure (unified across all backends)"""
    event_id: str
    volume_id: int
    batch_id: int
    description: str
    event_type: str
    importance_score: float
    chronological_order: Optional[int] = None
    primary_actors: Optional[List[str]] = None
    affected_characters: Optional[List[str]] = None
    caused_by_events: Optional[List[str]] = None
    causes_events: Optional[List[str]] = None
    temporal_markers: Optional[List[str]] = None
    confidence_level: float = 0.5
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            'event_id': self.event_id,
            'volume_id': self.volume_id,
            'batch_id': self.batch_id,
            'description': self.description,
            'event_type': self.event_type,
            'importance_score': self.importance_score,
            'chronological_order': self.chronological_order,
            'primary_actors': self.primary_actors or [],
            'affected_characters': self.affected_characters or [],
            'caused_by_events': self.caused_by_events or [],
            'causes_events': self.causes_events or [],
            'temporal_markers': self.temporal_markers or [],
            'confidence_level': self.confidence_level,
            'created_at': self.created_at or datetime.now()
        }


@dataclass
class CausalLink:
    """Causal relationship between events"""
    from_event: str
    to_event: str
    causality_type: str
    strength: float
    reasoning: str
    confidence: float = 0.5


@dataclass
class CharacterData:
    """Character information"""
    character_id: str
    name: str
    volume_id: int
    batch_id: int
    character_type: str
    aliases: Optional[List[str]] = None
    personality_traits: Optional[List[str]] = None
    first_appearance: Optional[int] = None
    confidence_score: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'character_id': self.character_id,
            'name': self.name,
            'volume_id': self.volume_id,
            'batch_id': self.batch_id,
            'character_type': self.character_type,
            'aliases': self.aliases or [],
            'personality_traits': self.personality_traits or [],
            'first_appearance': self.first_appearance,
            'confidence_score': self.confidence_score
        }


class BaseDatabaseAdapter(ABC):
    """
    Abstract base class for database adapters
    All database backends must implement these methods
    """

    def __init__(self, config: Any):
        """
        Initialize database adapter

        Args:
            config: Configuration object with database settings
        """
        self.config = config
        self.connected = False

    # ========================================================================
    # CONNECTION MANAGEMENT
    # ========================================================================

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish database connection

        Returns:
            bool: True if connection successful
        """
        pass

    @abstractmethod
    def disconnect(self):
        """Close database connection"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if database is connected"""
        pass

    # ========================================================================
    # SCHEMA MANAGEMENT
    # ========================================================================

    @abstractmethod
    def initialize_schema(self):
        """Create tables/schema if not exists"""
        pass

    @abstractmethod
    def verify_schema(self) -> bool:
        """Verify schema exists and is correct"""
        pass

    # ========================================================================
    # EVENT OPERATIONS
    # ========================================================================

    @abstractmethod
    async def store_event(self, event: TimelineEvent) -> bool:
        """
        Store a timeline event

        Args:
            event: TimelineEvent object

        Returns:
            bool: True if stored successfully
        """
        pass

    @abstractmethod
    async def get_event(self, event_id: str) -> Optional[TimelineEvent]:
        """
        Retrieve an event by ID

        Args:
            event_id: Event identifier

        Returns:
            TimelineEvent or None if not found
        """
        pass

    @abstractmethod
    async def query_events(self, filters: Dict[str, Any]) -> List[TimelineEvent]:
        """
        Query events with filters

        Args:
            filters: Dict of filter conditions
                - volume_id: int
                - event_type: str
                - importance_min: float
                - character: str

        Returns:
            List of matching TimelineEvent objects
        """
        pass

    # ========================================================================
    # CAUSALITY OPERATIONS
    # ========================================================================

    @abstractmethod
    async def store_causal_link(self, link: CausalLink) -> bool:
        """
        Store a causal relationship between events

        Args:
            link: CausalLink object

        Returns:
            bool: True if stored successfully
        """
        pass

    @abstractmethod
    async def query_causality_chain(self, start_event: str, end_event: str,
                                   max_depth: int = 5) -> List[Dict[str, Any]]:
        """
        Find causality chain between two events

        Args:
            start_event: Starting event ID
            end_event: Target event ID
            max_depth: Maximum chain length

        Returns:
            List of causal paths (implementation-specific format)
        """
        pass

    @abstractmethod
    async def get_event_causes(self, event_id: str) -> List[str]:
        """
        Get all events that caused this event

        Args:
            event_id: Target event ID

        Returns:
            List of event IDs that caused this event
        """
        pass

    @abstractmethod
    async def get_event_consequences(self, event_id: str) -> List[str]:
        """
        Get all events caused by this event

        Args:
            event_id: Source event ID

        Returns:
            List of event IDs caused by this event
        """
        pass

    # ========================================================================
    # CHARACTER OPERATIONS
    # ========================================================================

    @abstractmethod
    async def store_character(self, character: CharacterData) -> bool:
        """
        Store character information

        Args:
            character: CharacterData object

        Returns:
            bool: True if stored successfully
        """
        pass

    @abstractmethod
    async def get_character(self, character_id: str) -> Optional[CharacterData]:
        """
        Retrieve character by ID

        Args:
            character_id: Character identifier

        Returns:
            CharacterData or None if not found
        """
        pass

    @abstractmethod
    async def query_character_events(self, character_name: str,
                                    volume_id: Optional[int] = None) -> List[TimelineEvent]:
        """
        Get all events involving a character

        Args:
            character_name: Character name
            volume_id: Optional volume filter

        Returns:
            List of TimelineEvent objects
        """
        pass

    @abstractmethod
    async def get_character_relationships(self, character_id: str) -> List[Dict[str, Any]]:
        """
        Get relationships for a character

        Args:
            character_id: Character identifier

        Returns:
            List of relationship data
        """
        pass

    # ========================================================================
    # BATCH OPERATIONS (for performance)
    # ========================================================================

    @abstractmethod
    async def store_events_batch(self, events: List[TimelineEvent]) -> int:
        """
        Store multiple events in batch (optimized)

        Args:
            events: List of TimelineEvent objects

        Returns:
            Number of events successfully stored
        """
        pass

    @abstractmethod
    async def store_causal_links_batch(self, links: List[CausalLink]) -> int:
        """
        Store multiple causal links in batch

        Args:
            links: List of CausalLink objects

        Returns:
            Number of links successfully stored
        """
        pass

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_backend_type(self) -> DatabaseBackend:
        """Return the backend type"""
        return self._backend_type

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
