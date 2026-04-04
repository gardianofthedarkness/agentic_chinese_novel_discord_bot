"""
Data Models
===========
All shared data classes and enums in one place.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Graph node / relationship types
# ---------------------------------------------------------------------------

class GraphNodeType(Enum):
    EVENT = "Event"
    CHARACTER = "Character"
    LOCATION = "Location"
    CONCEPT = "Concept"
    VOLUME = "Volume"
    CHAPTER = "Chapter"
    DIALOGUE = "Dialogue"
    THOUGHT = "Thought"


class GraphRelationType(Enum):
    CAUSES = "CAUSES"
    INFLUENCES = "INFLUENCES"
    HAPPENS_BEFORE = "HAPPENS_BEFORE"
    HAPPENS_AFTER = "HAPPENS_AFTER"
    PARTICIPATES_IN = "PARTICIPATES_IN"
    AFFECTED_BY = "AFFECTED_BY"
    MENTIONS = "MENTIONS"
    KNOWS = "KNOWS"
    LOCATED_AT = "LOCATED_AT"
    BELONGS_TO = "BELONGS_TO"
    HAS_CHILD = "HAS_CHILD"
    SPEAKS = "SPEAKS"
    THINKS = "THINKS"
    NEXT = "NEXT"
    NEXT_SEQUENTIAL = "NEXT_SEQUENTIAL"
    FLASHBACK_TO = "FLASHBACK_TO"


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------

@dataclass
class TimelineEvent:
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
        return {
            "event_id": self.event_id,
            "volume_id": self.volume_id,
            "batch_id": self.batch_id,
            "description": self.description,
            "event_type": self.event_type,
            "importance_score": self.importance_score,
            "chronological_order": self.chronological_order,
            "primary_actors": self.primary_actors or [],
            "affected_characters": self.affected_characters or [],
            "caused_by_events": self.caused_by_events or [],
            "causes_events": self.causes_events or [],
            "temporal_markers": self.temporal_markers or [],
            "confidence_level": self.confidence_level,
            "created_at": (self.created_at or datetime.now()).isoformat(),
        }


@dataclass
class CausalLink:
    from_event: str
    to_event: str
    causality_type: str
    strength: float
    reasoning: str
    confidence: float = 0.5


@dataclass
class CharacterData:
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
        return {
            "character_id": self.character_id,
            "name": self.name,
            "volume_id": self.volume_id,
            "batch_id": self.batch_id,
            "character_type": self.character_type,
            "aliases": self.aliases or [],
            "personality_traits": self.personality_traits or [],
            "first_appearance": self.first_appearance,
            "confidence_score": self.confidence_score,
        }


@dataclass
class CharacterState:
    """Runtime state of a character during roleplay"""
    name: str
    personality: str
    background: str
    speech_patterns: List[str] = field(default_factory=list)
    current_emotions: Dict[str, float] = field(default_factory=dict)
    current_goals: List[str] = field(default_factory=list)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)

    def to_profile(self) -> Dict[str, Any]:
        return {
            "personality": self.personality,
            "background": self.background,
            "speech_patterns": self.speech_patterns,
            "current_emotions": self.current_emotions,
            "current_goals": self.current_goals,
        }


# ---------------------------------------------------------------------------
# LangGraph processing state
# ---------------------------------------------------------------------------

@dataclass
class ProcessingState:
    """State flowing through the LangGraph processing chain"""
    chapter_text: str
    chapter_index: int
    volume_id: Optional[int] = None

    events: List[Dict[str, Any]] = field(default_factory=list)
    characters: List[Dict[str, Any]] = field(default_factory=list)
    locations: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)

    processing_stage: str = "initial"
    current_step: str = ""
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    started_at: datetime = field(default_factory=datetime.now)
    finished_at: Optional[datetime] = None

    @property
    def processing_time(self) -> float:
        if self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return 0.0

    def finalize(self) -> None:
        self.finished_at = datetime.now()
        self.processing_stage = "completed"


# ---------------------------------------------------------------------------
# Graph node / edge containers
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    node_id: str
    node_type: GraphNodeType
    properties: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class GraphRelationship:
    from_node: str
    to_node: str
    relationship_type: GraphRelationType
    properties: Dict[str, Any]
    strength: float = 1.0
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class ProcessingMetrics:
    chapter_index: int
    processing_mode: str
    events_extracted: int = 0
    characters_found: int = 0
    relationships_created: int = 0
    locations_identified: int = 0
    processing_time_seconds: float = 0.0
    confidence_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chapter_index": self.chapter_index,
            "processing_mode": self.processing_mode,
            "events_extracted": self.events_extracted,
            "characters_found": self.characters_found,
            "relationships_created": self.relationships_created,
            "locations_identified": self.locations_identified,
            "processing_time_seconds": self.processing_time_seconds,
            "confidence_score": self.confidence_score,
        }
