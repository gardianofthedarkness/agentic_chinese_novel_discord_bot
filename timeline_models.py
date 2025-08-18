#!/usr/bin/env python3
"""
Timeline System Data Models
Defines core data structures for timeline management, parallel universes, and story context
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid

from pydantic import BaseModel, Field, validator


class EventType(Enum):
    """Types of story events that can occur"""
    DIALOGUE = "dialogue"
    ACTION = "action"
    PLOT_POINT = "plot_point"
    DECISION = "decision"
    CHARACTER_INTRO = "character_intro"
    LOCATION_CHANGE = "location_change"
    RELATIONSHIP_CHANGE = "relationship_change"
    USER_JOIN = "user_join"
    USER_CHARACTER_ACTION = "user_character_action"


class CharacterType(Enum):
    """Types of characters in the story"""
    NPC = "npc"           # AI-controlled story characters
    USER = "user"         # User-controlled characters
    NARRATOR = "narrator" # System narrator
    OBSERVER = "observer" # User in observer mode


class StoryEvent(BaseModel):
    """Individual story event within a timeline node"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.now)
    summary: str = Field(..., description="Brief description of the event")
    characters_involved: List[str] = Field(default=[], description="Character names involved")
    user_ids_involved: List[str] = Field(default=[], description="Discord user IDs involved")
    impact_score: float = Field(default=1.0, description="Significance of event (0-10)")
    consequences: List[str] = Field(default=[], description="What this event led to")
    emotional_impact: Dict[str, str] = Field(default={}, description="How this affected character emotions")
    
    @validator('impact_score')
    def validate_impact_score(cls, v):
        return max(0.0, min(10.0, v))


class CharacterState(BaseModel):
    """State of a character at a specific timeline node"""
    character_name: str
    character_type: CharacterType
    user_id: Optional[str] = Field(None, description="Discord user ID if user character")
    
    # Emotional state
    emotional_state: str = Field("neutral", description="Current emotional state")
    energy_level: float = Field(0.7, description="Energy level 0-1")
    mood_stability: float = Field(0.5, description="How stable the mood is")
    
    # Location and context
    current_location: str = Field("", description="Where the character currently is")
    recent_actions: List[str] = Field(default=[], description="Recent actions taken")
    
    # Knowledge and relationships
    knowledge_items: List[str] = Field(default=[], description="Things the character knows")
    relationships: Dict[str, float] = Field(default={}, description="Relationship scores with other characters (-1 to 1)")
    secrets_known: List[str] = Field(default=[], description="Secrets the character is aware of")
    
    # Goals and motivations
    current_goals: List[str] = Field(default=[], description="What the character wants to achieve")
    motivations: List[str] = Field(default=[], description="Why the character acts")
    
    # Story integration
    story_role: str = Field("participant", description="Role in the current story arc")
    plot_importance: float = Field(0.5, description="How important to main plot (0-1)")
    
    # User character specific
    abilities: List[str] = Field(default=[], description="Special abilities or skills")
    inventory: List[str] = Field(default=[], description="Items the character has")
    experience_points: int = Field(0, description="Character progression")


class ChapterContext(BaseModel):
    """Context information for the current story chapter/scenario"""
    chapter_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field("", description="Chapter/scenario title")
    setting: str = Field("", description="Current setting/location")
    time_period: str = Field("", description="When this takes place")
    
    # Story elements
    main_plot_threads: List[str] = Field(default=[], description="Active main plot lines")
    sub_plot_threads: List[str] = Field(default=[], description="Active sub-plots")
    recent_events: List[str] = Field(default=[], description="Recent significant events")
    
    # World state
    world_state: Dict[str, Any] = Field(default={}, description="Current state of the world")
    available_locations: List[str] = Field(default=[], description="Locations characters can visit")
    active_npcs: List[str] = Field(default=[], description="NPCs currently available for interaction")
    
    # Themes and tone
    story_themes: List[str] = Field(default=[], description="Themes being explored")
    narrative_tone: str = Field("neutral", description="Overall tone of the story")
    genre_tags: List[str] = Field(default=[], description="Genre classification")
    
    # Constraints and rules
    story_rules: List[str] = Field(default=[], description="Rules governing this story world")
    character_limitations: Dict[str, List[str]] = Field(default={}, description="What characters cannot do")


class UniverseFingerprint(BaseModel):
    """Compact representation of universe state for similarity comparison"""
    fingerprint_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Core story elements (for matching)
    main_characters: Set[str] = Field(default=set(), description="Primary characters in this universe")
    current_location: str = Field("", description="Primary story location")
    story_phase: str = Field("", description="What phase of the story this is")
    
    # Relationship matrix (simplified)
    relationship_summary: Dict[str, Dict[str, float]] = Field(default={}, description="Simplified relationship matrix")
    
    # Plot state
    completed_plot_points: Set[str] = Field(default=set(), description="Major events that have occurred")
    active_conflicts: Set[str] = Field(default=set(), description="Current conflicts/tensions")
    
    # World state hash
    world_state_hash: str = Field("", description="Hash of world state for quick comparison")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    
    def calculate_hash(self) -> str:
        """Calculate a hash representing this universe state"""
        hash_data = {
            "characters": sorted(list(self.main_characters)),
            "location": self.current_location,
            "phase": self.story_phase,
            "plot_points": sorted(list(self.completed_plot_points)),
            "conflicts": sorted(list(self.active_conflicts))
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def similarity_score(self, other: 'UniverseFingerprint') -> float:
        """Calculate similarity score with another universe fingerprint (0-1)"""
        scores = []
        
        # Character overlap
        char_overlap = len(self.main_characters.intersection(other.main_characters))
        char_union = len(self.main_characters.union(other.main_characters))
        char_score = char_overlap / char_union if char_union > 0 else 0
        scores.append(char_score * 0.3)  # 30% weight
        
        # Location similarity
        location_score = 1.0 if self.current_location == other.current_location else 0.0
        scores.append(location_score * 0.2)  # 20% weight
        
        # Story phase similarity
        phase_score = 1.0 if self.story_phase == other.story_phase else 0.0
        scores.append(phase_score * 0.15)  # 15% weight
        
        # Plot point overlap
        plot_overlap = len(self.completed_plot_points.intersection(other.completed_plot_points))
        plot_union = len(self.completed_plot_points.union(other.completed_plot_points))
        plot_score = plot_overlap / plot_union if plot_union > 0 else 0
        scores.append(plot_score * 0.25)  # 25% weight
        
        # Conflict overlap
        conflict_overlap = len(self.active_conflicts.intersection(other.active_conflicts))
        conflict_union = len(self.active_conflicts.union(other.active_conflicts))
        conflict_score = conflict_overlap / conflict_union if conflict_union > 0 else 0
        scores.append(conflict_score * 0.1)  # 10% weight
        
        return sum(scores)


class TimelineNode(BaseModel):
    """A single node in the timeline tree representing a story state"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = Field(None, description="Parent node ID (None for root)")
    channel_id: str = Field(..., description="Discord channel this timeline belongs to")
    
    # Temporal information
    created_at: datetime = Field(default_factory=datetime.now)
    story_timestamp: Optional[datetime] = Field(None, description="When this occurs in story time")
    
    # Story context
    chapter_context: ChapterContext = Field(default_factory=ChapterContext)
    character_states: Dict[str, CharacterState] = Field(default={}, description="State of all characters")
    events: List[StoryEvent] = Field(default=[], description="Events that occurred at this node")
    
    # Universe identification
    universe_fingerprint: UniverseFingerprint = Field(default_factory=UniverseFingerprint)
    
    # Branching information
    children: List[str] = Field(default=[], description="Child node IDs")
    branch_reason: Optional[str] = Field(None, description="Why this branch was created")
    branch_weight: float = Field(1.0, description="How significant this branch is")
    
    # Conversation data
    conversation_summary: str = Field("", description="Summary of conversation at this node")
    message_count: int = Field(0, description="Number of messages in this conversation segment")
    participants: List[str] = Field(default=[], description="User IDs who participated")
    
    # Metadata
    is_milestone: bool = Field(False, description="Whether this is a significant story point")
    tags: List[str] = Field(default=[], description="Tags for categorization")
    notes: str = Field("", description="Additional notes about this node")
    
    def add_child(self, child_id: str):
        """Add a child node ID"""
        if child_id not in self.children:
            self.children.append(child_id)
    
    def remove_child(self, child_id: str):
        """Remove a child node ID"""
        if child_id in self.children:
            self.children.remove(child_id)
    
    def add_event(self, event: StoryEvent):
        """Add an event to this node"""
        self.events.append(event)
        
        # Update universe fingerprint based on event
        if event.event_type == EventType.CHARACTER_INTRO:
            self.universe_fingerprint.main_characters.update(event.characters_involved)
        elif event.event_type == EventType.PLOT_POINT:
            self.universe_fingerprint.completed_plot_points.add(event.summary)
        elif event.event_type == EventType.LOCATION_CHANGE and event.summary:
            self.universe_fingerprint.current_location = event.summary
        
        self.universe_fingerprint.last_updated = datetime.now()
        self.universe_fingerprint.world_state_hash = self.universe_fingerprint.calculate_hash()
    
    def update_character_state(self, character_name: str, state: CharacterState):
        """Update the state of a character at this node"""
        self.character_states[character_name] = state
        
        # Update universe fingerprint
        if state.plot_importance > 0.5:
            self.universe_fingerprint.main_characters.add(character_name)
        
        if state.current_location:
            self.universe_fingerprint.current_location = state.current_location
        
        self.universe_fingerprint.last_updated = datetime.now()
        self.universe_fingerprint.world_state_hash = self.universe_fingerprint.calculate_hash()
    
    def get_character_names(self) -> List[str]:
        """Get all character names present at this node"""
        return list(self.character_states.keys())
    
    def get_user_characters(self) -> List[str]:
        """Get names of user-controlled characters"""
        return [
            name for name, state in self.character_states.items() 
            if state.character_type == CharacterType.USER
        ]
    
    def get_npc_characters(self) -> List[str]:
        """Get names of NPC characters"""
        return [
            name for name, state in self.character_states.items() 
            if state.character_type == CharacterType.NPC
        ]
    
    def calculate_divergence_score(self, other: 'TimelineNode') -> float:
        """Calculate how much this timeline has diverged from another (0-1, higher = more divergent)"""
        return 1.0 - self.universe_fingerprint.similarity_score(other.universe_fingerprint)
    
    def should_branch(self, threshold: float = 0.3) -> bool:
        """Determine if this node should create a new branch"""
        # Check if this is significantly different from parent
        if not self.parent_id:
            return False
        
        # Would need parent node to compare - this would be handled by TimelineManager
        return False
    
    def export_summary(self) -> Dict[str, Any]:
        """Export a summary of this node for external use"""
        return {
            "id": self.id,
            "chapter_title": self.chapter_context.title,
            "location": self.chapter_context.setting,
            "characters": self.get_character_names(),
            "user_characters": self.get_user_characters(),
            "event_count": len(self.events),
            "conversation_summary": self.conversation_summary,
            "created_at": self.created_at.isoformat(),
            "is_milestone": self.is_milestone,
            "tags": self.tags
        }


class UserCharacter(BaseModel):
    """User-created character for roleplay"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(..., description="Discord user ID who owns this character")
    character_name: str = Field(..., description="Character's name")
    
    # Basic character info
    personality: str = Field(..., description="Character personality description")
    background: str = Field("", description="Character's backstory")
    appearance: str = Field("", description="Physical description")
    
    # Roleplay mechanics
    abilities: List[str] = Field(default=[], description="Special abilities or skills")
    limitations: List[str] = Field(default=[], description="Character limitations or weaknesses")
    goals: List[str] = Field(default=[], description="Character's goals in the story")
    motivations: List[str] = Field(default=[], description="What drives this character")
    
    # Story integration
    integration_method: str = Field("natural", description="How character enters story (natural/dramatic/gradual)")
    story_role: str = Field("participant", description="Role in story (main/supporting/background)")
    plot_hooks: List[str] = Field(default=[], description="Connections to main plot")
    
    # Relationships
    initial_relationships: Dict[str, str] = Field(default={}, description="Starting relationships with other characters")
    relationship_preferences: Dict[str, float] = Field(default={}, description="Preferred relationship types")
    
    # Character progression
    experience_gained: int = Field(0, description="Experience points earned")
    character_growth: List[str] = Field(default=[], description="How character has developed")
    major_achievements: List[str] = Field(default=[], description="Significant character moments")
    
    # Meta information
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)
    is_active: bool = Field(True, description="Whether character is currently active")
    
    # User preferences
    roleplay_style: str = Field("immersive", description="How user likes to roleplay")
    communication_style: str = Field("first_person", description="First person vs third person")
    comfort_level: str = Field("moderate", description="Comfort with different story content")
    
    def validate_for_story(self, chapter_context: ChapterContext) -> Dict[str, Any]:
        """Validate if this character fits the current story context"""
        validation = {
            "is_valid": True,
            "warnings": [],
            "suggestions": []
        }
        
        # Check if character fits the genre
        if chapter_context.genre_tags:
            # This would contain more sophisticated validation logic
            pass
        
        # Check if abilities are appropriate
        if "magic" in self.abilities and "magic" not in chapter_context.story_themes:
            validation["warnings"].append("Character has magic abilities but story may not have magic")
        
        # Check if background conflicts with world state
        if self.background and chapter_context.world_state:
            # Would implement background consistency checking
            pass
        
        return validation
    
    def generate_integration_suggestions(self, timeline_node: TimelineNode) -> List[str]:
        """Generate suggestions for how this character could integrate into current story"""
        suggestions = []
        
        current_location = timeline_node.chapter_context.setting
        active_characters = timeline_node.get_character_names()
        
        # Suggest integration methods based on current context
        if self.integration_method == "natural":
            suggestions.append(f"Character could already be in {current_location} and encounter the group naturally")
        elif self.integration_method == "dramatic":
            suggestions.append("Character could arrive at a crucial moment to help or complicate the situation")
        
        # Suggest relationship connections
        for char_name in active_characters:
            if char_name in self.initial_relationships:
                rel_type = self.initial_relationships[char_name]
                suggestions.append(f"Character has a {rel_type} relationship with {char_name}")
        
        return suggestions
    
    def export_for_ai(self) -> str:
        """Export character info in format suitable for AI character simulation"""
        return f"""Character: {self.character_name}
Personality: {self.personality}
Background: {self.background}
Goals: {', '.join(self.goals)}
Abilities: {', '.join(self.abilities)}
Roleplay Style: {self.roleplay_style}
Communication: {self.communication_style}"""


# Utility functions for timeline management

def create_root_node(channel_id: str, initial_context: ChapterContext) -> TimelineNode:
    """Create the root node for a new timeline"""
    root = TimelineNode(
        channel_id=channel_id,
        chapter_context=initial_context,
        is_milestone=True,
        tags=["root", "story_start"]
    )
    
    root.universe_fingerprint.story_phase = "beginning"
    root.universe_fingerprint.current_location = initial_context.setting
    root.universe_fingerprint.world_state_hash = root.universe_fingerprint.calculate_hash()
    
    return root


def calculate_branch_similarity(node1: TimelineNode, node2: TimelineNode) -> float:
    """Calculate similarity between two timeline nodes"""
    return node1.universe_fingerprint.similarity_score(node2.universe_fingerprint)


def find_optimal_branch_point(timeline_nodes: List[TimelineNode], target_context: ChapterContext) -> Optional[str]:
    """Find the best existing node to branch from for a given context"""
    if not timeline_nodes:
        return None
    
    best_score = 0.0
    best_node_id = None
    
    for node in timeline_nodes:
        # Calculate context similarity (simplified)
        score = 0.0
        
        # Location match
        if node.chapter_context.setting == target_context.setting:
            score += 0.3
        
        # Theme overlap
        theme_overlap = len(set(node.chapter_context.story_themes).intersection(set(target_context.story_themes)))
        if target_context.story_themes:
            score += (theme_overlap / len(target_context.story_themes)) * 0.2
        
        # Character overlap
        target_chars = set(target_context.active_npcs)
        node_chars = set(node.get_character_names())
        char_overlap = len(target_chars.intersection(node_chars))
        if target_chars:
            score += (char_overlap / len(target_chars)) * 0.3
        
        # Recency bonus (prefer more recent nodes)
        time_diff = (datetime.now() - node.created_at).total_seconds()
        recency_bonus = max(0, 1 - (time_diff / (24 * 3600)))  # Decay over 24 hours
        score += recency_bonus * 0.2
        
        if score > best_score:
            best_score = score
            best_node_id = node.id
    
    return best_node_id if best_score > 0.5 else None


# Example usage and testing
def create_example_timeline():
    """Create an example timeline for testing"""
    # Create initial chapter context
    chapter = ChapterContext(
        title="The Magic Academy",
        setting="Arcane University Library", 
        story_themes=["magic", "learning", "friendship"],
        genre_tags=["fantasy", "slice_of_life"],
        active_npcs=["Professor Aria", "Librarian Magnus"]
    )
    
    # Create root node
    root = create_root_node("test_channel", chapter)
    
    # Add some characters
    aria_state = CharacterState(
        character_name="Professor Aria",
        character_type=CharacterType.NPC,
        emotional_state="curious",
        current_location="Library",
        current_goals=["teach magic basics", "help students"],
        story_role="mentor"
    )
    
    root.update_character_state("Professor Aria", aria_state)
    
    # Add an event
    event = StoryEvent(
        event_type=EventType.CHARACTER_INTRO,
        summary="Professor Aria introduces herself to new students",
        characters_involved=["Professor Aria"],
        impact_score=3.0
    )
    
    root.add_event(event)
    
    return root


if __name__ == "__main__":
    # Test the models
    example_timeline = create_example_timeline()
    print(f"Created timeline node: {example_timeline.id}")
    print(f"Universe fingerprint: {example_timeline.universe_fingerprint.calculate_hash()}")
    print(f"Character count: {len(example_timeline.character_states)}")
    print(f"Event count: {len(example_timeline.events)}")