#!/usr/bin/env python3
"""
Unified Novel Processor - Consolidation of All Processing Methods
Combines all previous processors into one unified, configurable system
"""

import os
import sys
import asyncio
import json
import time
import logging
import re
import sqlite3
try:
    import psycopg2
    import psycopg2.extras
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

# Setup UTF-8 environment
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepseek_integration import DeepSeekClient, create_deepseek_config
from qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unified_processing.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ProcessingMode(Enum):
    """Different processing modes available"""
    ITERATIVE = "iterative"           # Iterative refinement with early termination
    HIERARCHICAL = "hierarchical"     # Chapter-based hierarchical processing  
    COMPREHENSIVE = "comprehensive"   # Full depth analysis
    LIMITLESS = "limitless"          # Complete volume processing
    BATCH = "batch"                  # Batch processing mode

class ProcessingStage(Enum):
    """Novel processing stages"""
    BEGINNING = "beginning"
    EARLY = "early" 
    MIDDLE = "middle"
    CLIMAX = "climax"
    ENDING = "ending"

@dataclass
class ProcessingConfig:
    """Configuration for processing behavior"""
    mode: ProcessingMode = ProcessingMode.ITERATIVE
    max_iterations: int = 3
    satisfaction_threshold: float = 0.80
    min_improvement_threshold: float = 0.03
    early_termination_patience: int = 2
    batch_size: int = 5
    use_qdrant: bool = True
    # IMPORTANT: Uses QDRANT_URL environment variable from .env file
    # Default: http://localhost:32768/ (where test_novel2 collection exists)
    # To change: Update .env file, not this default value
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:32768/")
    collection_name: str = "test_novel2"
    database_path: str = "unified_results.db"
    # PostgreSQL settings (for containerized deployment)
    use_postgres: bool = False
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "novel_processing")
    postgres_user: str = os.getenv("POSTGRES_USER", "novel_user")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "novel_pass")
    
@dataclass 
class ProcessingContext:
    """Context information for intelligent processing"""
    batch_position: int
    total_batches: int
    processing_stage: ProcessingStage
    volume_id: int
    
    @property
    def progress_percentage(self) -> float:
        return (self.batch_position / self.total_batches) * 100 if self.total_batches > 0 else 0

@dataclass
class IterationResult:
    """Result of a single reasoning iteration"""
    iteration_num: int
    analysis: Dict
    confidence_scores: Dict
    satisfaction_level: float
    identified_issues: List[str]
    refinement_requests: List[str]
    reasoning_trace: List[str]
    improvement_score: float = 0.0
    tokens_used: int = 0
    
    @property
    def is_meaningful_improvement(self) -> bool:
        return self.improvement_score > 0.02

# ============================================================================
# TIMELINE SYSTEM DATA CLASSES
# ============================================================================

class EventType(Enum):
    """Types of timeline events"""
    DIALOGUE = "dialogue"
    ACTION = "action"
    REVELATION = "revelation"
    CONFLICT = "conflict"
    RESOLUTION = "resolution"
    INTERNAL_MONOLOGUE = "internal_monologue"
    ENVIRONMENTAL = "environmental"
    CHARACTER_INTRODUCTION = "character_introduction"
    PLOT_ADVANCEMENT = "plot_advancement"
    
class NarrativeFunction(Enum):
    """Narrative function of events in story structure"""
    SETUP = "setup"
    INCITING_INCIDENT = "inciting_incident"  
    RISING_ACTION = "rising_action"
    CLIMAX = "climax"
    FALLING_ACTION = "falling_action"
    RESOLUTION = "resolution"
    DENOUEMENT = "denouement"
    
class CausalityType(Enum):
    """Types of causal relationships between events"""
    DIRECT_CAUSE = "direct_cause"
    INDIRECT_CAUSE = "indirect_cause"
    ENABLING_CONDITION = "enabling_condition"
    PREVENTING_CONDITION = "preventing_condition"
    CATALYST = "catalyst"
    CONSEQUENCE = "consequence"
    
class ValidationStatus(Enum):
    """Status of event validation"""
    PENDING = "pending"
    VALIDATED = "validated"
    CONFLICTED = "conflicted"
    REJECTED = "rejected"

@dataclass 
class MotivationTrigger:
    """Character motivation trigger information"""
    character_name: str
    motivation_type: str            # "revenge", "protection", "curiosity", "duty"
    trigger_description: str
    motivation_strength: float      # 0.0-1.0
    duration: str                  # "immediate", "short_term", "ongoing"
    confidence: float = 0.5

@dataclass
class CharacterState:
    """Character state at a specific point in time"""
    emotional_state: str           # "angry", "confused", "determined"
    knowledge_level: Dict[str, float] = None  # What they know about various topics
    relationships: Dict[str, float] = None    # Relationship scores with other characters  
    goals: List[str] = None        # Current objectives
    capabilities: List[str] = None # Current abilities/resources
    
    def __post_init__(self):
        if self.knowledge_level is None:
            self.knowledge_level = {}
        if self.relationships is None:
            self.relationships = {}
        if self.goals is None:
            self.goals = []
        if self.capabilities is None:
            self.capabilities = []

@dataclass
class CausalLink:
    """A single causal relationship link"""
    from_event: str
    to_event: str
    causality_type: CausalityType
    strength: float                 # 0.0-1.0
    reasoning_summary: str
    confidence: float = 0.5

@dataclass
class TimelineEvent:
    """Core timeline event with comprehensive metadata"""
    # Core Identity
    event_id: str                    # Unique identifier
    volume_id: int
    batch_id: int
    chunk_ids: List[int] = None     # Source chunks
    
    # Temporal Positioning
    chronological_order: Optional[int] = None  # Global sequence number
    relative_time: str = ""         # "before_X", "during_Y", "after_Z"
    time_confidence: float = 0.5    # 0.0-1.0 confidence in temporal placement
    temporal_markers: List[str] = None  # ["morning", "three_days_later", "simultaneously"]
    
    # Event Content
    description: str = ""
    event_type: EventType = EventType.ACTION
    importance_score: float = 0.0   # 0.0-1.0 narrative significance
    
    # Character Integration
    primary_actors: List[str] = None      # Main characters involved
    affected_characters: List[str] = None # Characters indirectly affected
    character_states_before: Dict[str, CharacterState] = None
    character_states_after: Dict[str, CharacterState] = None
    
    # Causal Relationships
    caused_by_events: List[str] = None    # Event IDs that led to this
    causes_events: List[str] = None       # Event IDs this leads to
    motivation_triggers: List[MotivationTrigger] = None
    
    # Narrative Structure
    plot_thread: str = "main_plot"        # "main_plot", "character_arc", "subplot_A"
    narrative_function: NarrativeFunction = NarrativeFunction.RISING_ACTION
    foreshadowing_links: List[str] = None # Events this foreshadows/fulfills
    
    # Metadata
    confidence_level: float = 0.5
    validation_status: ValidationStatus = ValidationStatus.PENDING
    created_by: str = "ai_analysis"       # "ai_analysis", "function_call", "auto_inference"
    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        # Initialize None fields with empty lists/dicts
        if self.chunk_ids is None:
            self.chunk_ids = []
        if self.temporal_markers is None:
            self.temporal_markers = []
        if self.primary_actors is None:
            self.primary_actors = []
        if self.affected_characters is None:
            self.affected_characters = []
        if self.character_states_before is None:
            self.character_states_before = {}
        if self.character_states_after is None:
            self.character_states_after = {}
        if self.caused_by_events is None:
            self.caused_by_events = []
        if self.causes_events is None:
            self.causes_events = []
        if self.motivation_triggers is None:
            self.motivation_triggers = []
        if self.foreshadowing_links is None:
            self.foreshadowing_links = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_updated is None:
            self.last_updated = datetime.now()
    
    def add_causal_link(self, cause_event_id: str, causality_type: CausalityType, strength: float = 0.5):
        """Add a causal relationship where this event is the effect"""
        if cause_event_id not in self.caused_by_events:
            self.caused_by_events.append(cause_event_id)
            self.last_updated = datetime.now()
    
    def add_consequence(self, effect_event_id: str, causality_type: CausalityType, strength: float = 0.5):
        """Add a causal relationship where this event is the cause"""
        if effect_event_id not in self.causes_events:
            self.causes_events.append(effect_event_id)
            self.last_updated = datetime.now()

@dataclass
class TemporalConflict:
    """Represents a temporal consistency conflict"""
    conflict_id: str
    event_ids: List[str]
    conflict_type: str              # "chronological_order", "character_state", "causal_loop"
    description: str
    severity: float                 # 0.0-1.0
    suggested_resolution: str
    
@dataclass
class StorylineGraph:
    """Complete storyline graph with events and relationships"""
    # Event Storage
    events: Dict[str, TimelineEvent] = None           # event_id -> event
    chronological_sequence: List[str] = None          # Ordered event_ids
    
    # Relationship Networks - Will implement full graph structures later
    causal_links: Dict[str, List[CausalLink]] = None  # event_id -> outgoing causal links
    character_timeline: Dict[str, List[str]] = None   # character -> ordered event_ids
    plot_threads: Dict[str, List[str]] = None         # thread_name -> ordered event_ids
    
    # Temporal Anchors
    temporal_markers: Dict[str, int] = None           # "day_1" -> chronological_order
    chapter_boundaries: List[int] = None              # Chronological positions
    volume_boundaries: List[int] = None
    
    # Validation & Coherence
    consistency_score: float = 0.0                   # Overall timeline coherence
    unresolved_conflicts: List[TemporalConflict] = None
    confidence_distribution: Dict[str, int] = None    # confidence_level -> count
    
    def __post_init__(self):
        if self.events is None:
            self.events = {}
        if self.chronological_sequence is None:
            self.chronological_sequence = []
        if self.causal_links is None:
            self.causal_links = {}
        if self.character_timeline is None:
            self.character_timeline = {}
        if self.plot_threads is None:
            self.plot_threads = {}
        if self.temporal_markers is None:
            self.temporal_markers = {}
        if self.chapter_boundaries is None:
            self.chapter_boundaries = []
        if self.volume_boundaries is None:
            self.volume_boundaries = []
        if self.unresolved_conflicts is None:
            self.unresolved_conflicts = []
        if self.confidence_distribution is None:
            self.confidence_distribution = {}
    
    def add_event(self, event: TimelineEvent) -> None:
        """Add an event to the storyline graph"""
        self.events[event.event_id] = event
        
        # Update chronological sequence if order is specified
        if event.chronological_order is not None:
            self._insert_chronological_order(event.event_id, event.chronological_order)
        
        # Update character timelines
        for character in event.primary_actors + event.affected_characters:
            if character not in self.character_timeline:
                self.character_timeline[character] = []
            if event.event_id not in self.character_timeline[character]:
                self.character_timeline[character].append(event.event_id)
        
        # Update plot threads
        if event.plot_thread not in self.plot_threads:
            self.plot_threads[event.plot_thread] = []
        if event.event_id not in self.plot_threads[event.plot_thread]:
            self.plot_threads[event.plot_thread].append(event.event_id)
    
    def _insert_chronological_order(self, event_id: str, order: int) -> None:
        """Insert event into chronological sequence at correct position"""
        # Remove if already exists
        if event_id in self.chronological_sequence:
            self.chronological_sequence.remove(event_id)
        
        # Insert at correct position
        inserted = False
        for i, existing_id in enumerate(self.chronological_sequence):
            existing_event = self.events.get(existing_id)
            if existing_event and existing_event.chronological_order is not None:
                if order < existing_event.chronological_order:
                    self.chronological_sequence.insert(i, event_id)
                    inserted = True
                    break
        
        if not inserted:
            self.chronological_sequence.append(event_id)
    
    def get_events_by_character(self, character_name: str) -> List[TimelineEvent]:
        """Get all events involving a specific character"""
        if character_name not in self.character_timeline:
            return []
        
        return [self.events[event_id] for event_id in self.character_timeline[character_name]
                if event_id in self.events]
    
    def get_events_by_plot_thread(self, plot_thread: str) -> List[TimelineEvent]:
        """Get all events in a specific plot thread"""
        if plot_thread not in self.plot_threads:
            return []
        
        return [self.events[event_id] for event_id in self.plot_threads[plot_thread]
                if event_id in self.events]
    
    def get_chronological_events(self) -> List[TimelineEvent]:
        """Get all events in chronological order"""
        return [self.events[event_id] for event_id in self.chronological_sequence
                if event_id in self.events]

# ============================================================================
# COMBINATORIAL CAUSALITY SYSTEM
# ============================================================================

@dataclass
class CausalPath:
    """Compressed representation of causality chain for O(log n) queries"""
    target_event_id: str
    root_catalyst: str                    # Ultimate origin event
    path_length: int                      # Degrees of separation
    compressed_chain: List[CausalLink]    # Only critical nodes
    influence_strength: float             # Cumulative causality strength (0.0-1.0)
    confidence: float = 0.5              # Path reliability
    
    def __post_init__(self):
        if not self.compressed_chain:
            self.compressed_chain = []
    
    def compress_path(self, full_chain: List[CausalLink], compression_threshold: float = 0.3) -> List[CausalLink]:
        """Advanced path compression - remove low-strength intermediate links"""
        if len(full_chain) <= 2:
            return full_chain  # Can't compress short chains
        
        compressed = [full_chain[0]]  # Always keep first link (root connection)
        
        i = 1
        while i < len(full_chain) - 1:  # Never remove last link (target connection)
            current_link = full_chain[i]
            
            # Keep link if:
            # 1. High causality strength
            # 2. Critical character involvement
            # 3. Significant plot development
            keep_link = (
                current_link.strength >= compression_threshold or
                self._is_critical_character_link(current_link) or
                self._is_major_plot_development(current_link)
            )
            
            if keep_link:
                compressed.append(current_link)
            else:
                # Compress by combining with next link
                if i + 1 < len(full_chain):
                    next_link = full_chain[i + 1]
                    combined_link = self._combine_causal_links(current_link, next_link)
                    compressed.append(combined_link)
                    i += 1  # Skip next link since we combined it
            
            i += 1
        
        # Always add final link
        if len(full_chain) > 1:
            compressed.append(full_chain[-1])
        
        return compressed
    
    def _is_critical_character_link(self, link: CausalLink) -> bool:
        """Determine if link involves critical character developments"""
        # In a full implementation, this would check:
        # - Character first appearances
        # - Major character decisions
        # - Character relationship changes
        return link.causality_type in [CausalityType.CHARACTER_DECISION, CausalityType.CHARACTER_INTRODUCTION]
    
    def _is_major_plot_development(self, link: CausalLink) -> bool:
        """Determine if link represents major plot development"""
        # Check for significant plot advancement indicators
        significant_keywords = ['revelation', 'conflict', 'resolution', 'twist', 'climax']
        return any(keyword in link.reasoning_summary.lower() for keyword in significant_keywords)
    
    def _combine_causal_links(self, link1: CausalLink, link2: CausalLink) -> CausalLink:
        """Combine two adjacent causal links into one compressed link"""
        # Combined strength (multiplicative for cascading effects)
        combined_strength = link1.strength * link2.strength
        
        # Combined reasoning
        combined_reasoning = f"Compressed: {link1.reasoning_summary} → {link2.reasoning_summary}"
        
        # Choose strongest causality type
        stronger_type = link1.causality_type if link1.strength >= link2.strength else link2.causality_type
        
        return CausalLink(
            from_event=link1.from_event,
            to_event=link2.to_event,
            causality_type=stronger_type,
            strength=combined_strength,
            reasoning_summary=combined_reasoning
        )
    
    def get_compression_ratio(self) -> float:
        """Calculate compression ratio (compressed_length / original_length)"""
        if self.path_length <= 1:
            return 1.0
        return len(self.compressed_chain) / self.path_length
    
    def expand_path_segment(self, segment_index: int, full_events: Dict[str, 'TimelineEvent']) -> List[str]:
        """Expand a compressed segment back to full event sequence"""
        if segment_index >= len(self.compressed_chain):
            return []
        
        link = self.compressed_chain[segment_index]
        
        # If this is a compressed link (contains "Compressed:" in reasoning)
        if "Compressed:" in link.reasoning_summary:
            # In full implementation, would reconstruct intermediate events
            # For now, return the direct connection
            return [link.from_event, link.to_event]
        else:
            # Simple direct link
            return [link.from_event, link.to_event]
    
    def get_critical_waypoints(self) -> List[str]:
        """Get list of critical events in the causal path"""
        waypoints = [self.root_catalyst]
        
        for link in self.compressed_chain:
            # Add event if it's a critical waypoint
            if (link.strength >= 0.7 or  # High strength
                self._is_critical_character_link(link) or
                self._is_major_plot_development(link)):
                waypoints.append(link.to_event)
        
        # Always include target
        if self.target_event_id not in waypoints:
            waypoints.append(self.target_event_id)
        
        return waypoints
    
    def calculate_path_efficiency(self) -> float:
        """Calculate efficiency metric: influence_strength / path_length"""
        return self.influence_strength / max(1, self.path_length)
    
    def to_summary_string(self) -> str:
        """Generate human-readable summary of causal path"""
        if not self.compressed_chain:
            return f"Direct connection: {self.root_catalyst} → {self.target_event_id}"
        
        summary_parts = [f"Root: {self.root_catalyst}"]
        
        for i, link in enumerate(self.compressed_chain):
            strength_desc = "strong" if link.strength >= 0.7 else "moderate" if link.strength >= 0.4 else "weak"
            summary_parts.append(f"Step {i+1}: {strength_desc} causality → {link.to_event}")
        
        summary_parts.append(f"Final: {self.target_event_id} (strength: {self.influence_strength:.2f})")
        
        return " | ".join(summary_parts)

class PathCompressionEngine:
    """Advanced algorithms for causal path compression and optimization"""
    
    def __init__(self):
        self.compression_stats = {
            'paths_compressed': 0,
            'total_compression_ratio': 0.0,
            'avg_compression_ratio': 0.0
        }
    
    def compress_causal_path(self, events: List[str], causal_edges: Dict[str, List[Tuple[str, CausalityType, float]]], 
                           target_event: str, compression_level: str = "medium") -> CausalPath:
        """Main compression algorithm with configurable levels"""
        
        # Set compression parameters based on level
        thresholds = {
            "light": {"strength": 0.1, "max_hops": 10},
            "medium": {"strength": 0.3, "max_hops": 7}, 
            "aggressive": {"strength": 0.5, "max_hops": 4}
        }
        
        threshold = thresholds.get(compression_level, thresholds["medium"])
        
        # Build full causal chain
        full_chain = self._build_full_causal_chain(events, causal_edges, target_event)
        
        if not full_chain:
            return CausalPath(
                target_event_id=target_event,
                root_catalyst=events[0] if events else target_event,
                path_length=1,
                compressed_chain=[],
                influence_strength=0.0
            )
        
        # Apply compression algorithms
        compressed_chain = self._apply_multi_stage_compression(
            full_chain, threshold["strength"], threshold["max_hops"]
        )
        
        # Calculate metrics
        total_strength = self._calculate_path_strength(compressed_chain)
        
        # Update stats
        self.compression_stats['paths_compressed'] += 1
        compression_ratio = len(compressed_chain) / max(1, len(full_chain))
        self.compression_stats['total_compression_ratio'] += compression_ratio
        self.compression_stats['avg_compression_ratio'] = (
            self.compression_stats['total_compression_ratio'] / 
            self.compression_stats['paths_compressed']
        )
        
        return CausalPath(
            target_event_id=target_event,
            root_catalyst=events[0] if events else target_event,
            path_length=len(full_chain),
            compressed_chain=compressed_chain,
            influence_strength=total_strength,
            confidence=self._calculate_path_confidence(compressed_chain)
        )
    
    def _build_full_causal_chain(self, events: List[str], causal_edges: Dict[str, List[Tuple[str, CausalityType, float]]], 
                               target_event: str) -> List[CausalLink]:
        """Build complete causal chain from events to target"""
        if not events or not target_event:
            return []
        
        chain = []
        
        # Simple sequential chain building (in full implementation would use graph algorithms)
        for i in range(len(events) - 1):
            from_event = events[i]
            to_event = events[i + 1]
            
            # Find edge strength
            strength = 0.5  # Default
            causality_type = CausalityType.INDIRECT_INFLUENCE
            
            if from_event in causal_edges:
                for edge_to, edge_type, edge_strength in causal_edges[from_event]:
                    if edge_to == to_event:
                        strength = edge_strength
                        causality_type = edge_type
                        break
            
            chain.append(CausalLink(
                from_event=from_event,
                to_event=to_event,
                causality_type=causality_type,
                strength=strength,
                reasoning_summary=f"Chain link {i+1}: {from_event} → {to_event}"
            ))
        
        return chain
    
    def _apply_multi_stage_compression(self, full_chain: List[CausalLink], 
                                     strength_threshold: float, max_hops: int) -> List[CausalLink]:
        """Multi-stage compression pipeline"""
        
        # Stage 1: Remove weak links
        stage1 = self._remove_weak_links(full_chain, strength_threshold)
        
        # Stage 2: Combine redundant sequences  
        stage2 = self._combine_redundant_sequences(stage1)
        
        # Stage 3: Apply hop limit
        stage3 = self._apply_hop_limit(stage2, max_hops)
        
        # Stage 4: Preserve critical waypoints
        final = self._preserve_critical_waypoints(stage3, full_chain)
        
        return final
    
    def _remove_weak_links(self, chain: List[CausalLink], threshold: float) -> List[CausalLink]:
        """Stage 1: Remove links below strength threshold"""
        if not chain:
            return chain
        
        filtered = []
        
        for i, link in enumerate(chain):
            # Always keep first and last links
            if i == 0 or i == len(chain) - 1:
                filtered.append(link)
            elif link.strength >= threshold:
                filtered.append(link)
            # Skip weak intermediate links
        
        return filtered
    
    def _combine_redundant_sequences(self, chain: List[CausalLink]) -> List[CausalLink]:
        """Stage 2: Combine sequences of similar-strength links"""
        if len(chain) <= 2:
            return chain
        
        combined = [chain[0]]  # Always keep first
        
        i = 1
        while i < len(chain) - 1:  # Never modify last
            current = chain[i]
            
            # Look for combinable sequence
            if i + 1 < len(chain) - 1:  # Ensure we don't touch the last link
                next_link = chain[i + 1]
                
                # Combine if similar strength and compatible types
                if (abs(current.strength - next_link.strength) < 0.2 and
                    current.causality_type == next_link.causality_type):
                    
                    # Create combined link
                    combined_link = CausalLink(
                        from_event=current.from_event,
                        to_event=next_link.to_event,
                        causality_type=current.causality_type,
                        strength=(current.strength + next_link.strength) / 2,
                        reasoning_summary=f"Combined: {current.reasoning_summary} + {next_link.reasoning_summary}"
                    )
                    combined.append(combined_link)
                    i += 2  # Skip next link
                    continue
            
            combined.append(current)
            i += 1
        
        # Add final link
        if len(chain) > 1:
            combined.append(chain[-1])
        
        return combined
    
    def _apply_hop_limit(self, chain: List[CausalLink], max_hops: int) -> List[CausalLink]:
        """Stage 3: Limit maximum number of hops"""
        if len(chain) <= max_hops:
            return chain
        
        # Use intelligent sampling to preserve critical hops
        preserved = [chain[0]]  # Always keep first
        
        # Calculate step size for sampling
        middle_links = chain[1:-1]  # Exclude first and last
        if middle_links:
            step_size = max(1, len(middle_links) // (max_hops - 2))  # Reserve slots for first/last
            
            for i in range(0, len(middle_links), step_size):
                if len(preserved) < max_hops - 1:  # Reserve slot for last
                    preserved.append(middle_links[i])
        
        # Always add last link
        if len(chain) > 1:
            preserved.append(chain[-1])
        
        return preserved
    
    def _preserve_critical_waypoints(self, compressed: List[CausalLink], 
                                   original: List[CausalLink]) -> List[CausalLink]:
        """Stage 4: Ensure critical waypoints are preserved"""
        
        # Find critical waypoints in original chain
        critical_events = set()
        for link in original:
            if (link.strength >= 0.8 or  # Very high strength
                link.causality_type in [CausalityType.CHARACTER_DECISION, CausalityType.CHARACTER_INTRODUCTION]):
                critical_events.add(link.to_event)
                critical_events.add(link.from_event)
        
        # Ensure critical events are represented in compressed chain
        final_chain = []
        critical_covered = set()
        
        for link in compressed:
            final_chain.append(link)
            critical_covered.add(link.from_event)
            critical_covered.add(link.to_event)
        
        # Add missing critical waypoints (simplified - would use more sophisticated insertion in full implementation)
        missing_critical = critical_events - critical_covered
        if missing_critical and len(final_chain) > 1:
            # Insert critical waypoints as additional links (simplified approach)
            for critical_event in list(missing_critical)[:2]:  # Limit additions
                # Find best insertion point
                for i, link in enumerate(final_chain[:-1]):
                    critical_link = CausalLink(
                        from_event=link.to_event,
                        to_event=critical_event,
                        causality_type=CausalityType.CRITICAL_WAYPOINT,
                        strength=0.9,
                        reasoning_summary=f"Critical waypoint: {critical_event}"
                    )
                    final_chain.insert(i + 1, critical_link)
                    break
        
        return final_chain
    
    def _calculate_path_strength(self, chain: List[CausalLink]) -> float:
        """Calculate overall path strength"""
        if not chain:
            return 0.0
        
        # Multiplicative for cascading effects, but with dampening
        total_strength = 1.0
        for link in chain:
            total_strength *= (0.5 + link.strength * 0.5)  # Dampen to prevent too-low values
        
        return min(total_strength, 1.0)
    
    def _calculate_path_confidence(self, chain: List[CausalLink]) -> float:
        """Calculate confidence in compressed path"""
        if not chain:
            return 0.0
        
        # Base confidence on compression quality indicators
        avg_strength = sum(link.strength for link in chain) / len(chain)
        
        # Bonus for having critical waypoints
        critical_bonus = sum(1 for link in chain 
                           if link.causality_type == CausalityType.CRITICAL_WAYPOINT) * 0.1
        
        # Penalty for excessive compression
        compression_penalty = max(0, (len(chain) - 10) * 0.05)
        
        confidence = avg_strength + critical_bonus - compression_penalty
        return min(max(confidence, 0.0), 1.0)
    
    def get_compression_statistics(self) -> Dict[str, Any]:
        """Get compression performance statistics"""
        return self.compression_stats.copy()
    
    def benchmark_compression_algorithms(self, test_chains: List[List[CausalLink]]) -> Dict[str, float]:
        """Benchmark different compression approaches"""
        results = {}
        
        for level in ["light", "medium", "aggressive"]:
            compression_ratios = []
            for chain in test_chains:
                compressed = self._apply_multi_stage_compression(chain, 0.3, 7)
                ratio = len(compressed) / max(1, len(chain))
                compression_ratios.append(ratio)
            
            results[level] = sum(compression_ratios) / max(1, len(compression_ratios))
        
        return results

@dataclass 
class CausalIndex:
    """Multi-level indexing system for fast causality queries"""
    # Primary Index: Event -> Direct Causality Path
    event_to_path: Dict[str, CausalPath] = None
    
    # Secondary Index: Fast Lookups
    character_event_index: Dict[str, Set[str]] = None    # character -> event_ids
    motivation_index: Dict[str, Set[str]] = None         # motivation_type -> event_ids
    temporal_index: Dict[int, Set[str]] = None           # chronological_order -> event_ids
    
    # Tertiary Index: Combinatorial Queries
    causal_depth_index: Dict[int, Set[str]] = None       # depth_level -> event_ids
    cascade_strength_index: Dict[float, Set[str]] = None # influence_strength -> event_ids
    
    def __post_init__(self):
        if self.event_to_path is None:
            self.event_to_path = {}
        if self.character_event_index is None:
            self.character_event_index = {}
        if self.motivation_index is None:
            self.motivation_index = {}
        if self.temporal_index is None:
            self.temporal_index = {}
        if self.causal_depth_index is None:
            self.causal_depth_index = {}
        if self.cascade_strength_index is None:
            self.cascade_strength_index = {}
    
    def add_event_to_character_index(self, character_name: str, event_id: str):
        """Add event to character index for fast character-based queries"""
        if character_name not in self.character_event_index:
            self.character_event_index[character_name] = set()
        self.character_event_index[character_name].add(event_id)
    
    def add_event_to_temporal_index(self, chronological_order: int, event_id: str):
        """Add event to temporal index for chronological queries"""
        if chronological_order not in self.temporal_index:
            self.temporal_index[chronological_order] = set()
        self.temporal_index[chronological_order].add(event_id)
    
    def get_events_by_character(self, character_name: str) -> Set[str]:
        """O(1) lookup of events involving a character"""
        return self.character_event_index.get(character_name, set())
    
    def get_events_in_time_range(self, start_order: int, end_order: int) -> Set[str]:
        """O(log n) lookup of events in chronological range"""
        result = set()
        for order in range(start_order, end_order + 1):
            result.update(self.temporal_index.get(order, set()))
        return result
    
    def add_event_to_motivation_index(self, motivation_type: str, event_id: str):
        """Add event to motivation index for pattern-based queries"""
        if motivation_type not in self.motivation_index:
            self.motivation_index[motivation_type] = set()
        self.motivation_index[motivation_type].add(event_id)
    
    def add_causal_path(self, event_id: str, path: CausalPath):
        """Add causal path to primary index"""
        self.event_to_path[event_id] = path
        
        # Update depth index
        depth = path.path_length
        if depth not in self.causal_depth_index:
            self.causal_depth_index[depth] = set()
        self.causal_depth_index[depth].add(event_id)
        
        # Update strength index (quantized for efficiency)
        strength_bucket = round(path.influence_strength * 10) / 10  # Round to 0.1
        if strength_bucket not in self.cascade_strength_index:
            self.cascade_strength_index[strength_bucket] = set()
        self.cascade_strength_index[strength_bucket].add(event_id)
    
    def get_causal_path(self, event_id: str) -> Optional[CausalPath]:
        """O(1) lookup of causal path for event"""
        return self.event_to_path.get(event_id)
    
    def get_events_by_motivation(self, motivation_type: str) -> Set[str]:
        """O(1) lookup of events by motivation type"""
        return self.motivation_index.get(motivation_type, set())
    
    def get_events_by_depth(self, depth: int) -> Set[str]:
        """O(1) lookup of events by causal depth"""
        return self.causal_depth_index.get(depth, set())
    
    def get_events_by_strength_range(self, min_strength: float, max_strength: float) -> Set[str]:
        """O(log n) lookup of events by influence strength range"""
        result = set()
        min_bucket = round(min_strength * 10) / 10
        max_bucket = round(max_strength * 10) / 10
        
        bucket = min_bucket
        while bucket <= max_bucket:
            result.update(self.cascade_strength_index.get(bucket, set()))
            bucket = round((bucket + 0.1) * 10) / 10  # Avoid floating point precision issues
        
        return result
    
    def combinatorial_query(self, 
                          character_name: Optional[str] = None,
                          motivation_type: Optional[str] = None,
                          min_depth: Optional[int] = None,
                          max_depth: Optional[int] = None,
                          min_strength: Optional[float] = None,
                          max_strength: Optional[float] = None,
                          time_start: Optional[int] = None,
                          time_end: Optional[int] = None) -> Set[str]:
        """Advanced combinatorial query across multiple indices - O(log n)"""
        
        # Start with all events if no filters, or first filter result
        result_set = None
        
        # Apply character filter
        if character_name:
            candidate_set = self.get_events_by_character(character_name)
            result_set = candidate_set if result_set is None else result_set.intersection(candidate_set)
        
        # Apply motivation filter
        if motivation_type:
            candidate_set = self.get_events_by_motivation(motivation_type)
            result_set = candidate_set if result_set is None else result_set.intersection(candidate_set)
        
        # Apply depth range filter
        if min_depth is not None or max_depth is not None:
            depth_events = set()
            min_d = min_depth if min_depth is not None else 0
            max_d = max_depth if max_depth is not None else max(self.causal_depth_index.keys()) if self.causal_depth_index else 0
            
            for depth in range(min_d, max_d + 1):
                depth_events.update(self.get_events_by_depth(depth))
            
            result_set = depth_events if result_set is None else result_set.intersection(depth_events)
        
        # Apply strength range filter
        if min_strength is not None or max_strength is not None:
            min_s = min_strength if min_strength is not None else 0.0
            max_s = max_strength if max_strength is not None else 1.0
            strength_events = self.get_events_by_strength_range(min_s, max_s)
            result_set = strength_events if result_set is None else result_set.intersection(strength_events)
        
        # Apply temporal range filter
        if time_start is not None or time_end is not None:
            t_start = time_start if time_start is not None else min(self.temporal_index.keys()) if self.temporal_index else 0
            t_end = time_end if time_end is not None else max(self.temporal_index.keys()) if self.temporal_index else 0
            temporal_events = self.get_events_in_time_range(t_start, t_end)
            result_set = temporal_events if result_set is None else result_set.intersection(temporal_events)
        
        return result_set if result_set is not None else set()
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """Get indexing statistics for performance monitoring"""
        return {
            'total_causal_paths': len(self.event_to_path),
            'character_count': len(self.character_event_index),
            'motivation_types': len(self.motivation_index),
            'temporal_range': (min(self.temporal_index.keys()) if self.temporal_index else 0,
                             max(self.temporal_index.keys()) if self.temporal_index else 0),
            'depth_range': (min(self.causal_depth_index.keys()) if self.causal_depth_index else 0,
                          max(self.causal_depth_index.keys()) if self.causal_depth_index else 0),
            'strength_buckets': len(self.cascade_strength_index),
            'avg_events_per_character': (sum(len(events) for events in self.character_event_index.values()) / 
                                       max(1, len(self.character_event_index))) if self.character_event_index else 0
        }
    
    def rebuild_indices(self, events: List[TimelineEvent], causal_paths: Dict[str, CausalPath]):
        """Rebuild all indices from scratch - use for maintenance/optimization"""
        # Clear existing indices
        self.event_to_path.clear()
        self.character_event_index.clear()
        self.motivation_index.clear()
        self.temporal_index.clear()
        self.causal_depth_index.clear()
        self.cascade_strength_index.clear()
        
        # Rebuild from events and paths
        for event in events:
            # Character index
            for character in event.characters_involved:
                self.add_event_to_character_index(character, event.event_id)
            
            # Temporal index
            self.add_event_to_temporal_index(event.chronological_order, event.event_id)
            
            # Motivation index (if available in character states)
            for char_state in event.character_states.values():
                if hasattr(char_state, 'current_motivation') and char_state.current_motivation:
                    self.add_event_to_motivation_index(char_state.current_motivation, event.event_id)
        
        # Add causal paths
        for event_id, path in causal_paths.items():
            self.add_causal_path(event_id, path)

class CausalTree:
    """Single causality tree rooted at a catalyst event"""
    
    def __init__(self, root_event_id: str, catalyst_type: str = "story_origin"):
        self.root_event_id = root_event_id
        self.catalyst_type = catalyst_type  # "story_origin", "character_introduction", "plot_twist"
        self.nodes: Dict[str, TimelineEvent] = {}  # event_id -> event
        self.edges: Dict[str, List[str]] = {}      # parent_event_id -> [child_event_ids]
        self.reverse_edges: Dict[str, str] = {}    # child_event_id -> parent_event_id
        self.depth_map: Dict[str, int] = {}        # event_id -> depth from root
        
    def add_event(self, event: TimelineEvent, parent_event_id: Optional[str] = None):
        """Add event to the causal tree"""
        self.nodes[event.event_id] = event
        
        if parent_event_id is None:
            # This is a root or disconnected event
            if event.event_id == self.root_event_id:
                self.depth_map[event.event_id] = 0
        else:
            # Add edge from parent to this event
            if parent_event_id not in self.edges:
                self.edges[parent_event_id] = []
            self.edges[parent_event_id].append(event.event_id)
            self.reverse_edges[event.event_id] = parent_event_id
            
            # Calculate depth
            parent_depth = self.depth_map.get(parent_event_id, 0)
            self.depth_map[event.event_id] = parent_depth + 1
    
    def get_path_to_root(self, event_id: str) -> List[str]:
        """Get causality path from event to root catalyst - O(log n)"""
        path = []
        current = event_id
        
        while current and current in self.reverse_edges:
            path.append(current)
            current = self.reverse_edges[current]
        
        if current == self.root_event_id:
            path.append(current)
        
        return list(reversed(path))  # Root to target order
    
    def get_subtree_events(self, event_id: str, max_depth: int = 3) -> List[str]:
        """Get all events in subtree rooted at event_id - O(log n)"""
        if event_id not in self.nodes:
            return []
        
        result = [event_id]
        current_depth = 0
        queue = [(event_id, 0)]
        
        while queue and current_depth < max_depth:
            current_event, depth = queue.pop(0)
            current_depth = depth
            
            if current_event in self.edges:
                for child in self.edges[current_event]:
                    if child not in result:
                        result.append(child)
                        queue.append((child, depth + 1))
        
        return result

class CausalForest:
    """Multiple causality trees for different story threads"""
    
    def __init__(self):
        self.causal_trees: Dict[str, CausalTree] = {}  # root_event_id -> tree
        self.event_to_tree_map: Dict[str, str] = {}    # event_id -> root_event_id
        self.cross_tree_bridges: List[Tuple[str, str, float]] = []  # (from_tree_root, to_tree_root, strength)
        
    def add_tree(self, root_event_id: str, catalyst_type: str = "story_origin") -> CausalTree:
        """Add a new causality tree to the forest"""
        tree = CausalTree(root_event_id, catalyst_type)
        self.causal_trees[root_event_id] = tree
        self.event_to_tree_map[root_event_id] = root_event_id
        return tree
    
    def add_event_to_forest(self, event: TimelineEvent, cause_event_id: Optional[str] = None):
        """Add event to appropriate tree in the forest"""
        if cause_event_id and cause_event_id in self.event_to_tree_map:
            # Event has a cause, add to same tree
            tree_root = self.event_to_tree_map[cause_event_id]
            tree = self.causal_trees[tree_root]
            tree.add_event(event, cause_event_id)
            self.event_to_tree_map[event.event_id] = tree_root
        else:
            # Event is a new catalyst, create new tree
            tree = self.add_tree(event.event_id, "new_catalyst")
            tree.add_event(event)
    
    def find_causality_path(self, target_event_id: str, max_depth: int = 5) -> Optional[CausalPath]:
        """Find compressed causality path to target event - O(log n)"""
        if target_event_id not in self.event_to_tree_map:
            return None
        
        tree_root = self.event_to_tree_map[target_event_id]
        tree = self.causal_trees[tree_root]
        
        # Get path to root in this tree
        path_ids = tree.get_path_to_root(target_event_id)
        if len(path_ids) <= 1:
            return None
        
        # Convert to compressed causal links
        compressed_chain = []
        total_strength = 0.0
        
        for i in range(len(path_ids) - 1):
            from_event = path_ids[i]
            to_event = path_ids[i + 1]
            
            # Create causal link (simplified - in real implementation would query database)
            link = CausalLink(
                from_event=from_event,
                to_event=to_event,
                causality_type=CausalityType.DIRECT_CAUSE,
                strength=0.8,  # Would be calculated from database
                reasoning_summary=f"Chain link {i+1}"
            )
            compressed_chain.append(link)
            total_strength += link.strength
        
        return CausalPath(
            target_event_id=target_event_id,
            root_catalyst=tree_root,
            path_length=len(path_ids),
            compressed_chain=compressed_chain,
            influence_strength=total_strength / len(compressed_chain) if compressed_chain else 0.0
        )
    
    def query_character_causality(self, character_name: str, index: CausalIndex, 
                                max_events: int = 5) -> List[CausalPath]:
        """Query causality paths for character events - O(log n)"""
        character_events = index.get_events_by_character(character_name)
        paths = []
        
        for event_id in list(character_events)[:max_events]:
            path = self.find_causality_path(event_id)
            if path:
                paths.append(path)
        
        # Sort by influence strength
        return sorted(paths, key=lambda p: p.influence_strength, reverse=True)

class StratifiedCausalDAG:
    """Multi-layer causality with fast path finding - Alternative to Forest"""
    
    def __init__(self):
        # Layer 0: Root catalysts (story origins)
        self.catalyst_layer: Dict[str, TimelineEvent] = {}
        
        # Layer N: Causality levels (depth from root)
        self.causality_layers: Dict[int, Dict[str, TimelineEvent]] = {}
        
        # Cross-layer edges: Direct causality links
        self.causal_edges: Dict[str, List[Tuple[str, CausalityType, float]]] = {}  # from -> [(to, type, strength)]
        
        # Fast lookup: Event -> Layer mapping
        self.event_layer_map: Dict[str, int] = {}
        
    def add_catalyst(self, event: TimelineEvent):
        """Add root catalyst event"""
        self.catalyst_layer[event.event_id] = event
        self.event_layer_map[event.event_id] = 0
        
        # Initialize layer 0
        if 0 not in self.causality_layers:
            self.causality_layers[0] = {}
        self.causality_layers[0][event.event_id] = event
    
    def add_event(self, event: TimelineEvent, cause_events: List[str]):
        """Add event with causal dependencies"""
        if not cause_events:
            # No causes, treat as catalyst
            self.add_catalyst(event)
            return
        
        # Calculate layer based on maximum cause layer + 1
        max_cause_layer = 0
        for cause_id in cause_events:
            if cause_id in self.event_layer_map:
                cause_layer = self.event_layer_map[cause_id]
                max_cause_layer = max(max_cause_layer, cause_layer)
        
        event_layer = max_cause_layer + 1
        self.event_layer_map[event.event_id] = event_layer
        
        # Add to appropriate layer
        if event_layer not in self.causality_layers:
            self.causality_layers[event_layer] = {}
        self.causality_layers[event_layer][event.event_id] = event
        
        # Add causal edges
        for cause_id in cause_events:
            if cause_id not in self.causal_edges:
                self.causal_edges[cause_id] = []
            self.causal_edges[cause_id].append((event.event_id, CausalityType.DIRECT_CAUSE, 0.8))
    
    def find_causality_path_dijkstra(self, target_event_id: str, max_depth: int = 5) -> Optional[CausalPath]:
        """Dijkstra-based causality path finding - O(log n) with preprocessing"""
        if target_event_id not in self.event_layer_map:
            return None
        
        target_layer = self.event_layer_map[target_event_id]
        
        # Search only relevant layers (target_layer - max_depth : target_layer)
        search_layers = {}
        for layer in range(max(0, target_layer - max_depth), target_layer + 1):
            if layer in self.causality_layers:
                search_layers[layer] = self.causality_layers[layer]
        
        # Simple path reconstruction (simplified Dijkstra)
        # In full implementation would use proper priority queue
        
        # Find any path to root catalyst
        current = target_event_id
        path_events = [current]
        
        while current and target_layer > 0:
            # Find parent in previous layer
            found_parent = False
            for parent_id, edges in self.causal_edges.items():
                for edge_target, causality_type, strength in edges:
                    if edge_target == current:
                        path_events.append(parent_id)
                        current = parent_id
                        target_layer -= 1
                        found_parent = True
                        break
                if found_parent:
                    break
            
            if not found_parent:
                break
        
        if len(path_events) <= 1:
            return None
        
        # Reverse to get root->target order
        path_events.reverse()
        
        # Convert to CausalPath with accurate strength calculation
        compressed_chain = []
        total_strength = 1.0
        
        for i in range(len(path_events) - 1):
            # Find actual edge strength
            edge_strength = 0.8  # Default
            if path_events[i] in self.causal_edges:
                for target, causality_type, strength in self.causal_edges[path_events[i]]:
                    if target == path_events[i + 1]:
                        edge_strength = strength
                        break
            
            total_strength *= edge_strength
            
            link = CausalLink(
                from_event=path_events[i],
                to_event=path_events[i + 1],
                causality_type=CausalityType.DIRECT_CAUSE,
                strength=edge_strength,
                reasoning_summary=f"DAG layer {self.event_layer_map[path_events[i]]} -> {self.event_layer_map[path_events[i + 1]]}"
            )
            compressed_chain.append(link)
        
        return CausalPath(
            target_event_id=target_event_id,
            root_catalyst=path_events[0],
            path_length=len(path_events),
            compressed_chain=compressed_chain,
            influence_strength=total_strength
        )
    
    def get_layer_events(self, layer_num: int) -> Dict[str, TimelineEvent]:
        """Get all events in specific layer - O(1)"""
        return self.causality_layers.get(layer_num, {})
    
    def get_event_layer(self, event_id: str) -> Optional[int]:
        """Get layer number for event - O(1)"""
        return self.event_layer_map.get(event_id)
    
    def get_layer_count(self) -> int:
        """Get total number of layers"""
        return len(self.causality_layers)
    
    def get_catalysts(self) -> Dict[str, TimelineEvent]:
        """Get all root catalyst events - O(1)"""
        return self.catalyst_layer.copy()
    
    def get_events_caused_by(self, source_event_id: str, max_depth: int = 3) -> List[str]:
        """Get all events directly/indirectly caused by source - O(log n)"""
        if source_event_id not in self.event_layer_map:
            return []
        
        source_layer = self.event_layer_map[source_event_id]
        caused_events = []
        
        # BFS through causal edges within depth limit
        to_visit = [(source_event_id, 0)]  # (event_id, depth)
        visited = {source_event_id}
        
        while to_visit:
            current_event, current_depth = to_visit.pop(0)
            
            if current_depth >= max_depth:
                continue
            
            # Find direct effects
            for target, causality_type, strength in self.causal_edges.get(current_event, []):
                if target not in visited:
                    caused_events.append(target)
                    visited.add(target)
                    to_visit.append((target, current_depth + 1))
        
        return caused_events
    
    def query_layer_range(self, min_layer: int, max_layer: int) -> List[TimelineEvent]:
        """Query events within layer range - O(k) where k = events in range"""
        events = []
        for layer_num in range(min_layer, max_layer + 1):
            if layer_num in self.causality_layers:
                events.extend(self.causality_layers[layer_num].values())
        return events
    
    def get_causal_statistics(self) -> Dict[str, Any]:
        """Get DAG statistics for optimization insights"""
        total_events = sum(len(layer) for layer in self.causality_layers.values())
        total_edges = sum(len(edges) for edges in self.causal_edges.values())
        
        layer_sizes = {layer: len(events) for layer, events in self.causality_layers.items()}
        
        return {
            'total_events': total_events,
            'total_causal_edges': total_edges,
            'layer_count': self.get_layer_count(),
            'layer_sizes': layer_sizes,
            'avg_layer_size': total_events / max(1, self.get_layer_count()),
            'catalyst_count': len(self.catalyst_layer)
        }

class MotivationTrie:
    """Efficient storage/retrieval of motivation-driven event chains"""
    
    class TrieNode:
        def __init__(self):
            self.motivation_type: Optional[str] = None
            self.character_name: Optional[str] = None  
            self.children: Dict[str, 'MotivationTrie.TrieNode'] = {}    # next_motivation -> node
            self.terminal_events: List[str] = []       # Events at this path end
    
    def __init__(self):
        self.root = self.TrieNode()
    
    def add_motivation_chain(self, character_name: str, motivation_sequence: List[str], terminal_event_id: str):
        """Add motivation sequence pattern"""
        current = self.root
        
        # Navigate/create path through trie
        for motivation in motivation_sequence:
            key = f"{character_name}:{motivation}"
            if key not in current.children:
                current.children[key] = self.TrieNode()
                current.children[key].character_name = character_name
                current.children[key].motivation_type = motivation
            current = current.children[key]
        
        # Add terminal event
        current.terminal_events.append(terminal_event_id)
    
    def query_motivation_pattern(self, character_name: str, partial_pattern: List[str]) -> List[str]:
        """Find all events matching motivation pattern - O(m) where m = pattern length"""
        current = self.root
        
        # Navigate to pattern end
        for motivation in partial_pattern:
            key = f"{character_name}:{motivation}"
            if key not in current.children:
                return []
            current = current.children[key]
        
        # Collect all terminal events from this point
        result = []
        self._collect_terminal_events(current, result)
        return result
    
    def _collect_terminal_events(self, node: 'MotivationTrie.TrieNode', result: List[str]):
        """Recursively collect all terminal events from node subtree"""
        result.extend(node.terminal_events)
        for child in node.children.values():
            self._collect_terminal_events(child, result)

class OptimalCausalitySystem:
    """Combines best aspects of multiple causality structures"""
    
    def __init__(self):
        # Primary: Stratified DAG for general causality
        self.causal_dag = StratifiedCausalDAG()
        
        # Secondary: Forest for independent storylines
        self.causal_forest = CausalForest()
        
        # Tertiary: Character subgraphs for character-specific queries  
        self.character_graphs: Dict[str, CausalForest] = {}
        
        # Quaternary: Motivation trie for pattern matching
        self.motivation_trie = MotivationTrie()
        
        # Quintary: Multi-level indexing
        self.index = CausalIndex()
        
        # Path compression engine
        self.compression_engine = PathCompressionEngine()
        
        # Performance optimization engine
        self.performance_optimizer = CausalityPerformanceOptimizer()
        
        # Query routing preferences with performance metrics
        self.query_router = {
            "direct_causality": self._query_causal_dag,
            "character_motivation": self._query_character_forest,
            "storyline_analysis": self._query_causal_forest,
            "motivation_pattern": self._query_motivation_trie,
            "combinatorial_search": self._query_combinatorial_index,
            "compressed_path": self._query_compressed_path,
            "cross_structure": self._query_cross_structure
        }
        
        # Performance tracking for adaptive routing
        self.routing_stats = {
            "query_counts": {},
            "avg_response_times": {},
            "success_rates": {},
            "preferred_structures": {}
        }
    
    def add_timeline_event(self, event: TimelineEvent):
        """Add event to all relevant causality structures"""
        # Add to DAG
        self.causal_dag.add_event(event, event.caused_by_events)
        
        # Add to Forest  
        cause_event_id = event.caused_by_events[0] if event.caused_by_events else None
        self.causal_forest.add_event_to_forest(event, cause_event_id)
        
        # Add to character-specific forests
        for character in event.primary_actors + event.affected_characters:
            if character not in self.character_graphs:
                self.character_graphs[character] = CausalForest()
            self.character_graphs[character].add_event_to_forest(event, cause_event_id)
        
        # Update indexes
        self.index.event_to_path[event.event_id] = None  # Will be calculated on demand
        for character in event.primary_actors + event.affected_characters:
            self.index.add_event_to_character_index(character, event.event_id)
        
        if event.chronological_order:
            self.index.add_event_to_temporal_index(event.chronological_order, event.event_id)
    
    def smart_query(self, query_type: str, params: Dict) -> Optional[CausalPath]:
        """Route to optimal structure based on query type - O(log n)"""
        import time
        start_time = time.time()
        
        # Update query count
        if query_type not in self.routing_stats["query_counts"]:
            self.routing_stats["query_counts"][query_type] = 0
        self.routing_stats["query_counts"][query_type] += 1
        
        # Auto-route based on query characteristics
        optimal_route = self._determine_optimal_route(query_type, params)
        
        # Execute query with performance optimization
        result = None
        if optimal_route in self.query_router:
            try:
                # Generate unique query key
                query_key = self._generate_query_key(optimal_route, params)
                
                # Use performance optimizer
                result = self.performance_optimizer.optimize_query(
                    self.query_router[optimal_route], 
                    query_key, 
                    params
                )
                success = result is not None
            except Exception:
                success = False
        
        # Update performance stats
        end_time = time.time()
        response_time = end_time - start_time
        self._update_routing_stats(optimal_route, response_time, success)
        
        return result
    
    def _determine_optimal_route(self, query_type: str, params: Dict) -> str:
        """Intelligently determine optimal query route"""
        
        # Rule-based routing with performance feedback
        if query_type == "direct_causality":
            # Prefer DAG for direct causality queries
            return "direct_causality"
        
        elif query_type == "character_analysis":
            # Check if character-specific query
            if "character_name" in params:
                return "character_motivation"
            else:
                return "combinatorial_search"
        
        elif query_type == "motivation_search":
            # Use trie for motivation pattern matching
            return "motivation_pattern"
        
        elif query_type == "complex_causality":
            # Use multi-parameter search
            return "combinatorial_search"
        
        elif query_type == "story_flow":
            # Use forest for storyline analysis
            return "storyline_analysis"
        
        elif query_type == "compressed_analysis":
            # Use compressed path analysis
            return "compressed_path"
        
        else:
            # Use adaptive routing based on historical performance
            return self._get_adaptive_route(params)
    
    def _get_adaptive_route(self, params: Dict) -> str:
        """Adaptive routing based on historical performance"""
        
        # Get best performing route for similar queries
        best_route = "direct_causality"  # Default
        best_score = 0.0
        
        for route, success_list in self.routing_stats.get("success_rates", {}).items():
            success_rate = sum(success_list) / max(1, len(success_list)) if success_list else 0
            
            time_list = self.routing_stats.get("avg_response_times", {}).get(route, [1.0])
            avg_time = sum(time_list) / max(1, len(time_list)) if time_list else 1.0
            
            # Score = success_rate / response_time (higher is better)
            score = success_rate / max(avg_time, 0.001)
            
            if score > best_score:
                best_score = score
                best_route = route
        
        return best_route
    
    def _update_routing_stats(self, route: str, response_time: float, success: bool):
        """Update routing performance statistics"""
        
        # Update response times
        if route not in self.routing_stats["avg_response_times"]:
            self.routing_stats["avg_response_times"][route] = []
        
        # Keep last 100 response times for rolling average  
        if not isinstance(self.routing_stats["avg_response_times"][route], list):
            self.routing_stats["avg_response_times"][route] = []
            
        times_list = self.routing_stats["avg_response_times"][route]
        times_list.append(response_time)
        if len(times_list) > 100:
            times_list.pop(0)
        
        # Store the list, not the average
        self.routing_stats["avg_response_times"][route] = times_list
        
        # Update success rates
        if route not in self.routing_stats["success_rates"]:
            self.routing_stats["success_rates"][route] = []
        
        # Ensure it's a list
        if not isinstance(self.routing_stats["success_rates"][route], list):
            self.routing_stats["success_rates"][route] = []
            
        success_list = self.routing_stats["success_rates"][route]
        success_list.append(1.0 if success else 0.0)
        if len(success_list) > 100:
            success_list.pop(0)
        
        # Store the list, not the average
        self.routing_stats["success_rates"][route] = success_list
    
    def _query_causal_dag(self, params: Dict) -> Optional[CausalPath]:
        """Query using stratified DAG"""
        target_event = params.get("target_event_id")
        max_depth = params.get("max_depth", 5)
        
        if not target_event:
            return None
        
        return self.causal_dag.find_causality_path_dijkstra(target_event, max_depth)
    
    def _query_character_forest(self, params: Dict) -> Optional[CausalPath]:
        """Query using character-specific forest"""
        character_name = params.get("character_name")
        target_event = params.get("target_event_id")
        
        if not character_name or not target_event:
            return None
        
        if character_name in self.character_graphs:
            return self.character_graphs[character_name].find_causality_path(target_event)
        return None
    
    def _query_causal_forest(self, params: Dict) -> Optional[CausalPath]:
        """Query using main causal forest"""
        target_event = params.get("target_event_id")
        max_depth = params.get("max_depth", 5)
        
        if not target_event:
            return None
        
        return self.causal_forest.find_causality_path(target_event, max_depth)
    
    def _query_motivation_trie(self, params: Dict) -> Optional[CausalPath]:
        """Query using motivation pattern trie"""
        character_name = params.get("character_name")
        motivation_pattern = params.get("motivation_pattern", [])
        
        if not character_name or not motivation_pattern:
            return None
        
        # Get matching events from trie
        matching_events = self.motivation_trie.query_motivation_pattern(character_name, motivation_pattern)
        
        if matching_events:
            # Return path to first matching event
            return self._query_causal_dag({"target_event_id": matching_events[0]})
        
        return None
    
    def _query_combinatorial_index(self, params: Dict) -> Optional[CausalPath]:
        """Query using combinatorial index search"""
        
        # Extract all possible search parameters
        character_name = params.get("character_name")
        motivation_type = params.get("motivation_type") 
        min_depth = params.get("min_depth")
        max_depth = params.get("max_depth")
        min_strength = params.get("min_strength")
        max_strength = params.get("max_strength")
        time_start = params.get("time_start")
        time_end = params.get("time_end")
        
        # Perform combinatorial search
        matching_events = self.index.combinatorial_query(
            character_name=character_name,
            motivation_type=motivation_type,
            min_depth=min_depth,
            max_depth=max_depth,
            min_strength=min_strength,
            max_strength=max_strength,
            time_start=time_start,
            time_end=time_end
        )
        
        if matching_events:
            # Return path to highest-priority matching event
            target_event = params.get("target_event_id") or list(matching_events)[0]
            return self._query_causal_dag({"target_event_id": target_event})
        
        return None
    
    def _query_compressed_path(self, params: Dict) -> Optional[CausalPath]:
        """Query with path compression optimization"""
        target_event = params.get("target_event_id")
        compression_level = params.get("compression_level", "medium")
        
        if not target_event:
            return None
        
        # Get basic path first
        base_path = self._query_causal_dag(params)
        
        if base_path and base_path.compressed_chain:
            # Apply additional compression
            events = [link.from_event for link in base_path.compressed_chain] + [base_path.target_event_id]
            causal_edges = {}  # Would be populated from actual data structures
            
            return self.compression_engine.compress_causal_path(
                events, causal_edges, target_event, compression_level
            )
        
        return base_path
    
    def _query_cross_structure(self, params: Dict) -> Optional[CausalPath]:
        """Query across multiple structures and return best result"""
        target_event = params.get("target_event_id")
        
        if not target_event:
            return None
        
        # Try multiple structures
        candidates = []
        
        # Try DAG
        dag_result = self._query_causal_dag(params)
        if dag_result:
            candidates.append(("dag", dag_result))
        
        # Try Forest
        forest_result = self._query_causal_forest(params)
        if forest_result:
            candidates.append(("forest", forest_result))
        
        # Try character-specific if character provided
        if "character_name" in params:
            char_result = self._query_character_forest(params)
            if char_result:
                candidates.append(("character", char_result))
        
        if not candidates:
            return None
        
        # Select best result based on influence strength and path efficiency
        best_candidate = max(candidates, 
                           key=lambda x: x[1].influence_strength * x[1].calculate_path_efficiency())
        
        return best_candidate[1]
    
    def get_routing_performance_report(self) -> Dict[str, Any]:
        """Get detailed routing performance report"""
        return {
            "query_distribution": self.routing_stats["query_counts"],
            "average_response_times": self.routing_stats["avg_response_times"],
            "success_rates": self.routing_stats["success_rates"],
            "compression_stats": self.compression_engine.get_compression_statistics(),
            "index_stats": self.index.get_index_statistics(),
            "dag_stats": self.causal_dag.get_causal_statistics() if hasattr(self.causal_dag, 'get_causal_statistics') else {}
        }
    
    def optimize_routing_strategy(self) -> Dict[str, str]:
        """Analyze performance and suggest routing optimizations"""
        suggestions = {}
        
        # Analyze success rates
        success_rates = self.routing_stats.get("success_rates", {})
        for route, rate in success_rates.items():
            if rate < 0.5:
                suggestions[f"low_success_{route}"] = f"Consider alternative approach for {route} queries (success rate: {rate:.2f})"
        
        # Analyze response times
        response_times = self.routing_stats.get("avg_response_times", {})
        if response_times:
            slowest_route = max(response_times.items(), key=lambda x: x[1])
            if slowest_route[1] > 1.0:  # > 1 second
                suggestions["slow_route"] = f"Route {slowest_route[0]} is slow ({slowest_route[1]:.2f}s avg)"
        
        # Suggest structure-specific optimizations
        index_stats = self.index.get_index_statistics()
        if index_stats.get("character_count", 0) > 50:
            suggestions["character_optimization"] = "Consider character-specific caching for large character sets"
        
        return suggestions
    
    def _generate_query_key(self, route: str, params: Dict) -> str:
        """Generate unique key for query caching"""
        key_parts = [route]
        
        # Add relevant parameters to key
        for key in ['target_event_id', 'character_name', 'max_depth', 'compression_level']:
            if key in params:
                key_parts.append(f"{key}:{params[key]}")
        
        return "_".join(key_parts)
    
    def precompute_hot_paths(self, event_ids: List[str]):
        """Precompute paths for commonly queried events"""
        return self.performance_optimizer.batch_precompute_paths(self, event_ids)
    
    def get_comprehensive_performance_report(self) -> Dict[str, Any]:
        """Get complete performance report across all systems"""
        return {
            "routing_performance": self.get_routing_performance_report(),
            "cache_performance": self.performance_optimizer.get_cache_statistics(),
            "optimization_suggestions": self.performance_optimizer.suggest_optimizations(self),
            "structure_health": {
                "dag_layers": self.causal_dag.get_layer_count() if hasattr(self.causal_dag, 'get_layer_count') else 0,
                "forest_trees": len(self.causal_forest.causal_trees),
                "character_graphs": len(self.character_graphs),
                "index_coverage": self.index.get_index_statistics()
            }
        }
    
    def auto_optimize_performance(self):
        """Automatically apply performance optimizations"""
        # Get optimization suggestions
        suggestions = self.performance_optimizer.suggest_optimizations(self)
        
        optimizations_applied = []
        
        # Apply memory optimization if needed
        if any("memory" in s.lower() for s in suggestions):
            self.performance_optimizer.optimize_memory_usage()
            optimizations_applied.append("memory_cleanup")
        
        # Precompute very hot paths
        very_hot_paths = [k for k, v in self.performance_optimizer.hot_paths.items() if v > 20]
        if very_hot_paths:
            # Extract event IDs from hot path keys
            event_ids = []
            for hot_key in very_hot_paths[:10]:  # Limit to top 10
                parts = hot_key.split('_')
                if len(parts) > 1 and parts[1] != 'unknown':
                    event_ids.append(parts[1])
            
            if event_ids:
                precomputed = self.precompute_hot_paths(event_ids)
                optimizations_applied.append(f"precomputed_{precomputed}_paths")
        
        return optimizations_applied

# ============================================================================
# CAUSALITY PERFORMANCE OPTIMIZATION ENGINE 
# ============================================================================

class CausalityPerformanceOptimizer:
    """Advanced performance optimization for causality queries"""
    
    def __init__(self):
        self.query_cache: Dict[str, CausalPath] = {}
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        # Precomputed optimization structures
        self.precomputed_paths: Dict[str, CausalPath] = {}
        self.hot_paths: Dict[str, int] = {}  # path_signature -> access_count
        
        # Performance metrics
        self.optimization_stats = {
            'cache_hits': 0,
            'cache_misses': 0, 
            'precomputation_savings': 0,
            'avg_query_time': 0.0,
            'memory_usage_mb': 0.0
        }
        
        # Adaptive optimization thresholds
        self.hot_path_threshold = 5  # Precompute after 5 accesses
        self.cache_size_limit = 1000
        
    def optimize_query(self, query_func, query_key: str, *args, **kwargs) -> Optional[CausalPath]:
        """Wrapper for query optimization with caching and precomputation"""
        import time
        start_time = time.time()
        
        # Check cache first
        if query_key in self.query_cache:
            self.cache_hit_count += 1
            self.optimization_stats['cache_hits'] += 1
            return self.query_cache[query_key]
        
        # Check precomputed paths
        if query_key in self.precomputed_paths:
            result = self.precomputed_paths[query_key]
            self.optimization_stats['precomputation_savings'] += 1
        else:
            # Execute query
            result = query_func(*args, **kwargs)
            self.cache_miss_count += 1
            self.optimization_stats['cache_misses'] += 1
        
        # Cache result if valid
        if result and len(self.query_cache) < self.cache_size_limit:
            self.query_cache[query_key] = result
        
        # Track hot paths
        if query_key not in self.hot_paths:
            self.hot_paths[query_key] = 0
        self.hot_paths[query_key] += 1
        
        # Trigger precomputation for hot paths
        if self.hot_paths[query_key] >= self.hot_path_threshold and query_key not in self.precomputed_paths:
            self._schedule_precomputation(query_key, query_func, *args, **kwargs)
        
        # Update performance metrics
        end_time = time.time()
        query_time = end_time - start_time
        self.optimization_stats['avg_query_time'] = (
            (self.optimization_stats['avg_query_time'] * (self.cache_hit_count + self.cache_miss_count - 1) + query_time) /
            (self.cache_hit_count + self.cache_miss_count)
        )
        
        return result
    
    def _schedule_precomputation(self, query_key: str, query_func, *args, **kwargs):
        """Schedule precomputation for frequently accessed paths"""
        try:
            # Precompute and store
            precomputed_result = query_func(*args, **kwargs)
            if precomputed_result:
                self.precomputed_paths[query_key] = precomputed_result
        except Exception:
            pass  # Ignore precomputation failures
    
    def batch_precompute_paths(self, causality_system: 'OptimalCausalitySystem', 
                             event_ids: List[str], max_depth: int = 5):
        """Batch precompute paths for given events"""
        precomputed_count = 0
        
        for event_id in event_ids:
            query_key = f"dag_{event_id}_{max_depth}"
            
            if query_key not in self.precomputed_paths:
                try:
                    path = causality_system.causal_dag.find_causality_path_dijkstra(event_id, max_depth)
                    if path:
                        self.precomputed_paths[query_key] = path
                        precomputed_count += 1
                except Exception:
                    continue
        
        return precomputed_count
    
    def optimize_memory_usage(self):
        """Optimize memory usage by cleaning up least-used cache entries"""
        if len(self.query_cache) <= self.cache_size_limit:
            return
        
        # Remove least accessed paths
        sorted_hot_paths = sorted(self.hot_paths.items(), key=lambda x: x[1])
        to_remove = len(self.query_cache) - self.cache_size_limit + 100  # Remove extra for breathing room
        
        for query_key, _ in sorted_hot_paths[:to_remove]:
            if query_key in self.query_cache:
                del self.query_cache[query_key]
            if query_key in self.hot_paths:
                del self.hot_paths[query_key]
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get caching performance statistics"""
        total_queries = self.cache_hit_count + self.cache_miss_count
        hit_rate = self.cache_hit_count / max(1, total_queries)
        
        return {
            'cache_hit_rate': hit_rate,
            'total_queries': total_queries,
            'cache_size': len(self.query_cache),
            'precomputed_paths': len(self.precomputed_paths),
            'hot_paths_count': len([count for count in self.hot_paths.values() if count >= self.hot_path_threshold]),
            'memory_savings_estimate': len(self.precomputed_paths) * self.optimization_stats['avg_query_time']
        }
    
    def suggest_optimizations(self, causality_system: 'OptimalCausalitySystem') -> List[str]:
        """Suggest performance optimizations based on usage patterns"""
        suggestions = []
        
        # Cache performance suggestions
        hit_rate = self.cache_hit_count / max(1, self.cache_hit_count + self.cache_miss_count)
        if hit_rate < 0.3:
            suggestions.append("Low cache hit rate - consider increasing cache size or adjusting query patterns")
        
        # Hot path analysis
        very_hot_paths = [k for k, v in self.hot_paths.items() if v > 20]
        if very_hot_paths:
            suggestions.append(f"Found {len(very_hot_paths)} very hot paths - consider permanent precomputation")
        
        # Memory usage analysis
        if len(self.query_cache) > self.cache_size_limit * 0.9:
            suggestions.append("Cache approaching size limit - consider memory optimization")
        
        # Structure-specific suggestions
        dag_stats = causality_system.causal_dag.get_causal_statistics() if hasattr(causality_system.causal_dag, 'get_causal_statistics') else {}
        if dag_stats.get('layer_count', 0) > 20:
            suggestions.append("Deep causality layers detected - consider layer-specific indexing")
        
        return suggestions
    
    def benchmark_performance(self, causality_system: 'OptimalCausalitySystem', 
                            test_queries: List[Dict[str, Any]]) -> Dict[str, float]:
        """Benchmark performance across different optimization strategies"""
        import time
        
        results = {
            'no_optimization': 0.0,
            'with_caching': 0.0,
            'with_precomputation': 0.0,
            'full_optimization': 0.0
        }
        
        # Temporarily disable optimizations for baseline
        original_cache = self.query_cache.copy()
        original_precomputed = self.precomputed_paths.copy()
        
        # Test without optimization
        self.query_cache.clear()
        self.precomputed_paths.clear()
        
        start_time = time.time()
        for query in test_queries:
            causality_system._query_causal_dag(query)
        results['no_optimization'] = time.time() - start_time
        
        # Test with caching
        start_time = time.time()
        for query in test_queries:
            query_key = f"test_{query.get('target_event_id', 'unknown')}"
            self.optimize_query(causality_system._query_causal_dag, query_key, query)
        results['with_caching'] = time.time() - start_time
        
        # Test with precomputation
        self.query_cache.clear()
        event_ids = [q.get('target_event_id') for q in test_queries if q.get('target_event_id')]
        self.batch_precompute_paths(causality_system, event_ids)
        
        start_time = time.time()
        for query in test_queries:
            query_key = f"dag_{query.get('target_event_id')}_{query.get('max_depth', 5)}"
            self.optimize_query(causality_system._query_causal_dag, query_key, query)
        results['with_precomputation'] = time.time() - start_time
        
        # Test with full optimization
        start_time = time.time()
        for query in test_queries:
            query_key = f"full_{query.get('target_event_id', 'unknown')}"
            self.optimize_query(causality_system._query_causal_dag, query_key, query)
        results['full_optimization'] = time.time() - start_time
        
        # Restore original state
        self.query_cache = original_cache
        self.precomputed_paths = original_precomputed
        
        return results
    
    def _query_motivation_trie(self, params: Dict) -> List[str]:
        """Query using motivation trie"""
        character_name = params.get("character_name")
        motivation_pattern = params.get("motivation_pattern", [])
        
        if character_name:
            return self.motivation_trie.query_motivation_pattern(character_name, motivation_pattern)
        return []

class QdrantDataLoader:
    """Handles loading data from Qdrant database"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.client = None
        
    def _connect_to_qdrant(self) -> bool:
        """Connect to Qdrant database"""
        try:
            self.client = QdrantClient(url=self.config.qdrant_url, verify=False)
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            return False
    
    def load_volume_chunks(self, volume_ids: List[int]) -> Dict[int, List[Dict]]:
        """Load chunks for specified volumes from Qdrant using coordinate-based approach"""
        print(f"📖 LOADING CHUNKS FOR VOLUMES: {volume_ids}")
        print("=" * 50)
        
        if not self.config.use_qdrant or not self._connect_to_qdrant():
            raise RuntimeError(f"Cannot load volume chunks: Qdrant connection failed or disabled. Check connection to {self.config.qdrant_url}")
        
        try:
            # Get all points from Qdrant
            print("   🔄 Fetching all data from Qdrant...")
            scroll_result = self.client.scroll(
                collection_name=self.config.collection_name,
                with_payload=True,
                limit=50000
            )
            
            all_points = scroll_result[0]
            print(f"   ✅ Retrieved {len(all_points)} total points")
            
            # Organize by volume using coordinate[0] as volume indicator (0-based)
            volume_chunks = defaultdict(list)
            
            for point in all_points:
                payload = point.payload
                content = payload.get('chunk', '')
                
                if not content.strip():
                    continue
                
                # Extract volume from coordinate[0] (convert from 0-based to 1-based)
                if 'coordinate' in payload and isinstance(payload['coordinate'], list) and len(payload['coordinate']) >= 2:
                    volume_index = payload['coordinate'][0]  # 0-based volume index
                    chunk_index = payload['coordinate'][1]   # chunk index within volume
                    volume_id = volume_index + 1  # Convert to 1-based volume ID
                    
                    if volume_id in volume_ids:
                        chunk_data = {
                            'id': chunk_index + 1,  # Use actual chunk index
                            'content': content,
                            'volume': volume_id,
                            'chunk_index': chunk_index,
                            'chapter': (chunk_index // 20) + 1,  # Estimate chapter from chunk position
                            'metadata': payload.get('metadata', {}),
                            'source': 'qdrant',
                            'coordinate': payload['coordinate']
                        }
                        volume_chunks[volume_id].append(chunk_data)
            
            # Sort chunks by their chunk_index to maintain order
            for vol_id in volume_chunks:
                volume_chunks[vol_id].sort(key=lambda x: x['chunk_index'])
                # Renumber IDs sequentially
                for i, chunk in enumerate(volume_chunks[vol_id]):
                    chunk['id'] = i + 1
            
            # Display results
            for vol_id in volume_ids:
                if vol_id in volume_chunks:
                    chunks = volume_chunks[vol_id]
                    total_chars = sum(len(chunk['content']) for chunk in chunks)
                    chunk_range = f"{chunks[0]['chunk_index']}-{chunks[-1]['chunk_index']}" if chunks else "N/A"
                    print(f"   📕 Volume {vol_id}: {len(chunks)} chunks (indices {chunk_range}), {total_chars:,} characters")
                else:
                    print(f"   ❌ Volume {vol_id}: No chunks found")
            
            return dict(volume_chunks)
            
        except Exception as e:
            logger.error(f"Error loading from Qdrant: {e}")
            raise RuntimeError(f"Failed to load chunks from Qdrant for volumes {volume_ids}: {str(e)}")
    
    def _extract_volume_id(self, payload: Dict, content: str, point_index: int, total_points: int) -> int:
        """Extract volume ID from payload or content"""
        # Method 1: Check payload keys
        for key in ['volume', 'volume_id', 'volume_number']:
            if key in payload:
                return int(payload[key])
        
        # Method 2: Check metadata
        if 'metadata' in payload and isinstance(payload['metadata'], dict):
            meta = payload['metadata']
            for key in ['volume', 'volume_id']:
                if key in meta:
                    return int(meta[key])
        
        # Method 3: Use coordinate field - this is the PRIMARY method for this dataset
        if 'coordinate' in payload:
            coord = payload['coordinate']
            if isinstance(coord, list) and len(coord) >= 1:
                # coordinate format: [volume_number, chunk_index]
                volume_id = coord[0] + 1  # Convert 0-based to 1-based
                if 1 <= volume_id <= 22:
                    return volume_id
            elif isinstance(coord, (int, float)):
                # Single number coordinate
                volume_id = max(1, min(22, int(coord) + 1))
                return volume_id
        
        # Method 4: Extract from content using expanded patterns (fallback)
        volume_patterns = [
            r'魔法禁书目录\s*(\d+)',
            r'第(\d+)卷',
            r'Volume\s*(\d+)',
            r'卷(\d+)',
            r'ħ������Ŀ¼\s*(\d+)',  # Encoded Chinese
            r'INDEX\s*(\d+)',
            r'学园都市\s*(\d+)',
            r'第(\d+)册',
            r'VOLUME\s*(\d+)'
        ]
        
        for pattern in volume_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                vol_num = int(match.group(1))
                if 1 <= vol_num <= 22:  # Valid volume range
                    return vol_num
        
        # Method 5: Return None if volume cannot be determined
        return None
    

class UnifiedReasonerEngine:
    """Unified reasoning engine combining all processing methods"""
    
    def __init__(self, config: ProcessingConfig, processor=None):
        self.config = config
        self.processor = processor  # Reference to UnifiedNovelProcessor for database access
        self.deepseek_config = create_deepseek_config()
        self.deepseek_client = DeepSeekClient(self.deepseek_config)
        
        # Initialize Qdrant for context
        try:
            self.qdrant_client = QdrantClient(url=config.qdrant_url, verify=False)
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise RuntimeError(f"Cannot connect to Qdrant at {config.qdrant_url}. Please check the connection.")
        
        print(f"🧠 Unified Reasoner Engine Initialized")
        print(f"   Mode: {config.mode.value}")
        print(f"   Max iterations: {config.max_iterations}")
        print(f"   Satisfaction threshold: {config.satisfaction_threshold}")
    
    
    def _get_character_registry_context(self, volume_id: int) -> List[Dict]:
        """Get existing character registry for context - LLM will manage registration"""
        print(f"\n👥 CHARACTER DEBUG: AGENTIC CHARACTER SYSTEM ({volume_id})")
        print(f"   🤖 LLM will dynamically register and manage characters through function calls")
        
        # Query existing character registry for this volume
        existing_characters = self._query_character_registry(volume_id)
        
        if existing_characters:
            print(f"   📋 Existing registry ({len(existing_characters)} total): {[c['name'] for c in existing_characters[:5]]}")
            if len(existing_characters) > 5:
                print(f"   📋 + {len(existing_characters) - 5} more characters...")
        else:
            print(f"   ✨ Clean slate - LLM will discover and register new characters")
        
        return existing_characters
    
    def _query_character_registry(self, volume_id: int) -> List[Dict]:
        """Query character registry database"""
        try:
            if self.config.use_postgres and POSTGRES_AVAILABLE:
                return self._query_character_registry_postgres(volume_id)
            else:
                return self._query_character_registry_sqlite(volume_id)
        except Exception as e:
            print(f"   ⚠️ Character registry query failed: {e}")
            return []
    
    def _query_character_registry_postgres(self, volume_id: int) -> List[Dict]:
        """Query PostgreSQL character registry"""
        try:
            conn = psycopg2.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_db,
                user=self.config.postgres_user,
                password=self.config.postgres_password
            )
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT character_name, character_role, confidence_score
                FROM character_registry 
                WHERE volume_id = %s 
                ORDER BY confidence_score DESC, appearance_count DESC
                LIMIT 10
            """, (volume_id,))
            
            results = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'name': row[0],
                    'role': row[1] or 'Unknown',
                    'confidence': row[2] or 0.5
                }
                for row in results
            ]
        except Exception as e:
            print(f"   ⚠️ PostgreSQL character query failed: {e}")
            return []
    
    def _query_character_registry_sqlite(self, volume_id: int) -> List[Dict]:
        """Query SQLite character registry"""
        try:
            conn = sqlite3.connect("unified_results.db")
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT character_name, character_role, confidence_score
                FROM character_registry 
                WHERE volume_id = ? 
                ORDER BY confidence_score DESC, appearance_count DESC
                LIMIT 10
            """, (volume_id,))
            
            results = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'name': row[0],
                    'role': row[1] or 'Unknown',
                    'confidence': row[2] or 0.5
                }
                for row in results
            ]
        except Exception as e:
            print(f"   ⚠️ SQLite character query failed: {e}")
            return []
    
    def _process_function_calls(self, analysis: Dict, volume_id: int, batch_id: int):
        """Process LLM function calls for character management"""
        if 'function_calls' not in analysis:
            return
        
        function_calls = analysis['function_calls']
        if not isinstance(function_calls, list):
            return
        
        print(f"\n🤖 PROCESSING FUNCTION CALLS ({len(function_calls)} calls)")
        
        for call in function_calls:
            try:
                if isinstance(call, dict) and 'action' in call and 'params' in call:
                    action = call['action']
                    params = call['params']
                    
                    # Character management functions
                    if action == 'register_character':
                        self._register_character(volume_id, batch_id, params)
                    elif action == 'update_character':
                        self._update_character(volume_id, params)
                    elif action == 'query_character':
                        self._query_character_info(volume_id, params)
                    
                    # Timeline management functions
                    elif action == 'create_timeline_event':
                        self._create_timeline_event(volume_id, batch_id, params)
                    elif action == 'update_timeline_event':
                        self._update_timeline_event(volume_id, params)
                    elif action == 'establish_causality':
                        self._establish_causality(volume_id, params)
                    elif action == 'update_character_timeline':
                        self._update_character_timeline(volume_id, params)
                    elif action == 'query_timeline_context':
                        self._query_timeline_context(volume_id, params)
                    elif action == 'validate_temporal_consistency':
                        self._validate_temporal_consistency(volume_id, params)
                    elif action == 'trace_motivation_chain':
                        self._trace_motivation_chain(volume_id, params)
                    
                    else:
                        print(f"   ⚠️ Unknown function call: {action}")
                        
            except Exception as e:
                print(f"   ❌ Function call error: {e}")
    
    def _register_character(self, volume_id: int, batch_id: int, params: Dict):
        """Register a new character in the registry"""
        try:
            name = params.get('name', '').strip()
            role = params.get('role', 'Unknown')
            description = params.get('description', '')
            confidence = float(params.get('confidence', 0.5))
            
            if not name:
                print(f"   ⚠️ Cannot register character with empty name")
                return
            
            print(f"   📝 Registering character: {name} ({role})")
            
            if self.config.use_postgres and POSTGRES_AVAILABLE:
                self._register_character_postgres(volume_id, batch_id, name, role, description, confidence)
            else:
                self._register_character_sqlite(volume_id, batch_id, name, role, description, confidence)
                
        except Exception as e:
            print(f"   ❌ Character registration failed: {e}")
    
    def _register_character_postgres(self, volume_id: int, batch_id: int, name: str, role: str, description: str, confidence: float):
        """Register character in PostgreSQL"""
        try:
            conn = psycopg2.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_db,
                user=self.config.postgres_user,
                password=self.config.postgres_password
            )
            cursor = conn.cursor()
            
            # Try to insert, on conflict update
            cursor.execute("""
                INSERT INTO character_registry 
                (volume_id, character_name, character_role, character_description, confidence_score, first_appearance)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (volume_id, character_name) 
                DO UPDATE SET 
                    character_role = EXCLUDED.character_role,
                    character_description = EXCLUDED.character_description,
                    confidence_score = GREATEST(character_registry.confidence_score, EXCLUDED.confidence_score),
                    appearance_count = character_registry.appearance_count + 1,
                    last_updated = CURRENT_TIMESTAMP
            """, (volume_id, name, role, description, confidence, f"Volume {volume_id}, Batch {batch_id}"))
            
            conn.commit()
            conn.close()
            print(f"   ✅ Character '{name}' registered in PostgreSQL")
            
        except Exception as e:
            print(f"   ❌ PostgreSQL character registration failed: {e}")
    
    def _register_character_sqlite(self, volume_id: int, batch_id: int, name: str, role: str, description: str, confidence: float):
        """Register character in SQLite"""
        try:
            conn = sqlite3.connect("unified_results.db")
            cursor = conn.cursor()
            
            # Check if character exists
            cursor.execute("""
                SELECT id, appearance_count, confidence_score FROM character_registry 
                WHERE volume_id = ? AND character_name = ?
            """, (volume_id, name))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing character
                char_id, count, old_confidence = existing
                new_confidence = max(old_confidence, confidence)
                cursor.execute("""
                    UPDATE character_registry 
                    SET character_role = ?, character_description = ?, 
                        confidence_score = ?, appearance_count = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (role, description, new_confidence, count + 1, char_id))
                print(f"   ✅ Character '{name}' updated in SQLite (count: {count + 1})")
            else:
                # Insert new character
                cursor.execute("""
                    INSERT INTO character_registry 
                    (volume_id, character_name, character_role, character_description, 
                     confidence_score, first_appearance)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (volume_id, name, role, description, confidence, f"Volume {volume_id}, Batch {batch_id}"))
                print(f"   ✅ Character '{name}' registered in SQLite")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"   ❌ SQLite character registration failed: {e}")
    
    def _update_character(self, volume_id: int, params: Dict):
        """Update existing character information"""
        try:
            name = params.get('name', '').strip()
            new_role = params.get('new_role', '')
            confidence = float(params.get('confidence', 0.5))
            
            if not name:
                return
            
            print(f"   📝 Updating character: {name} -> {new_role}")
            
            if self.config.use_postgres and POSTGRES_AVAILABLE:
                self._update_character_postgres(volume_id, name, new_role, confidence)
            else:
                self._update_character_sqlite(volume_id, name, new_role, confidence)
                
        except Exception as e:
            print(f"   ❌ Character update failed: {e}")
    
    def _update_character_postgres(self, volume_id: int, name: str, new_role: str, confidence: float):
        """Update character in PostgreSQL"""
        try:
            conn = psycopg2.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_db,
                user=self.config.postgres_user,
                password=self.config.postgres_password
            )
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE character_registry 
                SET character_role = %s, confidence_score = GREATEST(confidence_score, %s),
                    last_updated = CURRENT_TIMESTAMP
                WHERE volume_id = %s AND character_name = %s
            """, (new_role, confidence, volume_id, name))
            
            conn.commit()
            conn.close()
            print(f"   ✅ Character '{name}' updated in PostgreSQL")
            
        except Exception as e:
            print(f"   ❌ PostgreSQL character update failed: {e}")
    
    def _update_character_sqlite(self, volume_id: int, name: str, new_role: str, confidence: float):
        """Update character in SQLite"""
        try:
            conn = sqlite3.connect("unified_results.db")
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE character_registry 
                SET character_role = ?, confidence_score = MAX(confidence_score, ?),
                    last_updated = CURRENT_TIMESTAMP
                WHERE volume_id = ? AND character_name = ?
            """, (new_role, confidence, volume_id, name))
            
            conn.commit()
            conn.close()
            print(f"   ✅ Character '{name}' updated in SQLite")
            
        except Exception as e:
            print(f"   ❌ SQLite character update failed: {e}")
    
    def _query_character_info(self, volume_id: int, params: Dict):
        """Query character information for LLM context"""
        try:
            name = params.get('name', '').strip()
            if not name:
                return
            
            print(f"   🔍 Querying character: {name}")
            
            # Query implementation would return character info for context
            # This is mainly for logging/debugging in this implementation
            
        except Exception as e:
            print(f"   ❌ Character query failed: {e}")
    
    # ============================================================================
    # TIMELINE FUNCTION CALL HANDLERS
    # ============================================================================
    
    def _create_timeline_event(self, volume_id: int, batch_id: int, params: Dict):
        """Create a new timeline event"""
        try:
            # Extract parameters
            description = params.get('description', '').strip()
            event_type_str = params.get('event_type', 'action')
            primary_actors = params.get('primary_actors', [])
            affected_characters = params.get('affected_characters', [])
            importance_score = float(params.get('importance_score', 0.5))
            chronological_order = params.get('chronological_order')
            plot_thread = params.get('plot_thread', 'main_plot')
            causal_triggers = params.get('causal_triggers', [])
            temporal_position = params.get('temporal_position', '')
            
            if not description:
                print(f"   ⚠️ Timeline event creation failed: No description provided")
                return
            
            # Generate unique event ID
            import time
            event_id = f"evt_{volume_id}_{batch_id}_{int(time.time())}"
            
            # Parse event type
            try:
                event_type = EventType(event_type_str)
            except ValueError:
                event_type = EventType.ACTION
            
            # Create timeline event
            timeline_event = TimelineEvent(
                event_id=event_id,
                volume_id=volume_id,
                batch_id=batch_id,
                description=description,
                event_type=event_type,
                primary_actors=primary_actors if isinstance(primary_actors, list) else [primary_actors],
                affected_characters=affected_characters if isinstance(affected_characters, list) else [affected_characters],
                importance_score=importance_score,
                chronological_order=chronological_order,
                plot_thread=plot_thread,
                relative_time=temporal_position,
                caused_by_events=causal_triggers if isinstance(causal_triggers, list) else [causal_triggers] if causal_triggers else [],
                created_by='function_call'
            )
            
            # Save to database
            success = self.processor.save_timeline_event(timeline_event) if self.processor else False
            if success:
                print(f"   ✅ Timeline event created: {event_id}")
                print(f"      Description: {description}")
                print(f"      Type: {event_type.value}")
                print(f"      Actors: {primary_actors}")
                
                # Save causal relationships if provided
                for cause_event_id in timeline_event.caused_by_events:
                    if cause_event_id and self.processor:
                        self.processor.save_causal_relationship(
                            cause_event_id, event_id, CausalityType.DIRECT_CAUSE,
                            confidence=0.7, reasoning="Created via function call"
                        )
            else:
                print(f"   ❌ Failed to save timeline event: {event_id}")
                
        except Exception as e:
            print(f"   ❌ Timeline event creation failed: {e}")
    
    def _update_timeline_event(self, volume_id: int, params: Dict):
        """Update an existing timeline event"""
        try:
            event_id = params.get('event_id', '').strip()
            if not event_id:
                print(f"   ⚠️ Timeline event update failed: No event_id provided")
                return
            
            # Load existing event
            existing_event = self.processor.load_timeline_event(event_id) if self.processor else None
            if not existing_event:
                print(f"   ⚠️ Timeline event not found: {event_id}")
                return
            
            # Update fields
            if 'new_importance' in params:
                existing_event.importance_score = float(params['new_importance'])
            
            if 'add_causal_link' in params:
                cause_event_id = params['add_causal_link']
                if cause_event_id not in existing_event.caused_by_events:
                    existing_event.caused_by_events.append(cause_event_id)
            
            if 'update_character_states' in params:
                char_updates = params['update_character_states']
                for char_name, state_updates in char_updates.items():
                    # Update character state after
                    if char_name not in existing_event.character_states_after:
                        existing_event.character_states_after[char_name] = CharacterState(emotional_state='')
                    
                    char_state = existing_event.character_states_after[char_name]
                    if 'emotional_state' in state_updates:
                        char_state.emotional_state = state_updates['emotional_state']
                    if 'new_goal' in state_updates:
                        char_state.goals.append(state_updates['new_goal'])
            
            existing_event.last_updated = datetime.now()
            
            # Save updated event
            success = self.processor.save_timeline_event(existing_event) if self.processor else False
            if success:
                print(f"   ✅ Timeline event updated: {event_id}")
            else:
                print(f"   ❌ Failed to update timeline event: {event_id}")
                
        except Exception as e:
            print(f"   ❌ Timeline event update failed: {e}")
    
    def _establish_causality(self, volume_id: int, params: Dict):
        """Establish causal relationship between events"""
        try:
            cause_event_id = params.get('cause_event_id', '').strip()
            effect_event_id = params.get('effect_event_id', '').strip()
            causality_type_str = params.get('causality_type', 'direct_cause')
            confidence = float(params.get('confidence', 0.5))
            reasoning = params.get('reasoning', '')
            
            if not cause_event_id or not effect_event_id:
                print(f"   ⚠️ Causality establishment failed: Missing event IDs")
                return
            
            # Parse causality type
            try:
                causality_type = CausalityType(causality_type_str)
            except ValueError:
                causality_type = CausalityType.DIRECT_CAUSE
            
            # Save causal relationship
            success = self.processor.save_causal_relationship(
                cause_event_id, effect_event_id, causality_type, confidence, reasoning
            ) if self.processor else False
            
            if success:
                print(f"   ✅ Causality established: {cause_event_id} → {effect_event_id}")
                print(f"      Type: {causality_type.value}")
                print(f"      Confidence: {confidence}")
                
                # Update the effect event's caused_by_events list
                if self.processor:
                    effect_event = self.processor.load_timeline_event(effect_event_id)
                    if effect_event and cause_event_id not in effect_event.caused_by_events:
                        effect_event.add_causal_link(cause_event_id, causality_type)
                        self.processor.save_timeline_event(effect_event)
            else:
                print(f"   ❌ Failed to establish causality: {cause_event_id} → {effect_event_id}")
                
        except Exception as e:
            print(f"   ❌ Causality establishment failed: {e}")
    
    def _update_character_timeline(self, volume_id: int, params: Dict):
        """Update character timeline with event participation"""
        try:
            character_name = params.get('character_name', '').strip()
            event_id = params.get('event_id', '').strip()
            character_role = params.get('character_role', 'participant')
            state_changes = params.get('state_changes', {})
            
            if not character_name or not event_id:
                print(f"   ⚠️ Character timeline update failed: Missing character or event ID")
                return
            
            # Load the event
            event = self.processor.load_timeline_event(event_id) if self.processor else None
            if not event:
                print(f"   ⚠️ Event not found for character timeline update: {event_id}")
                return
            
            # Add character to event if not already present
            if character_name not in event.primary_actors and character_name not in event.affected_characters:
                if character_role == 'protagonist':
                    event.primary_actors.append(character_name)
                else:
                    event.affected_characters.append(character_name)
            
            # Update character state changes
            if state_changes:
                if character_name not in event.character_states_after:
                    event.character_states_after[character_name] = CharacterState(emotional_state='')
                
                char_state = event.character_states_after[character_name]
                
                if 'knowledge_gained' in state_changes:
                    knowledge_key = f"knowledge_{len(char_state.knowledge_level)}"
                    char_state.knowledge_level[knowledge_key] = 1.0
                
                if 'emotional_shift' in state_changes:
                    shift = state_changes['emotional_shift']
                    if ' -> ' in shift:
                        before, after = shift.split(' -> ')
                        char_state.emotional_state = after.strip()
                
                if 'relationship_change' in state_changes:
                    rel_changes = state_changes['relationship_change']
                    for other_char, change in rel_changes.items():
                        if ' -> ' in change:
                            before_str, after_str = change.split(' -> ')
                            try:
                                char_state.relationships[other_char] = float(after_str)
                            except ValueError:
                                pass
            
            event.last_updated = datetime.now()
            
            # Save updated event
            success = self.processor.save_timeline_event(event) if self.processor else False
            if success:
                print(f"   ✅ Character timeline updated: {character_name} in {event_id}")
            else:
                print(f"   ❌ Failed to update character timeline: {character_name} in {event_id}")
                
        except Exception as e:
            print(f"   ❌ Character timeline update failed: {e}")
    
    def _query_timeline_context(self, volume_id: int, params: Dict):
        """Query timeline context for character or event analysis"""
        try:
            query_type = params.get('query_type', 'character_motivation')
            character_name = params.get('character_name', '').strip()
            time_window = params.get('time_window', 'last_5_events')
            context_request = params.get('context_request', '')
            
            print(f"   🔍 Timeline context query: {query_type}")
            print(f"      Character: {character_name}")
            print(f"      Request: {context_request}")
            
            # Load recent events for the volume
            limit = 5 if 'last_5' in time_window else 10
            recent_events = self.processor.load_events_by_volume(volume_id, limit=limit) if self.processor else []
            
            # Filter events involving the character
            relevant_events = []
            if character_name:
                for event in recent_events:
                    if (character_name in event.primary_actors or 
                        character_name in event.affected_characters):
                        relevant_events.append(event)
            else:
                relevant_events = recent_events
            
            print(f"      Found {len(relevant_events)} relevant events")
            
            # This would typically return structured data for the LLM to use
            # For now, we just log the query for debugging
            
        except Exception as e:
            print(f"   ❌ Timeline context query failed: {e}")
    
    def _validate_temporal_consistency(self, volume_id: int, params: Dict):
        """Validate temporal consistency in timeline"""
        try:
            validation_scope = params.get('validation_scope', f'volume_{volume_id}')
            check_types = params.get('check_types', ['chronological_order'])
            
            print(f"   🔍 Temporal consistency validation: {validation_scope}")
            print(f"      Check types: {check_types}")
            
            # Load all events for the volume
            events = self.processor.load_events_by_volume(volume_id) if self.processor else []
            
            issues_found = 0
            
            if 'chronological_order' in check_types:
                # Check for chronological order issues
                ordered_events = [e for e in events if e.chronological_order is not None]
                ordered_events.sort(key=lambda x: x.chronological_order)
                
                for i in range(len(ordered_events) - 1):
                    current = ordered_events[i]
                    next_event = ordered_events[i + 1]
                    if current.chronological_order >= next_event.chronological_order:
                        print(f"      ⚠️ Chronological order conflict: {current.event_id} >= {next_event.event_id}")
                        issues_found += 1
            
            if issues_found == 0:
                print(f"   ✅ Temporal consistency validation passed")
            else:
                print(f"   ⚠️ Found {issues_found} temporal consistency issues")
            
        except Exception as e:
            print(f"   ❌ Temporal consistency validation failed: {e}")
    
    def _trace_motivation_chain(self, volume_id: int, params: Dict):
        """Trace character motivation chain through events"""
        try:
            character_name = params.get('character_name', '').strip()
            starting_event = params.get('starting_event', '')
            ending_event = params.get('ending_event', '')
            motivation_hypothesis = params.get('motivation_hypothesis', '')
            
            print(f"   🔍 Motivation chain trace: {character_name}")
            print(f"      Hypothesis: {motivation_hypothesis}")
            print(f"      Range: {starting_event} → {ending_event}")
            
            # Load events for the volume
            events = self.processor.load_events_by_volume(volume_id) if self.processor else []
            
            # Filter events involving the character
            character_events = []
            for event in events:
                if (character_name in event.primary_actors or 
                    character_name in event.affected_characters):
                    character_events.append(event)
            
            # Sort by chronological order
            character_events.sort(key=lambda x: x.chronological_order or 0)
            
            print(f"      Found {len(character_events)} events involving {character_name}")
            
            # This would perform actual motivation analysis
            # For now, we log the trace request
            
        except Exception as e:
            print(f"   ❌ Motivation chain trace failed: {e}")
    
    def _auto_register_ai_characters(self, analysis: Dict, volume_id: int, batch_id: int):
        """Auto-register characters discovered by AI if not explicitly handled by function calls"""
        if 'characters' not in analysis:
            return
        
        characters = analysis['characters']
        if not isinstance(characters, list):
            return
        
        print(f"\n🔄 AUTO-REGISTERING AI DISCOVERED CHARACTERS")
        
        for char_info in characters:
            try:
                if isinstance(char_info, dict) and 'name' in char_info:
                    name = char_info['name'].strip()
                    role = char_info.get('role', 'Unknown')
                    
                    if name:
                        # Check if character already exists in registry
                        existing_chars = self._query_character_registry(volume_id)
                        existing_names = [c['name'] for c in existing_chars]
                        
                        if name not in existing_names:
                            print(f"   📝 Auto-registering new character: {name}")
                            params = {
                                'name': name,
                                'role': role,
                                'description': f"Auto-discovered from AI analysis",
                                'confidence': 0.7  # Medium confidence for auto-registration
                            }
                            self._register_character(volume_id, batch_id, params)
                        else:
                            print(f"   ✅ Character already registered: {name}")
                            
            except Exception as e:
                print(f"   ❌ Auto-registration error: {e}")
    
    def _store_timeline_events(self, analysis: Dict, volume_id: int, batch_id: int):
        """Store timeline events from AI analysis to event_analysis table"""
        if 'events' not in analysis:
            return
        
        events = analysis['events']
        if not isinstance(events, list):
            return
        
        print(f"\n📅 STORING TIMELINE EVENTS ({len(events)} events)")
        
        for event_info in events:
            try:
                if isinstance(event_info, dict) and 'description' in event_info:
                    description = event_info['description']
                    importance = float(event_info.get('importance', 0.5))
                    timeline = event_info.get('timeline', '')
                    
                    if description:
                        print(f"   📝 Storing event: {description[:50]}...")
                        
                        if self.config.use_postgres and POSTGRES_AVAILABLE:
                            self._store_event_postgres(volume_id, batch_id, description, importance, timeline)
                        else:
                            self._store_event_sqlite(volume_id, batch_id, description, importance, timeline)
                            
            except Exception as e:
                print(f"   ❌ Event storage error: {e}")
    
    def _store_event_postgres(self, volume_id: int, batch_id: int, description: str, importance: float, timeline: str):
        """Store event in PostgreSQL"""
        try:
            conn = psycopg2.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_db,
                user=self.config.postgres_user,
                password=self.config.postgres_password
            )
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO event_analysis 
                (volume_id, batch_id, event_description, importance_score, timeline_position)
                VALUES (%s, %s, %s, %s, %s)
            """, (volume_id, batch_id, description, importance, timeline))
            
            conn.commit()
            conn.close()
            print(f"   ✅ Event stored in PostgreSQL")
            
        except Exception as e:
            print(f"   ❌ PostgreSQL event storage failed: {e}")
    
    def _store_event_sqlite(self, volume_id: int, batch_id: int, description: str, importance: float, timeline: str):
        """Store event in SQLite"""
        try:
            conn = sqlite3.connect("unified_results.db")
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS event_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    volume_id INTEGER NOT NULL,
                    batch_id INTEGER NOT NULL,
                    event_description TEXT NOT NULL,
                    importance_score REAL DEFAULT 0.0,
                    timeline_position TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                INSERT INTO event_analysis 
                (volume_id, batch_id, event_description, importance_score, timeline_position)
                VALUES (?, ?, ?, ?, ?)
            """, (volume_id, batch_id, description, importance, timeline))
            
            conn.commit()
            conn.close()
            print(f"   ✅ Event stored in SQLite")
            
        except Exception as e:
            print(f"   ❌ SQLite event storage failed: {e}")
    
    def _store_character_analysis(self, analysis: Dict, volume_id: int, batch_id: int):
        """Store character analysis from AI to character_analysis table"""
        if 'characters' not in analysis:
            return
        
        characters = analysis['characters']
        if not isinstance(characters, list):
            return
        
        print(f"\n🎭 STORING CHARACTER ANALYSIS ({len(characters)} characters)")
        
        for char_info in characters:
            try:
                if isinstance(char_info, dict) and 'name' in char_info:
                    name = char_info['name']
                    role = char_info.get('role', 'Unknown')
                    key_actions = char_info.get('key_actions', [])
                    
                    if name:
                        print(f"   📝 Storing analysis for: {name}")
                        
                        if self.config.use_postgres and POSTGRES_AVAILABLE:
                            self._store_char_analysis_postgres(volume_id, batch_id, name, role, key_actions)
                        else:
                            self._store_char_analysis_sqlite(volume_id, batch_id, name, role, key_actions)
                            
            except Exception as e:
                print(f"   ❌ Character analysis storage error: {e}")
    
    def _store_char_analysis_postgres(self, volume_id: int, batch_id: int, name: str, role: str, key_actions: list):
        """Store character analysis in PostgreSQL"""
        try:
            conn = psycopg2.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_db,
                user=self.config.postgres_user,
                password=self.config.postgres_password
            )
            cursor = conn.cursor()
            
            # Convert key_actions to PostgreSQL array format
            actions_array = key_actions if isinstance(key_actions, list) else []
            
            cursor.execute("""
                INSERT INTO character_analysis 
                (volume_id, batch_id, character_name, character_role, key_actions, confidence_score)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (volume_id, batch_id, character_name) 
                DO UPDATE SET 
                    character_role = EXCLUDED.character_role,
                    key_actions = EXCLUDED.key_actions,
                    confidence_score = GREATEST(character_analysis.confidence_score, EXCLUDED.confidence_score)
            """, (volume_id, batch_id, name, role, actions_array, 0.8))
            
            conn.commit()
            conn.close()
            print(f"   ✅ Character analysis stored in PostgreSQL")
            
        except Exception as e:
            print(f"   ❌ PostgreSQL character analysis storage failed: {e}")
    
    def _store_char_analysis_sqlite(self, volume_id: int, batch_id: int, name: str, role: str, key_actions: list):
        """Store character analysis in SQLite"""
        try:
            conn = sqlite3.connect("unified_results.db")
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS character_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    volume_id INTEGER NOT NULL,
                    batch_id INTEGER NOT NULL,
                    character_name TEXT NOT NULL,
                    character_role TEXT,
                    key_actions TEXT,
                    confidence_score REAL DEFAULT 0.8,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(volume_id, batch_id, character_name)
                )
            """)
            
            # Convert key_actions to JSON string for SQLite
            actions_json = json.dumps(key_actions) if isinstance(key_actions, list) else "[]"
            
            cursor.execute("""
                INSERT OR REPLACE INTO character_analysis 
                (volume_id, batch_id, character_name, character_role, key_actions, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (volume_id, batch_id, name, role, actions_json, 0.8))
            
            conn.commit()
            conn.close()
            print(f"   ✅ Character analysis stored in SQLite")
            
        except Exception as e:
            print(f"   ❌ SQLite character analysis storage failed: {e}")
    
    def _determine_processing_stage(self, batch_position: int, total_batches: int) -> ProcessingStage:
        """Determine processing stage based on position"""
        progress = (batch_position / total_batches) * 100
        
        if progress <= 20:
            return ProcessingStage.BEGINNING
        elif progress <= 40:
            return ProcessingStage.EARLY
        elif progress <= 70:
            return ProcessingStage.MIDDLE
        elif progress <= 90:
            return ProcessingStage.CLIMAX
        else:
            return ProcessingStage.ENDING
    
    async def process_batch_with_unified_logic(self, chunks: List[Dict], batch_id: int, total_batches: int, volume_id: int) -> Dict[str, Any]:
        """Process batch with unified logic from all processors"""
        
        # Determine processing context
        processing_context = ProcessingContext(
            batch_position=batch_id,
            total_batches=total_batches,
            processing_stage=self._determine_processing_stage(batch_id, total_batches),
            volume_id=volume_id
        )
        
        # Get existing character context for agentic system
        existing_characters = self._get_character_registry_context(volume_id)
        character_names = [char['name'] for char in existing_characters[:5]]  # Top 5 for context
        
        print(f"\n🔄 Processing Batch {batch_id}")
        print(f"   Volume: {volume_id}")
        print(f"   Stage: {processing_context.processing_stage.value}")
        print(f"   Known characters ({len(character_names)} active): {character_names[:5]}")
        if len(character_names) > 5:
            print(f"   + {len(character_names) - 5} more in registry")
        
        iterations = []
        current_satisfaction = 0.0
        no_improvement_count = 0
        total_tokens = 0
        
        for iteration_num in range(1, self.config.max_iterations + 1):
            print(f"\n   Iteration {iteration_num}:")
            start_time = time.time()
            
            # Build unified prompt
            prompt = self._build_unified_prompt(
                chunks, processing_context, existing_characters, 
                iteration_num, current_satisfaction
            )
            
            # Query DeepSeek
            response_data = await self._query_deepseek_with_retry(prompt)
            iteration_time = time.time() - start_time
            
            if response_data['success']:
                # Parse response
                result = self._parse_json_response(response_data['response'], iteration_num)
                result.tokens_used = response_data['tokens_used']
                result.improvement_score = result.satisfaction_level - current_satisfaction
                
                # Process function calls for character management
                if hasattr(result, 'analysis') and result.analysis:
                    self._process_function_calls(result.analysis, volume_id, batch_id)
                    
                    # Auto-register characters discovered by AI if not already handled by function calls
                    self._auto_register_ai_characters(result.analysis, volume_id, batch_id)
                    
                    # Store timeline events and character analysis
                    self._store_timeline_events(result.analysis, volume_id, batch_id)
                    self._store_character_analysis(result.analysis, volume_id, batch_id)
                
                iterations.append(result)
                total_tokens += result.tokens_used
                
                print(f"     Satisfaction: {result.satisfaction_level:.3f} (Δ{result.improvement_score:+.3f})")
                print(f"     Tokens: {result.tokens_used:,}")
                print(f"     Time: {iteration_time:.1f}s")
                
                # Debug: Show analysis results and extract AI-discovered characters
                ai_discovered_chars = []
                if hasattr(result, 'analysis') and result.analysis:
                    analysis = result.analysis
                    if 'characters' in analysis:
                        ai_chars = analysis['characters'][:3]
                        print(f"     🎭 AI Characters: {ai_chars}")
                        
                        # Extract character names from AI analysis
                        for char_info in analysis['characters']:
                            if isinstance(char_info, dict) and 'name' in char_info:
                                char_name = char_info['name']
                                if char_name not in character_names:  # Not in our registry list
                                    ai_discovered_chars.append(char_name)
                    
                    if 'events' in analysis:
                        events = analysis['events'][:2]  # Show first 2 events
                        event_descriptions = [e.get('description', 'Unknown')[:50] + '...' if len(e.get('description', '')) > 50 else e.get('description', 'Unknown') for e in events]
                        print(f"     📅 Timeline Events: {event_descriptions}")
                    
                    if 'analysis' in analysis:
                        plot_info = analysis['analysis']
                        if isinstance(plot_info, dict) and 'plot_development' in plot_info:
                            print(f"     📖 Plot: {plot_info['plot_development'][:80]}...")
                
                # Show missed characters
                if ai_discovered_chars:
                    print(f"     ⚠️ AI found additional characters: {ai_discovered_chars}")
                    print(f"     💡 Consider adding these to character tracking")
                
                # Debug: Show reasoning
                if hasattr(result, 'reasoning_trace') and result.reasoning_trace:
                    print(f"     🧠 Reasoning: {result.reasoning_trace[0][:60]}...")
                
                # Check termination conditions
                if result.satisfaction_level >= self.config.satisfaction_threshold:
                    print(f"     ✅ Threshold reached: {result.satisfaction_level:.3f}")
                    break
                elif result.is_meaningful_improvement:
                    print(f"     📈 Meaningful improvement: +{result.improvement_score:.3f}")
                    no_improvement_count = 0
                    current_satisfaction = result.satisfaction_level
                else:
                    print(f"     ⚠️  Insufficient improvement: +{result.improvement_score:.3f}")
                    no_improvement_count += 1
                    current_satisfaction = result.satisfaction_level
                    
                    if no_improvement_count >= self.config.early_termination_patience:
                        print(f"     ⏹️  Early termination after {no_improvement_count} attempts")
                        break
            else:
                print(f"     ❌ Request failed: {response_data.get('error')}")
                break
        
        # Final result
        final_satisfaction = iterations[-1].satisfaction_level if iterations else 0.0
        meaningful_improvements = sum(1 for it in iterations if it.is_meaningful_improvement)
        
        # Collect AI-discovered characters from final iteration
        ai_characters = []
        if iterations and hasattr(iterations[-1], 'analysis') and iterations[-1].analysis:
            analysis = iterations[-1].analysis
            if 'characters' in analysis:
                for char_info in analysis['characters']:
                    if isinstance(char_info, dict) and 'name' in char_info:
                        ai_characters.append({
                            'name': char_info['name'],
                            'role': char_info.get('role', 'Unknown'),
                            'source': 'AI_analysis'
                        })

        batch_result = {
            'batch_id': batch_id,
            'volume_id': volume_id,
            'chunks_processed': len(chunks),
            'total_iterations': len(iterations),
            'final_satisfaction': final_satisfaction,
            'processing_stage': processing_context.processing_stage.value,
            'meaningful_improvements': meaningful_improvements,
            'total_tokens': total_tokens,
            'total_cost': total_tokens * 0.00002,
            'characters_detected': character_names,  # Registry-based characters
            'ai_characters': ai_characters,     # AI-discovered characters
            'iterations': iterations,
            'early_terminated': no_improvement_count >= self.config.early_termination_patience
        }
        
        # Debug: Show batch summary
        print(f"\n📋 BATCH SUMMARY:")
        print(f"   🎯 Final Satisfaction: {final_satisfaction:.3f}")
        print(f"   🔄 Iterations: {len(iterations)}")
        print(f"   💰 Cost: ${batch_result['total_cost']:.4f}")
        print(f"   👥 Registry Characters: {character_names}")
        if ai_characters:
            ai_char_names = [char['name'] for char in ai_characters]
            print(f"   🤖 AI Characters: {ai_char_names}")
            print(f"   📊 Total Unique Characters: {len(set(character_names + ai_char_names))}")
        print(f"   ⏹️ Early Terminated: {batch_result['early_terminated']}")
        
        return batch_result
    
    def _build_unified_prompt(self, chunks: List[Dict], context: ProcessingContext, 
                            existing_characters: List[Dict], iteration: int, current_satisfaction: float) -> str:
        """Build unified prompt combining all processor methods"""
        
        # Prepare content
        batch_content = "\n\n".join([
            f"片段{i+1}: {chunk['content'][:400]}" 
            for i, chunk in enumerate(chunks)
        ])
        
        # Stage-specific instructions
        stage_instructions = {
            ProcessingStage.BEGINNING: "分析小说开头，重点关注角色介绍和设定建立。",
            ProcessingStage.EARLY: "分析小说早期，重点关注角色发展和关系建立。", 
            ProcessingStage.MIDDLE: "分析小说中段，重点关注复杂情节和角色关系。",
            ProcessingStage.CLIMAX: "分析小说高潮，重点关注关键事件和转折。",
            ProcessingStage.ENDING: "分析小说结尾，重点关注情节解决和角色成长。"
        }
        
        stage_instruction = stage_instructions.get(context.processing_stage, "分析以下文本")
        
        # Context info with character registry
        context_info = ""
        if existing_characters:
            char_context = ', '.join([f"{char['name']}({char['role']})" for char in existing_characters[:5]])
            context_info = f"\n角色注册表 ({len(existing_characters)} total): {char_context}"
            if len(existing_characters) > 5:
                context_info += f" + {len(existing_characters) - 5} more"
        
        # Function call instructions for character and timeline management
        function_instructions = """

🤖 重要提示：请在analysis中包含function_calls来管理角色和时间线！

可用功能调用:

角色管理:
- register_character: {"name": "角色名", "role": "角色职责", "description": "描述", "confidence": 0.8}
- update_character: {"name": "现有角色名", "new_role": "更新职责", "confidence": 0.9}

时间线管理:
- create_timeline_event: {"description": "事件描述", "event_type": "dialogue|action|revelation", "primary_actors": ["角色A"], "importance_score": 0.8, "chronological_order": 1}
- update_timeline_event: {"event_id": "evt_123", "new_importance": 0.9, "update_character_states": {"角色A": {"emotional_state": "愤怒"}}}
- establish_causality: {"cause_event_id": "evt_123", "effect_event_id": "evt_456", "causality_type": "direct_cause", "confidence": 0.8, "reasoning": "因果解释"}
- update_character_timeline: {"character_name": "角色A", "event_id": "evt_123", "state_changes": {"knowledge_gained": "新信息", "emotional_shift": "平静 -> 震惊"}}
- query_timeline_context: {"query_type": "character_motivation", "character_name": "角色A", "context_request": "为什么做出这个决定？"}

示例格式：
"function_calls": [
  {"action": "register_character", "params": {"name": "上条当麻", "role": "主角", "confidence": 0.9}},
  {"action": "create_timeline_event", "params": {"description": "上条发现真相", "event_type": "revelation", "primary_actors": ["上条当麻"], "importance_score": 0.9}}
]"""
        
        prompt = f"""
{stage_instruction}

处理信息:
- 卷: {context.volume_id}
- 批次: {context.batch_position}/{context.total_batches} ({context.progress_percentage:.1f}%)
- 阶段: {context.processing_stage.value}
- 迭代: {iteration}
- 当前满意度: {current_satisfaction:.2f}{context_info}

文本内容:
{batch_content}{function_instructions}

请分析以上内容并返回JSON格式结果。必须包含以下字段:

{{
  "characters": [
    {{
      "name": "角色名",
      "role": "角色作用",
      "key_actions": ["关键行为"]
    }}
  ],
  "events": [
    {{
      "description": "事件描述",  
      "importance": 0.8,
      "timeline": "时间位置"
    }}
  ],
  "analysis": {{
    "main_theme": "主要主题",
    "plot_significance": "情节意义",
    "function_calls": [
      {{"action": "register_character", "params": {{"name": "新角色", "role": "职责", "confidence": 0.8}}}}
    ]
  }},
  "satisfaction_level": 0.85,
  "issues": ["问题1", "问题2"],
  "improvements": ["改进1", "改进2"]
}}

请确保返回有效的JSON格式，satisfaction_level必须是0到1之间的数值。
如果发现新角色，请在function_calls中使用register_character注册。
如果识别重要事件，请在function_calls中使用create_timeline_event创建时间线事件。
如果发现事件间的因果关系，请使用establish_causality建立联系。
"""
        
        return prompt.strip()
    
    async def _query_deepseek_with_retry(self, prompt: str) -> Dict[str, Any]:
        """Query DeepSeek with retry logic"""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                messages = [{"role": "user", "content": prompt}]
                response = await self.deepseek_client.generate_character_response(
                    messages=messages,
                    max_tokens=1500,
                    temperature=0.0,
                    top_p=1.0,
                    frequency_penalty=0.0
                )
                
                if response.get("success"):
                    prompt_tokens = len(prompt.encode('utf-8')) // 4
                    response_tokens = len(response["response"].encode('utf-8')) // 4
                    total_tokens = prompt_tokens + response_tokens
                    
                    return {
                        'success': True,
                        'response': response["response"],
                        'tokens_used': total_tokens
                    }
                else:
                    logger.warning(f"DeepSeek request failed (attempt {attempt + 1}): {response.get('error')}")
                    
            except Exception as e:
                logger.error(f"DeepSeek query error (attempt {attempt + 1}): {e}")
        
        return {
            'success': False,
            'error': 'Max retries exceeded',
            'tokens_used': 0
        }
    
    def _parse_json_response(self, response: str, iteration_num: int) -> IterationResult:
        """Robust JSON parsing with fallback strategies"""
        try:
            # Clean response
            json_text = response.strip()
            
            # Remove markdown code blocks
            if "```json" in json_text:
                start = json_text.find("```json") + 7
                end = json_text.find("```", start)
                if end == -1:
                    end = len(json_text)
                json_text = json_text[start:end].strip()
            elif json_text.startswith('```') and json_text.endswith('```'):
                json_text = json_text[3:-3].strip()
            
            # Find JSON boundaries
            if not json_text.startswith('{'):
                brace_start = json_text.find('{')
                if brace_start != -1:
                    json_text = json_text[brace_start:]
            
            if not json_text.endswith('}'):
                brace_end = json_text.rfind('}')
                if brace_end != -1:
                    json_text = json_text[:brace_end + 1]
            
            # Parse JSON
            parsed = json.loads(json_text)
            
            # Extract satisfaction level
            satisfaction = parsed.get('satisfaction_level', 0.5)
            if not isinstance(satisfaction, (int, float)) or satisfaction < 0 or satisfaction > 1:
                satisfaction = 0.5
            
            return IterationResult(
                iteration_num=iteration_num,
                analysis={
                    'characters': parsed.get('characters', []),
                    'events': parsed.get('events', []),
                    'analysis': parsed.get('analysis', {})
                },
                confidence_scores={'overall': satisfaction},
                satisfaction_level=satisfaction,
                identified_issues=parsed.get('issues', []),
                refinement_requests=parsed.get('improvements', []),
                reasoning_trace=[f"Successfully parsed JSON response with {len(parsed)} fields"]
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
            
            # Fallback: extract satisfaction using regex
            satisfaction_match = re.search(r'"satisfaction_level"\s*:\s*([0-9.]+)', response)
            if satisfaction_match:
                try:
                    satisfaction = float(satisfaction_match.group(1))
                    satisfaction = max(0.0, min(1.0, satisfaction))
                except ValueError:
                    satisfaction = 0.5
            else:
                satisfaction = 0.5
            
            return IterationResult(
                iteration_num=iteration_num,
                analysis={'fallback_parsing': True},
                confidence_scores={'fallback': satisfaction},
                satisfaction_level=satisfaction,
                identified_issues=["JSON parsing failed - using fallback"],
                refinement_requests=["Fix JSON format in response"],
                reasoning_trace=[f"Fallback parsing due to: {str(e)}"]
            )
        
        except Exception as e:
            logger.error(f"Unexpected parsing error: {e}")
            return IterationResult(
                iteration_num=iteration_num,
                analysis={},
                confidence_scores={},
                satisfaction_level=0.3,
                identified_issues=["Critical parsing failure"],
                refinement_requests=["Complete retry needed"],
                reasoning_trace=[f"Critical error: {str(e)}"]
            )

class UnifiedNovelProcessor:
    """Main unified processor combining all previous processors"""
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.data_loader = QdrantDataLoader(self.config)
        self._initialize_database()
        self.reasoner = UnifiedReasonerEngine(self.config, processor=self)  # Pass self reference
        self._validate_environment()
        
        print("🚀 UNIFIED NOVEL PROCESSOR INITIALIZED")
        print("=" * 60)
        print(f"   🎯 Mode: {self.config.mode.value}")
        print(f"   💾 Database: {self.config.database_path}")
        print(f"   🔗 Qdrant: {'Enabled' if self.config.use_qdrant else 'Disabled'}")
        print(f"   🌐 Qdrant URL: {self.config.qdrant_url}")
        print(f"   📚 Collection: {self.config.collection_name}")
        print("=" * 60)
    
    def _validate_environment(self):
        """Validate that the environment configuration is correct"""
        if self.config.use_qdrant:
            try:
                from qdrant_client import QdrantClient
                client = QdrantClient(url=self.config.qdrant_url, verify=False, check_compatibility=False)
                collections = client.get_collections()
                collection_names = [c.name for c in collections.collections]
                
                if self.config.collection_name in collection_names:
                    print(f"✅ Environment validation passed")
                    print(f"   🔍 Found collection '{self.config.collection_name}' in Qdrant")
                    print(f"   📋 Available collections: {collection_names}")
                else:
                    print(f"⚠️  WARNING: Collection '{self.config.collection_name}' not found!")
                    print(f"   📋 Available collections: {collection_names}")
                    print(f"   💡 Check your collection name or data migration")
                
            except Exception as e:
                print(f"⚠️  WARNING: Could not validate Qdrant environment: {e}")
                print(f"   🔧 Check .env file and ensure Qdrant is running at {self.config.qdrant_url}")
    
    def _initialize_database(self):
        """Initialize unified database (SQLite or PostgreSQL)"""
        if self.config.use_postgres and POSTGRES_AVAILABLE:
            self._initialize_postgres_database()
        else:
            self._initialize_sqlite_database()
    
    def _initialize_postgres_database(self):
        """Initialize PostgreSQL database with automatic table creation"""
        try:
            print("🐘 Initializing PostgreSQL database...")
            conn = psycopg2.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_db,
                user=self.config.postgres_user,
                password=self.config.postgres_password
            )
            cursor = conn.cursor()
            
            # Create tables automatically
            print("📋 Creating database tables if they don't exist...")
            
            # Create unified_results table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS unified_results (
                id SERIAL PRIMARY KEY,
                volume_id INTEGER NOT NULL,
                batch_id INTEGER NOT NULL,
                chunks_processed INTEGER NOT NULL,
                total_iterations INTEGER NOT NULL,
                final_satisfaction REAL NOT NULL,
                processing_stage TEXT NOT NULL,
                processing_mode TEXT NOT NULL,
                meaningful_improvements INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                total_cost REAL NOT NULL DEFAULT 0.0,
                early_terminated BOOLEAN DEFAULT FALSE,
                characters_detected TEXT,
                processing_metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_unified_results_volume_id ON unified_results(volume_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_unified_results_created_at ON unified_results(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_unified_results_processing_stage ON unified_results(processing_stage)')
            
            # Create unique constraint (use DO $$ block to handle "already exists" error)
            cursor.execute('''
            DO $$
            BEGIN
                ALTER TABLE unified_results ADD CONSTRAINT unique_volume_batch UNIQUE (volume_id, batch_id);
            EXCEPTION
                WHEN duplicate_table THEN NULL;
            END $$
            ''')
            
            # Create character_analysis table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS character_analysis (
                id SERIAL PRIMARY KEY,
                volume_id INTEGER NOT NULL,
                batch_id INTEGER NOT NULL,
                character_name TEXT NOT NULL,
                character_role TEXT,
                key_actions TEXT[],
                appearance_frequency INTEGER DEFAULT 1,
                confidence_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(volume_id, batch_id, character_name)
            )
            ''')
            
            # Create event_analysis table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS event_analysis (
                id SERIAL PRIMARY KEY,
                volume_id INTEGER NOT NULL,
                batch_id INTEGER NOT NULL,
                event_description TEXT NOT NULL,
                importance_score REAL DEFAULT 0.0,
                timeline_position TEXT,
                event_metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create character_registry table for agentic character management
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS character_registry (
                id SERIAL PRIMARY KEY,
                volume_id INTEGER NOT NULL,
                character_name TEXT NOT NULL,
                character_role TEXT,
                character_description TEXT,
                confidence_score REAL DEFAULT 0.5,
                appearance_count INTEGER DEFAULT 1,
                first_appearance TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(volume_id, character_name)
            )
            ''')
            
            # Create timeline_events table for timeline system
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS timeline_events (
                event_id VARCHAR PRIMARY KEY,
                volume_id INTEGER NOT NULL,
                batch_id INTEGER NOT NULL,
                chunk_ids INTEGER[],
                chronological_order INTEGER,
                relative_time TEXT,
                time_confidence REAL DEFAULT 0.5,
                temporal_markers TEXT[],
                description TEXT NOT NULL,
                event_type VARCHAR,
                importance_score REAL DEFAULT 0.0,
                primary_actors TEXT[],
                affected_characters TEXT[],
                character_states_before JSONB,
                character_states_after JSONB,
                caused_by_events TEXT[],
                causes_events TEXT[],
                plot_thread VARCHAR,
                narrative_function VARCHAR,
                foreshadowing_links TEXT[],
                confidence_level REAL DEFAULT 0.5,
                validation_status VARCHAR DEFAULT 'pending',
                created_by VARCHAR DEFAULT 'ai_analysis',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create event_causality table for causal relationships
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS event_causality (
                id SERIAL PRIMARY KEY,
                cause_event_id VARCHAR NOT NULL,
                effect_event_id VARCHAR NOT NULL,
                causality_type VARCHAR NOT NULL,
                confidence REAL DEFAULT 0.5,
                reasoning TEXT,
                influence_strength REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(cause_event_id, effect_event_id)
            )
            ''')
            
            # Create character_event_participation table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS character_event_participation (
                id SERIAL PRIMARY KEY,
                character_name VARCHAR NOT NULL,
                event_id VARCHAR NOT NULL,
                participation_type VARCHAR NOT NULL,
                character_state_before JSONB,
                character_state_after JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(character_name, event_id)
            )
            ''')
            
            # Create motivation_chains table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS motivation_chains (
                chain_id VARCHAR PRIMARY KEY,
                character_name VARCHAR NOT NULL,
                motivation_type VARCHAR NOT NULL,
                trigger_event_id VARCHAR,
                culmination_event_id VARCHAR,
                chain_strength REAL DEFAULT 0.5,
                duration VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create indexes for timeline performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timeline_events_volume ON timeline_events(volume_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timeline_events_chronological ON timeline_events(chronological_order)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timeline_events_type ON timeline_events(event_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_causality_cause ON event_causality(cause_event_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_causality_effect ON event_causality(effect_event_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_character_participation ON character_event_participation(character_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_motivation_chains_character ON motivation_chains(character_name)')
            
            conn.commit()
            conn.close()
            
            print("✅ PostgreSQL database and tables ready!")
            logger.info(f"PostgreSQL database connected: {self.config.postgres_host}:{self.config.postgres_port}")
            
        except Exception as e:
            print(f"❌ PostgreSQL initialization failed: {e}")
            logger.error(f"PostgreSQL connection failed: {e}")
            logger.info("Falling back to SQLite")
            self.config.use_postgres = False
            self._initialize_sqlite_database()
    
    def _initialize_sqlite_database(self):
        """Initialize SQLite database"""
        db_dir = os.path.dirname(self.config.database_path)
        if db_dir:  # Only create directory if path has a directory component
            os.makedirs(db_dir, exist_ok=True)
        
        conn = sqlite3.connect(self.config.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS unified_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            volume_id INTEGER,
            batch_id INTEGER,
            chunks_processed INTEGER,
            total_iterations INTEGER,
            final_satisfaction REAL,
            processing_stage TEXT,
            processing_mode TEXT,
            meaningful_improvements INTEGER,
            total_tokens INTEGER,
            total_cost REAL,
            early_terminated BOOLEAN,
            characters_detected TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create character_registry table for agentic character management
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS character_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            volume_id INTEGER NOT NULL,
            character_name TEXT NOT NULL,
            character_role TEXT,
            character_description TEXT,
            confidence_score REAL DEFAULT 0.5,
            appearance_count INTEGER DEFAULT 1,
            first_appearance TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(volume_id, character_name)
        )
        ''')
        
        # Create timeline_events table for timeline system
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS timeline_events (
            event_id TEXT PRIMARY KEY,
            volume_id INTEGER NOT NULL,
            batch_id INTEGER NOT NULL,
            chunk_ids TEXT,
            chronological_order INTEGER,
            relative_time TEXT,
            time_confidence REAL DEFAULT 0.5,
            temporal_markers TEXT,
            description TEXT NOT NULL,
            event_type TEXT,
            importance_score REAL DEFAULT 0.0,
            primary_actors TEXT,
            affected_characters TEXT,
            character_states_before TEXT,
            character_states_after TEXT,
            caused_by_events TEXT,
            causes_events TEXT,
            plot_thread TEXT,
            narrative_function TEXT,
            foreshadowing_links TEXT,
            confidence_level REAL DEFAULT 0.5,
            validation_status TEXT DEFAULT 'pending',
            created_by TEXT DEFAULT 'ai_analysis',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create event_causality table for causal relationships
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS event_causality (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cause_event_id TEXT NOT NULL,
            effect_event_id TEXT NOT NULL,
            causality_type TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            reasoning TEXT,
            influence_strength REAL DEFAULT 0.5,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(cause_event_id, effect_event_id)
        )
        ''')
        
        # Create character_event_participation table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS character_event_participation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            character_name TEXT NOT NULL,
            event_id TEXT NOT NULL,
            participation_type TEXT NOT NULL,
            character_state_before TEXT,
            character_state_after TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(character_name, event_id)
        )
        ''')
        
        # Create motivation_chains table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS motivation_chains (
            chain_id TEXT PRIMARY KEY,
            character_name TEXT NOT NULL,
            motivation_type TEXT NOT NULL,
            trigger_event_id TEXT,
            culmination_event_id TEXT,
            chain_strength REAL DEFAULT 0.5,
            duration TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create indexes for timeline performance  
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timeline_events_volume ON timeline_events(volume_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timeline_events_chronological ON timeline_events(chronological_order)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timeline_events_type ON timeline_events(event_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_causality_cause ON event_causality(cause_event_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_causality_effect ON event_causality(effect_event_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_character_participation ON character_event_participation(character_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_motivation_chains_character ON motivation_chains(character_name)')
        
        conn.commit()
        conn.close()
        logger.info(f"SQLite database initialized: {self.config.database_path}")
    
    # ============================================================================
    # TIMELINE DATABASE ACCESS LAYER
    # ============================================================================
    
    def save_timeline_event(self, event: TimelineEvent) -> bool:
        """Save a timeline event to the database"""
        try:
            if self.config.use_postgres and POSTGRES_AVAILABLE:
                return self._save_timeline_event_postgres(event)
            else:
                return self._save_timeline_event_sqlite(event)
        except Exception as e:
            logger.error(f"Failed to save timeline event {event.event_id}: {e}")
            return False
    
    def _save_timeline_event_postgres(self, event: TimelineEvent) -> bool:
        """Save timeline event to PostgreSQL"""
        try:
            conn = psycopg2.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_db,
                user=self.config.postgres_user,
                password=self.config.postgres_password
            )
            cursor = conn.cursor()
            
            # Convert complex fields to PostgreSQL format
            chunk_ids_array = event.chunk_ids if event.chunk_ids else []
            temporal_markers_array = event.temporal_markers if event.temporal_markers else []
            primary_actors_array = event.primary_actors if event.primary_actors else []
            affected_characters_array = event.affected_characters if event.affected_characters else []
            caused_by_events_array = event.caused_by_events if event.caused_by_events else []
            causes_events_array = event.causes_events if event.causes_events else []
            foreshadowing_links_array = event.foreshadowing_links if event.foreshadowing_links else []
            
            # Convert character states to JSONB
            states_before_json = {}
            if event.character_states_before:
                for char_name, state in event.character_states_before.items():
                    states_before_json[char_name] = {
                        'emotional_state': state.emotional_state,
                        'knowledge_level': state.knowledge_level,
                        'relationships': state.relationships,
                        'goals': state.goals,
                        'capabilities': state.capabilities
                    }
            
            states_after_json = {}
            if event.character_states_after:
                for char_name, state in event.character_states_after.items():
                    states_after_json[char_name] = {
                        'emotional_state': state.emotional_state,
                        'knowledge_level': state.knowledge_level,
                        'relationships': state.relationships,
                        'goals': state.goals,
                        'capabilities': state.capabilities
                    }
            
            cursor.execute("""
                INSERT INTO timeline_events (
                    event_id, volume_id, batch_id, chunk_ids, chronological_order,
                    relative_time, time_confidence, temporal_markers, description,
                    event_type, importance_score, primary_actors, affected_characters,
                    character_states_before, character_states_after, caused_by_events,
                    causes_events, plot_thread, narrative_function, foreshadowing_links,
                    confidence_level, validation_status, created_by, created_at, last_updated
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (event_id) DO UPDATE SET
                    chronological_order = EXCLUDED.chronological_order,
                    relative_time = EXCLUDED.relative_time,
                    time_confidence = EXCLUDED.time_confidence,
                    temporal_markers = EXCLUDED.temporal_markers,
                    description = EXCLUDED.description,
                    event_type = EXCLUDED.event_type,
                    importance_score = EXCLUDED.importance_score,
                    primary_actors = EXCLUDED.primary_actors,
                    affected_characters = EXCLUDED.affected_characters,
                    character_states_before = EXCLUDED.character_states_before,
                    character_states_after = EXCLUDED.character_states_after,
                    caused_by_events = EXCLUDED.caused_by_events,
                    causes_events = EXCLUDED.causes_events,
                    plot_thread = EXCLUDED.plot_thread,
                    narrative_function = EXCLUDED.narrative_function,
                    foreshadowing_links = EXCLUDED.foreshadowing_links,
                    confidence_level = EXCLUDED.confidence_level,
                    validation_status = EXCLUDED.validation_status,
                    last_updated = EXCLUDED.last_updated
            """, (
                event.event_id, event.volume_id, event.batch_id, chunk_ids_array, event.chronological_order,
                event.relative_time, event.time_confidence, temporal_markers_array, event.description,
                event.event_type.value, event.importance_score, primary_actors_array, affected_characters_array,
                json.dumps(states_before_json), json.dumps(states_after_json), caused_by_events_array,
                causes_events_array, event.plot_thread, event.narrative_function.value, foreshadowing_links_array,
                event.confidence_level, event.validation_status.value, event.created_by, 
                event.created_at, event.last_updated
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"PostgreSQL timeline event save failed: {e}")
            return False
    
    def _save_timeline_event_sqlite(self, event: TimelineEvent) -> bool:
        """Save timeline event to SQLite"""
        try:
            conn = sqlite3.connect(self.config.database_path)
            cursor = conn.cursor()
            
            # Convert arrays and objects to JSON strings for SQLite
            chunk_ids_json = json.dumps(event.chunk_ids) if event.chunk_ids else "[]"
            temporal_markers_json = json.dumps(event.temporal_markers) if event.temporal_markers else "[]"
            primary_actors_json = json.dumps(event.primary_actors) if event.primary_actors else "[]"
            affected_characters_json = json.dumps(event.affected_characters) if event.affected_characters else "[]"
            caused_by_events_json = json.dumps(event.caused_by_events) if event.caused_by_events else "[]"
            causes_events_json = json.dumps(event.causes_events) if event.causes_events else "[]"
            foreshadowing_links_json = json.dumps(event.foreshadowing_links) if event.foreshadowing_links else "[]"
            
            # Convert character states to JSON
            states_before_json = "{}"
            if event.character_states_before:
                states_before_dict = {}
                for char_name, state in event.character_states_before.items():
                    states_before_dict[char_name] = {
                        'emotional_state': state.emotional_state,
                        'knowledge_level': state.knowledge_level,
                        'relationships': state.relationships,
                        'goals': state.goals,
                        'capabilities': state.capabilities
                    }
                states_before_json = json.dumps(states_before_dict)
            
            states_after_json = "{}"
            if event.character_states_after:
                states_after_dict = {}
                for char_name, state in event.character_states_after.items():
                    states_after_dict[char_name] = {
                        'emotional_state': state.emotional_state,
                        'knowledge_level': state.knowledge_level,
                        'relationships': state.relationships,
                        'goals': state.goals,
                        'capabilities': state.capabilities
                    }
                states_after_json = json.dumps(states_after_dict)
            
            cursor.execute("""
                INSERT OR REPLACE INTO timeline_events (
                    event_id, volume_id, batch_id, chunk_ids, chronological_order,
                    relative_time, time_confidence, temporal_markers, description,
                    event_type, importance_score, primary_actors, affected_characters,
                    character_states_before, character_states_after, caused_by_events,
                    causes_events, plot_thread, narrative_function, foreshadowing_links,
                    confidence_level, validation_status, created_by, created_at, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id, event.volume_id, event.batch_id, chunk_ids_json, event.chronological_order,
                event.relative_time, event.time_confidence, temporal_markers_json, event.description,
                event.event_type.value, event.importance_score, primary_actors_json, affected_characters_json,
                states_before_json, states_after_json, caused_by_events_json,
                causes_events_json, event.plot_thread, event.narrative_function.value, foreshadowing_links_json,
                event.confidence_level, event.validation_status.value, event.created_by,
                event.created_at.isoformat() if event.created_at else None,
                event.last_updated.isoformat() if event.last_updated else None
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"SQLite timeline event save failed: {e}")
            return False
    
    def load_timeline_event(self, event_id: str) -> Optional[TimelineEvent]:
        """Load a timeline event from the database"""
        try:
            if self.config.use_postgres and POSTGRES_AVAILABLE:
                return self._load_timeline_event_postgres(event_id)
            else:
                return self._load_timeline_event_sqlite(event_id)
        except Exception as e:
            logger.error(f"Failed to load timeline event {event_id}: {e}")
            return None
    
    def _load_timeline_event_postgres(self, event_id: str) -> Optional[TimelineEvent]:
        """Load timeline event from PostgreSQL"""
        try:
            conn = psycopg2.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_db,
                user=self.config.postgres_user,
                password=self.config.postgres_password
            )
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT event_id, volume_id, batch_id, chunk_ids, chronological_order,
                       relative_time, time_confidence, temporal_markers, description,
                       event_type, importance_score, primary_actors, affected_characters,
                       character_states_before, character_states_after, caused_by_events,
                       causes_events, plot_thread, narrative_function, foreshadowing_links,
                       confidence_level, validation_status, created_by, created_at, last_updated
                FROM timeline_events WHERE event_id = %s
            """, (event_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            return self._row_to_timeline_event(row, is_postgres=True)
            
        except Exception as e:
            logger.error(f"PostgreSQL timeline event load failed: {e}")
            return None
    
    def _load_timeline_event_sqlite(self, event_id: str) -> Optional[TimelineEvent]:
        """Load timeline event from SQLite"""
        try:
            conn = sqlite3.connect(self.config.database_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT event_id, volume_id, batch_id, chunk_ids, chronological_order,
                       relative_time, time_confidence, temporal_markers, description,
                       event_type, importance_score, primary_actors, affected_characters,
                       character_states_before, character_states_after, caused_by_events,
                       causes_events, plot_thread, narrative_function, foreshadowing_links,
                       confidence_level, validation_status, created_by, created_at, last_updated
                FROM timeline_events WHERE event_id = ?
            """, (event_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            return self._row_to_timeline_event(row, is_postgres=False)
            
        except Exception as e:
            logger.error(f"SQLite timeline event load failed: {e}")
            return None
    
    def _row_to_timeline_event(self, row, is_postgres: bool) -> TimelineEvent:
        """Convert database row to TimelineEvent object"""
        (event_id, volume_id, batch_id, chunk_ids, chronological_order,
         relative_time, time_confidence, temporal_markers, description,
         event_type, importance_score, primary_actors, affected_characters,
         character_states_before, character_states_after, caused_by_events,
         causes_events, plot_thread, narrative_function, foreshadowing_links,
         confidence_level, validation_status, created_by, created_at, last_updated) = row
        
        # Parse arrays/JSON based on database type
        if is_postgres:
            # PostgreSQL returns arrays directly
            chunk_ids_list = chunk_ids if chunk_ids else []
            temporal_markers_list = temporal_markers if temporal_markers else []
            primary_actors_list = primary_actors if primary_actors else []
            affected_characters_list = affected_characters if affected_characters else []
            caused_by_events_list = caused_by_events if caused_by_events else []
            causes_events_list = causes_events if causes_events else []
            foreshadowing_links_list = foreshadowing_links if foreshadowing_links else []
        else:
            # SQLite returns JSON strings that need parsing
            chunk_ids_list = json.loads(chunk_ids) if chunk_ids else []
            temporal_markers_list = json.loads(temporal_markers) if temporal_markers else []
            primary_actors_list = json.loads(primary_actors) if primary_actors else []
            affected_characters_list = json.loads(affected_characters) if affected_characters else []
            caused_by_events_list = json.loads(caused_by_events) if caused_by_events else []
            causes_events_list = json.loads(causes_events) if causes_events else []
            foreshadowing_links_list = json.loads(foreshadowing_links) if foreshadowing_links else []
        
        # Parse character states
        states_before_dict = {}
        if character_states_before:
            states_data = json.loads(character_states_before) if isinstance(character_states_before, str) else character_states_before
            for char_name, state_data in states_data.items():
                states_before_dict[char_name] = CharacterState(
                    emotional_state=state_data.get('emotional_state', ''),
                    knowledge_level=state_data.get('knowledge_level', {}),
                    relationships=state_data.get('relationships', {}),
                    goals=state_data.get('goals', []),
                    capabilities=state_data.get('capabilities', [])
                )
        
        states_after_dict = {}
        if character_states_after:
            states_data = json.loads(character_states_after) if isinstance(character_states_after, str) else character_states_after
            for char_name, state_data in states_data.items():
                states_after_dict[char_name] = CharacterState(
                    emotional_state=state_data.get('emotional_state', ''),
                    knowledge_level=state_data.get('knowledge_level', {}),
                    relationships=state_data.get('relationships', {}),
                    goals=state_data.get('goals', []),
                    capabilities=state_data.get('capabilities', [])
                )
        
        # Parse timestamps
        created_at_dt = None
        if created_at:
            if isinstance(created_at, str):
                created_at_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            else:
                created_at_dt = created_at
        
        last_updated_dt = None
        if last_updated:
            if isinstance(last_updated, str):
                last_updated_dt = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
            else:
                last_updated_dt = last_updated
        
        return TimelineEvent(
            event_id=event_id,
            volume_id=volume_id,
            batch_id=batch_id,
            chunk_ids=chunk_ids_list,
            chronological_order=chronological_order,
            relative_time=relative_time or "",
            time_confidence=time_confidence or 0.5,
            temporal_markers=temporal_markers_list,
            description=description or "",
            event_type=EventType(event_type) if event_type else EventType.ACTION,
            importance_score=importance_score or 0.0,
            primary_actors=primary_actors_list,
            affected_characters=affected_characters_list,
            character_states_before=states_before_dict,
            character_states_after=states_after_dict,
            caused_by_events=caused_by_events_list,
            causes_events=causes_events_list,
            plot_thread=plot_thread or "main_plot",
            narrative_function=NarrativeFunction(narrative_function) if narrative_function else NarrativeFunction.RISING_ACTION,
            foreshadowing_links=foreshadowing_links_list,
            confidence_level=confidence_level or 0.5,
            validation_status=ValidationStatus(validation_status) if validation_status else ValidationStatus.PENDING,
            created_by=created_by or "ai_analysis",
            created_at=created_at_dt,
            last_updated=last_updated_dt
        )
    
    def save_causal_relationship(self, cause_event_id: str, effect_event_id: str, 
                                causality_type: CausalityType, confidence: float = 0.5,
                                reasoning: str = "", influence_strength: float = 0.5) -> bool:
        """Save a causal relationship between two events"""
        try:
            if self.config.use_postgres and POSTGRES_AVAILABLE:
                return self._save_causal_relationship_postgres(
                    cause_event_id, effect_event_id, causality_type, confidence, reasoning, influence_strength
                )
            else:
                return self._save_causal_relationship_sqlite(
                    cause_event_id, effect_event_id, causality_type, confidence, reasoning, influence_strength
                )
        except Exception as e:
            logger.error(f"Failed to save causal relationship {cause_event_id} -> {effect_event_id}: {e}")
            return False
    
    def _save_causal_relationship_postgres(self, cause_event_id: str, effect_event_id: str,
                                         causality_type: CausalityType, confidence: float,
                                         reasoning: str, influence_strength: float) -> bool:
        """Save causal relationship to PostgreSQL"""
        try:
            conn = psycopg2.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_db,
                user=self.config.postgres_user,
                password=self.config.postgres_password
            )
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO event_causality (
                    cause_event_id, effect_event_id, causality_type, confidence, reasoning, influence_strength
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (cause_event_id, effect_event_id) DO UPDATE SET
                    causality_type = EXCLUDED.causality_type,
                    confidence = EXCLUDED.confidence,
                    reasoning = EXCLUDED.reasoning,
                    influence_strength = EXCLUDED.influence_strength
            """, (cause_event_id, effect_event_id, causality_type.value, confidence, reasoning, influence_strength))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"PostgreSQL causal relationship save failed: {e}")
            return False
    
    def _save_causal_relationship_sqlite(self, cause_event_id: str, effect_event_id: str,
                                       causality_type: CausalityType, confidence: float,
                                       reasoning: str, influence_strength: float) -> bool:
        """Save causal relationship to SQLite"""
        try:
            conn = sqlite3.connect(self.config.database_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO event_causality (
                    cause_event_id, effect_event_id, causality_type, confidence, reasoning, influence_strength
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (cause_event_id, effect_event_id, causality_type.value, confidence, reasoning, influence_strength))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"SQLite causal relationship save failed: {e}")
            return False
    
    def load_events_by_volume(self, volume_id: int, limit: Optional[int] = None) -> List[TimelineEvent]:
        """Load all timeline events for a specific volume"""
        try:
            if self.config.use_postgres and POSTGRES_AVAILABLE:
                return self._load_events_by_volume_postgres(volume_id, limit)
            else:
                return self._load_events_by_volume_sqlite(volume_id, limit)
        except Exception as e:
            logger.error(f"Failed to load events for volume {volume_id}: {e}")
            return []
    
    def _load_events_by_volume_postgres(self, volume_id: int, limit: Optional[int] = None) -> List[TimelineEvent]:
        """Load events by volume from PostgreSQL"""
        try:
            conn = psycopg2.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_db,
                user=self.config.postgres_user,
                password=self.config.postgres_password
            )
            cursor = conn.cursor()
            
            query = """
                SELECT event_id, volume_id, batch_id, chunk_ids, chronological_order,
                       relative_time, time_confidence, temporal_markers, description,
                       event_type, importance_score, primary_actors, affected_characters,
                       character_states_before, character_states_after, caused_by_events,
                       causes_events, plot_thread, narrative_function, foreshadowing_links,
                       confidence_level, validation_status, created_by, created_at, last_updated
                FROM timeline_events WHERE volume_id = %s
                ORDER BY chronological_order ASC, created_at ASC
            """
            
            params = [volume_id]
            if limit:
                query += " LIMIT %s"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            return [self._row_to_timeline_event(row, is_postgres=True) for row in rows]
            
        except Exception as e:
            logger.error(f"PostgreSQL events by volume load failed: {e}")
            return []
    
    def _load_events_by_volume_sqlite(self, volume_id: int, limit: Optional[int] = None) -> List[TimelineEvent]:
        """Load events by volume from SQLite"""
        try:
            conn = sqlite3.connect(self.config.database_path)
            cursor = conn.cursor()
            
            query = """
                SELECT event_id, volume_id, batch_id, chunk_ids, chronological_order,
                       relative_time, time_confidence, temporal_markers, description,
                       event_type, importance_score, primary_actors, affected_characters,
                       character_states_before, character_states_after, caused_by_events,
                       causes_events, plot_thread, narrative_function, foreshadowing_links,
                       confidence_level, validation_status, created_by, created_at, last_updated
                FROM timeline_events WHERE volume_id = ?
                ORDER BY chronological_order ASC, created_at ASC
            """
            
            params = [volume_id]
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            return [self._row_to_timeline_event(row, is_postgres=False) for row in rows]
            
        except Exception as e:
            logger.error(f"SQLite events by volume load failed: {e}")
            return []

    async def process_volumes(self, volume_ids: List[int]) -> Dict[str, Any]:
        """Process specified volumes with unified logic"""
        print(f"\n🚀 PROCESSING VOLUMES: {volume_ids}")
        print("=" * 60)
        print(f"🕐 Start time: {datetime.now()}")
        print(f"🎯 Mode: {self.config.mode.value}")
        print("=" * 60)
        
        start_time = datetime.now()
        all_results = []
        
        try:
            # Load data for all volumes
            volume_chunks = self.data_loader.load_volume_chunks(volume_ids)
            
            if not volume_chunks:
                print("❌ No chunks loaded. Check Qdrant connection or volume IDs.")
                return {'success': False, 'error': 'No data loaded'}
            
            # Process each volume
            for volume_id in volume_ids:
                if volume_id not in volume_chunks:
                    print(f"⚠️ Volume {volume_id} not found, skipping")
                    continue
                
                chunks = volume_chunks[volume_id]
                volume_results = await self._process_single_volume(volume_id, chunks)
                all_results.extend(volume_results)
            
            # Generate summary
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            summary = self._generate_summary(all_results, volume_ids, processing_time)
            
            # Display results
            self._display_results(summary)
            
            # Save report
            self._save_processing_report(summary, volume_ids)
            
            return summary
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
        
        finally:
            if self.reasoner.deepseek_client.session:
                await self.reasoner.deepseek_client.close()
    
    async def _process_single_volume(self, volume_id: int, chunks: List[Dict]) -> List[Dict]:
        """Process a single volume"""
        print(f"\n📖 PROCESSING VOLUME {volume_id}")
        print(f"   📄 Chunks: {len(chunks)}")
        print(f"   📦 Batches: {(len(chunks) + self.config.batch_size - 1) // self.config.batch_size}")
        
        volume_results = []
        
        # Process in batches
        for batch_start in range(0, len(chunks), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            batch_id = (batch_start // self.config.batch_size) + 1
            total_batches = (len(chunks) + self.config.batch_size - 1) // self.config.batch_size
            
            # Process batch
            batch_result = await self.reasoner.process_batch_with_unified_logic(
                batch_chunks, batch_id, total_batches, volume_id
            )
            
            volume_results.append(batch_result)
            
            # Save to database
            self._save_batch_result(batch_result)
            
            print(f"✅ Batch {batch_id} completed: {batch_result['final_satisfaction']:.3f} satisfaction")
        
        return volume_results
    
    def _save_batch_result(self, batch_result: Dict[str, Any]):
        """Save batch result to unified database"""
        print(f"\n🗄️ SQL DEBUG: SAVING BATCH RESULT")
        print(f"   Volume {batch_result['volume_id']}, Batch {batch_result['batch_id']}")
        print(f"   Chunks: {batch_result['chunks_processed']}")
        print(f"   Iterations: {batch_result['total_iterations']}")
        print(f"   Satisfaction: {batch_result['final_satisfaction']:.3f}")
        print(f"   Stage: {batch_result['processing_stage']}")
        print(f"   Characters: {batch_result['characters_detected']}")
        print(f"   Cost: ${batch_result['total_cost']:.4f}")
        print(f"   Early terminated: {batch_result['early_terminated']}")
        
        if self.config.use_postgres and POSTGRES_AVAILABLE:
            print(f"   Database: PostgreSQL")
            self._save_batch_result_postgres(batch_result)
        else:
            print(f"   Database: SQLite")
            self._save_batch_result_sqlite(batch_result)
    
    def _save_batch_result_postgres(self, batch_result: Dict[str, Any]):
        """Save to PostgreSQL database with schema validation"""
        try:
            print(f"   🐘 PostgreSQL: Connecting to {self.config.postgres_host}:{self.config.postgres_port}")
            conn = psycopg2.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_db,
                user=self.config.postgres_user,
                password=self.config.postgres_password
            )
            cursor = conn.cursor()
            
            # Verify table exists before operations
            try:
                cursor.execute("SELECT 1 FROM unified_results LIMIT 1")
                print(f"   📋 Table 'unified_results' verified")
            except psycopg2.errors.UndefinedTable:
                print(f"   ❌ Table 'unified_results' missing! Cannot save data.")
                print(f"   🔄 Database initialization may have failed")
                conn.close()
                return
            except Exception as table_error:
                print(f"   ⚠️ Table verification error: {table_error}")
                conn.rollback()
            
            # Check existing records for this volume
            cursor.execute("SELECT COUNT(*) FROM unified_results WHERE volume_id = %s", (batch_result['volume_id'],))
            existing_count = cursor.fetchone()[0]
            print(f"   📊 Existing records for Volume {batch_result['volume_id']}: {existing_count}")
            
            # Use parameterized query with explicit field mapping to prevent SQL injection
            insert_query = '''
            INSERT INTO unified_results 
            (volume_id, batch_id, chunks_processed, total_iterations, final_satisfaction,
             processing_stage, processing_mode, meaningful_improvements, total_tokens, 
             total_cost, early_terminated, characters_detected)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            '''
            
            cursor.execute(insert_query, (
                batch_result['volume_id'],
                batch_result['batch_id'],
                batch_result['chunks_processed'],
                batch_result['total_iterations'],
                batch_result['final_satisfaction'],
                batch_result['processing_stage'],
                self.config.mode.value,
                batch_result['meaningful_improvements'],
                batch_result['total_tokens'],
                batch_result['total_cost'],
                batch_result['early_terminated'],
                json.dumps(batch_result['characters_detected'])
            ))
            
            conn.commit()
            print(f"   ✅ PostgreSQL: Record inserted successfully")
            conn.close()
            
        except psycopg2.errors.UndefinedTable as e:
            print(f"   ❌ PostgreSQL Schema Error: {e}")
            print(f"   🔄 Required table does not exist - check database initialization")
            logger.error(f"Schema error - table missing: {e}")
        except psycopg2.errors.DatabaseError as e:
            print(f"   ❌ PostgreSQL Database Error: {e}")
            logger.error(f"Database error during save: {e}")
        except Exception as e:
            print(f"   ❌ PostgreSQL Error: {e}")
            logger.error(f"Failed to save batch result to PostgreSQL: {e}")
    
    def _save_batch_result_sqlite(self, batch_result: Dict[str, Any]):
        """Save to SQLite database"""
        try:
            print(f"   💾 SQLite: Connecting to {self.config.database_path}")
            conn = sqlite3.connect(self.config.database_path)
            cursor = conn.cursor()
            
            # Check existing records for this volume
            cursor.execute("SELECT COUNT(*) FROM unified_results WHERE volume_id = ?", (batch_result['volume_id'],))
            existing_count = cursor.fetchone()[0]
            print(f"   📊 Existing records for Volume {batch_result['volume_id']}: {existing_count}")
            
            cursor.execute('''
            INSERT INTO unified_results 
            (volume_id, batch_id, chunks_processed, total_iterations, final_satisfaction,
             processing_stage, processing_mode, meaningful_improvements, total_tokens, 
             total_cost, early_terminated, characters_detected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                batch_result['volume_id'],
                batch_result['batch_id'],
                batch_result['chunks_processed'],
                batch_result['total_iterations'],
                batch_result['final_satisfaction'],
                batch_result['processing_stage'],
                self.config.mode.value,
                batch_result['meaningful_improvements'],
                batch_result['total_tokens'],
                batch_result['total_cost'],
                batch_result['early_terminated'],
                json.dumps(batch_result['characters_detected'])
            ))
            conn.commit()
            print(f"   ✅ SQLite: Record inserted successfully")
            conn.close()
        except Exception as e:
            print(f"   ❌ SQLite Error: {e}")
            logger.error(f"Failed to save batch result to SQLite: {e}")
    
    def _generate_summary(self, all_results: List[Dict], volume_ids: List[int], processing_time: float) -> Dict[str, Any]:
        """Generate comprehensive processing summary"""
        if not all_results:
            return {'success': False, 'error': 'No results to summarize'}
        
        total_batches = len(all_results)
        total_chunks = sum(r['chunks_processed'] for r in all_results)
        total_iterations = sum(r['total_iterations'] for r in all_results)
        total_tokens = sum(r['total_tokens'] for r in all_results)
        total_cost = sum(r['total_cost'] for r in all_results)
        avg_satisfaction = sum(r['final_satisfaction'] for r in all_results) / total_batches
        total_improvements = sum(r['meaningful_improvements'] for r in all_results)
        total_terminations = sum(1 for r in all_results if r['early_terminated'])
        
        # Volume breakdown
        volume_breakdown = {}
        for vol_id in volume_ids:
            vol_results = [r for r in all_results if r['volume_id'] == vol_id]
            if vol_results:
                volume_breakdown[vol_id] = {
                    'batches': len(vol_results),
                    'chunks': sum(r['chunks_processed'] for r in vol_results),
                    'iterations': sum(r['total_iterations'] for r in vol_results),
                    'tokens': sum(r['total_tokens'] for r in vol_results),
                    'cost': sum(r['total_cost'] for r in vol_results),
                    'avg_satisfaction': sum(r['final_satisfaction'] for r in vol_results) / len(vol_results),
                    'improvements': sum(r['meaningful_improvements'] for r in vol_results),
                    'early_terminations': sum(1 for r in vol_results if r['early_terminated'])
                }
        
        return {
            'success': True,
            'volumes_processed': volume_ids,
            'processing_time': processing_time,
            'total_batches': total_batches,
            'total_chunks': total_chunks,
            'total_iterations': total_iterations,
            'total_tokens': total_tokens,
            'total_cost': total_cost,
            'avg_satisfaction': avg_satisfaction,
            'meaningful_improvements': total_improvements,
            'early_terminations': total_terminations,
            'volume_breakdown': volume_breakdown,
            'batch_results': all_results,
            'config': {
                'mode': self.config.mode.value,
                'max_iterations': self.config.max_iterations,
                'satisfaction_threshold': self.config.satisfaction_threshold
            }
        }
    
    def _display_results(self, summary: Dict[str, Any]):
        """Display comprehensive results"""
        if not summary['success']:
            print(f"❌ Processing failed: {summary.get('error')}")
            return
        
        print(f"\n" + "=" * 80)
        print("🎉 UNIFIED PROCESSING COMPLETED")
        print("=" * 80)
        
        print("📊 OVERALL RESULTS:")
        print(f"   📖 Volumes: {summary['volumes_processed']}")
        print(f"   📦 Batches: {summary['total_batches']}")
        print(f"   📄 Chunks: {summary['total_chunks']}")
        print(f"   🔄 Iterations: {summary['total_iterations']}")
        print(f"   📈 Avg satisfaction: {summary['avg_satisfaction']:.3f}")
        print(f"   ⏱️ Time: {summary['processing_time']:.1f}s ({summary['processing_time']/60:.1f} min)")
        
        print(f"\n💰 COST ANALYSIS:")
        print(f"   🔢 Tokens: {summary['total_tokens']:,}")
        print(f"   💵 Cost: ${summary['total_cost']:.4f}")
        print(f"   📊 Tokens/chunk: {summary['total_tokens']/summary['total_chunks']:.0f}")
        print(f"   📈 Improvements: {summary['meaningful_improvements']}")
        print(f"   ⏹️ Early terminations: {summary['early_terminations']}")
        
        if summary['volume_breakdown']:
            print(f"\n📖 VOLUME BREAKDOWN:")
            print("   " + "-" * 70)
            print("   VOL | BATCHES | CHUNKS | ITERATIONS | SATISFACTION | TOKENS   | COST")
            print("   " + "-" * 70)
            
            for vol_id, stats in summary['volume_breakdown'].items():
                print(f"   {vol_id:3} | {stats['batches']:7} | {stats['chunks']:6} | {stats['iterations']:10} | {stats['avg_satisfaction']:12.3f} | {stats['tokens']:8,} | ${stats['cost']:.3f}")
            
            print("   " + "-" * 70)
    
    def _save_processing_report(self, summary: Dict[str, Any], volume_ids: List[int]):
        """Save comprehensive processing report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"unified_processing_report_{timestamp}.json"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
            print(f"📋 Processing report saved: {report_path}")
        except Exception as e:
            print(f"⚠️ Failed to save report: {e}")

# Convenience functions for different processing modes
async def process_iterative(volume_ids: List[int], **kwargs) -> Dict[str, Any]:
    """Process volumes with iterative mode"""
    config = ProcessingConfig(mode=ProcessingMode.ITERATIVE, **kwargs)
    processor = UnifiedNovelProcessor(config)
    return await processor.process_volumes(volume_ids)

async def process_comprehensive(volume_ids: List[int], **kwargs) -> Dict[str, Any]:
    """Process volumes with comprehensive mode"""
    config = ProcessingConfig(mode=ProcessingMode.COMPREHENSIVE, **kwargs)
    processor = UnifiedNovelProcessor(config)
    return await processor.process_volumes(volume_ids)

async def process_batch(volume_ids: List[int], **kwargs) -> Dict[str, Any]:
    """Process volumes with batch mode"""
    config = ProcessingConfig(mode=ProcessingMode.BATCH, **kwargs)
    processor = UnifiedNovelProcessor(config)
    return await processor.process_volumes(volume_ids)

# Main execution
async def main():
    """Interactive main execution - asks user for volume choices and processes"""
    
    print("🚀 UNIFIED NOVEL PROCESSOR")
    print("=" * 80)
    print("📖 Automatically loads chunks from Qdrant and processes any volumes")
    print("✅ Features: JSON parsing fixes, token optimization, early termination")
    print("=" * 80)
    
    try:
        # Simple interactive input
        print("\n📋 Volume Selection:")
        print("Enter volume numbers to process (e.g., '2 3' for volumes 2 and 3)")
        print("Or just press Enter to process volumes 2 and 3 by default")
        
        user_input = input("Volumes to process: ").strip()
        
        if user_input:
            try:
                volume_ids = [int(x) for x in user_input.split()]
            except ValueError:
                print("❌ Invalid input. Using default volumes 2 and 3.")
                volume_ids = [2, 3]
        else:
            volume_ids = [2, 3]  # Default to continue from Volume 1
        
        print(f"\n🎯 Selected volumes: {volume_ids}")
        
        # Processing mode selection
        print("\n🔧 Processing Mode:")
        print("1. Iterative (default) - Fast with early termination")
        print("2. Comprehensive - Deep analysis")  
        print("3. Batch - Large scale processing")
        
        mode_input = input("Select mode (1-3, or Enter for default): ").strip()
        
        if mode_input == "2":
            mode = ProcessingMode.COMPREHENSIVE
            max_iterations = 5
        elif mode_input == "3":
            mode = ProcessingMode.BATCH
            max_iterations = 2
        else:
            mode = ProcessingMode.ITERATIVE
            max_iterations = 3
        
        print(f"🎯 Processing mode: {mode.value}")
        
        # Create configuration and process
        config = ProcessingConfig(
            mode=mode,
            max_iterations=max_iterations,
            satisfaction_threshold=0.80,
            batch_size=5,
            use_qdrant=True
        )
        
        print(f"\n🚀 Starting processing...")
        print("=" * 80)
        
        processor = UnifiedNovelProcessor(config)
        result = await processor.process_volumes(volume_ids)
        
        if result and result.get('success'):
            print(f"\n🎉 SUCCESS! Processed {len(result['volumes_processed'])} volumes")
            print(f"📊 Summary:")
            print(f"   📦 Total batches: {result['total_batches']}")
            print(f"   📄 Total chunks: {result['total_chunks']}")
            print(f"   🔄 Total iterations: {result['total_iterations']}")
            print(f"   📈 Average satisfaction: {result['avg_satisfaction']:.3f}")
            print(f"   💰 Total cost: ${result['total_cost']:.4f}")
            print(f"   ⏱️ Processing time: {result['processing_time']:.1f}s ({result['processing_time']/60:.1f} min)")
            
            print(f"\n💾 Results saved to: {config.database_path}")
            print(f"📋 Report saved to: unified_processing_report_*.json")
            
        else:
            print("❌ Processing failed")
            if result:
                print(f"Error: {result.get('error')}")
            return False
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️ Processing interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# DOCKER POSTGRES DATABASE INTEGRATION FOR PHASE 2
# ============================================================================

class DockerPostgresAdapter:
    """Database adapter for Docker Postgres integration with Phase 2 causality system"""
    
    def __init__(self):
        self.connection_config = {
            'host': 'localhost',
            'port': 5433,
            'database': 'novel_processing',
            'user': 'novel_user',
            'password': 'novel_pass'
        }
    
    def get_connection(self):
        """Get Docker Postgres connection"""
        try:
            return psycopg2.connect(**self.connection_config)
        except Exception as e:
            logging.error(f"Failed to connect to Docker Postgres: {e}")
            return None
    
    def save_timeline_event(self, event: TimelineEvent) -> bool:
        """Save timeline event to Docker Postgres"""
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            # Convert lists to JSON strings for storage
            primary_actors_json = json.dumps(event.primary_actors) if event.primary_actors else '[]'
            affected_characters_json = json.dumps(event.affected_characters) if event.affected_characters else '[]'
            caused_by_events_json = json.dumps(event.caused_by_events) if event.caused_by_events else '[]'
            
            cursor.execute('''
                INSERT INTO timeline_events (
                    event_id, volume_id, batch_id, chronological_order,
                    description, event_type, importance_score,
                    primary_actors, affected_characters, caused_by_events,
                    confidence_level, validation_status, created_by
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (event_id) 
                DO UPDATE SET
                    description = EXCLUDED.description,
                    importance_score = EXCLUDED.importance_score,
                    last_updated = CURRENT_TIMESTAMP
            ''', (
                event.event_id,
                event.volume_id,
                event.batch_id,
                event.chronological_order or 1,
                event.description,
                event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type),
                event.importance_score,
                primary_actors_json,
                affected_characters_json,
                caused_by_events_json,
                event.confidence_level,
                event.validation_status.value if hasattr(event.validation_status, 'value') else str(event.validation_status),
                event.created_by
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            logging.error(f"Error saving timeline event to Docker Postgres: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def save_causality_relationship(self, cause_event_id: str, effect_event_id: str, 
                                  causality_type: str, reasoning: str, influence_strength: float) -> bool:
        """Save causality relationship to Docker Postgres"""
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO event_causality (
                    cause_event_id, effect_event_id, causality_type,
                    reasoning, influence_strength, confidence
                ) VALUES (%s, %s, %s, %s, %s, %s)
            ''', (
                cause_event_id,
                effect_event_id,
                causality_type,
                reasoning,
                influence_strength,
                0.8  # Default confidence
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            logging.error(f"Error saving causality relationship to Docker Postgres: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def load_timeline_events(self, volume_id: int = None, limit: Optional[int] = None) -> List[TimelineEvent]:
        """Load timeline events from Docker Postgres"""
        conn = self.get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            
            if volume_id:
                query = '''
                    SELECT event_id, volume_id, batch_id, chronological_order,
                           description, event_type, importance_score,
                           primary_actors, affected_characters, caused_by_events,
                           confidence_level, validation_status, created_by, created_at
                    FROM timeline_events
                    WHERE volume_id = %s
                    ORDER BY chronological_order
                '''
                params = (volume_id,)
            else:
                query = '''
                    SELECT event_id, volume_id, batch_id, chronological_order,
                           description, event_type, importance_score,
                           primary_actors, affected_characters, caused_by_events,
                           confidence_level, validation_status, created_by, created_at
                    FROM timeline_events
                    ORDER BY chronological_order
                '''
                params = ()
            
            if limit:
                query += f' LIMIT {limit}'
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            events = []
            for row in rows:
                try:
                    # Parse JSON fields safely
                    primary_actors = json.loads(row[7]) if row[7] else []
                    affected_characters = json.loads(row[8]) if row[8] else []
                    caused_by_events = json.loads(row[9]) if row[9] else []
                except:
                    # Fallback if JSON parsing fails
                    primary_actors = []
                    affected_characters = []
                    caused_by_events = []
                
                event = TimelineEvent(
                    event_id=row[0],
                    volume_id=row[1],
                    batch_id=row[2],
                    chronological_order=row[3],
                    description=row[4] or '',
                    event_type=EventType.ACTION,  # Default safe value
                    importance_score=row[6] or 0.0,
                    primary_actors=primary_actors,
                    affected_characters=affected_characters,
                    caused_by_events=caused_by_events,
                    confidence_level=row[10] or 0.5,
                    created_by=row[12] or 'system',
                    created_at=row[13]
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            logging.error(f"Error loading events from Docker Postgres: {e}")
            return []
        finally:
            conn.close()
    
    def test_connection(self) -> bool:
        """Test Docker Postgres connection"""
        conn = self.get_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute('SELECT 1')
                result = cursor.fetchone()
                conn.close()
                return result[0] == 1
            except:
                if conn:
                    conn.close()
                return False
        return False

# ============================================================================
# DOCKER-INTEGRATED OPTIMAL CAUSALITY SYSTEM
# ============================================================================

class DockerIntegratedCausalitySystem(OptimalCausalitySystem):
    """Phase 2 causality system integrated with Docker Postgres database"""
    
    def __init__(self):
        super().__init__()
        self.docker_db = DockerPostgresAdapter()
        
        # Test Docker connection
        if self.docker_db.test_connection():
            logging.info("Successfully connected to Docker Postgres database")
            # Load existing events from Docker
            self._load_events_from_docker()
        else:
            logging.warning("Failed to connect to Docker Postgres - using in-memory only")
    
    def _load_events_from_docker(self):
        """Load existing events from Docker Postgres into the causality system"""
        try:
            events = self.docker_db.load_timeline_events(limit=100)  # Load recent events
            logging.info(f"Loading {len(events)} events from Docker Postgres")
            
            for event in events:
                self.add_timeline_event(event)
            
            logging.info(f"Successfully loaded {len(events)} events into causality system")
            
        except Exception as e:
            logging.error(f"Error loading events from Docker: {e}")
    
    def add_timeline_event(self, event: TimelineEvent):
        """Add timeline event to both in-memory system and Docker Postgres"""
        # Add to in-memory causality system
        super().add_timeline_event(event)
        
        # Persist to Docker Postgres
        try:
            success = self.docker_db.save_timeline_event(event)
            if success:
                logging.debug(f"Saved event {event.event_id} to Docker Postgres")
            else:
                logging.warning(f"Failed to save event {event.event_id} to Docker Postgres")
            
            # Save causality relationships
            for cause_event_id in event.caused_by_events or []:
                self.docker_db.save_causality_relationship(
                    cause_event_id, 
                    event.event_id, 
                    'direct_cause', 
                    'Event causality', 
                    0.8
                )
                
        except Exception as e:
            logging.error(f"Error saving event to Docker: {e}")


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Processing interrupted by user")
        exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        exit(1)