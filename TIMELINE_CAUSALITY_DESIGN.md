# 🗓️ Timeline & Causality System Design Document

## 📋 Project Overview

**Goal**: Design and implement a self-aware agentic timeline system with combinatorial causality structures for efficient narrative analysis without massive context consumption.

**Current Status**: Design phase completed, ready for implementation.

---

## 🎯 Core Requirements Identified

### 1. **Current System Limitations**
- ✅ Character registry with function calls (`register_character`, `update_character`)
- ❌ **Missing**: Timeline management function calls
- ❌ **Missing**: Causality reasoning system
- ❌ **Missing**: Character-timeline correlation
- ❌ **Missing**: Long-term narrative continuity tracking

### 2. **Design Goals**
1. **Chronological Coherence**: Maintain consistent temporal ordering across chunks
2. **Causal Reasoning**: Track cause-effect relationships and character motivations
3. **Character Integration**: Deep correlation with character development and actions
4. **Self-Aware Iteration**: Enable agent to query, validate, and refine timeline understanding
5. **Long-term Continuity**: Bridge narrative gaps between distant chunks
6. **Context Efficiency**: Logarithmic query time instead of linear context loading

---

## 📊 Comprehensive Data Structure Design

### **1. Timeline Event Node**
```python
@dataclass
class TimelineEvent:
    # Core Identity
    event_id: str                    # Unique identifier
    volume_id: int
    batch_id: int
    chunk_ids: List[int]            # Source chunks
    
    # Temporal Positioning
    chronological_order: int        # Global sequence number
    relative_time: str              # "before_X", "during_Y", "after_Z"
    time_confidence: float          # 0.0-1.0 confidence in temporal placement
    temporal_markers: List[str]     # ["morning", "three_days_later", "simultaneously"]
    
    # Event Content
    description: str
    event_type: EventType           # DIALOGUE, ACTION, REVELATION, CONFLICT, etc.
    importance_score: float         # 0.0-1.0 narrative significance
    
    # Character Integration
    primary_actors: List[str]       # Main characters involved
    affected_characters: List[str]  # Characters indirectly affected
    character_states_before: Dict[str, CharacterState]
    character_states_after: Dict[str, CharacterState]
    
    # Causal Relationships
    caused_by_events: List[str]     # Event IDs that led to this
    causes_events: List[str]        # Event IDs this leads to
    motivation_triggers: List[MotivationTrigger]
    
    # Narrative Structure
    plot_thread: str                # "main_plot", "character_arc", "subplot_A"
    narrative_function: NarrativeFunction  # SETUP, CONFLICT, CLIMAX, RESOLUTION
    foreshadowing_links: List[str]  # Events this foreshadows/fulfills
    
    # Metadata
    confidence_level: float
    validation_status: ValidationStatus
    last_updated: datetime
    created_by: str                # "ai_analysis", "function_call", "auto_inference"
```

### **2. Supporting Data Structures**
```python
@dataclass 
class MotivationTrigger:
    character_name: str
    motivation_type: str            # "revenge", "protection", "curiosity", "duty"
    trigger_description: str
    motivation_strength: float      # 0.0-1.0
    duration: str                  # "immediate", "short_term", "ongoing"

@dataclass
class CharacterState:
    emotional_state: str           # "angry", "confused", "determined"
    knowledge_level: Dict[str, float]  # What they know about various topics
    relationships: Dict[str, float]    # Relationship scores with other characters  
    goals: List[str]              # Current objectives
    capabilities: List[str]        # Current abilities/resources

class EventType(Enum):
    DIALOGUE = "dialogue"
    ACTION = "action"
    REVELATION = "revelation"
    CONFLICT = "conflict"
    RESOLUTION = "resolution"
    INTERNAL_MONOLOGUE = "internal_monologue"
    ENVIRONMENTAL = "environmental"
    
class NarrativeFunction(Enum):
    SETUP = "setup"
    INCITING_INCIDENT = "inciting_incident"  
    RISING_ACTION = "rising_action"
    CLIMAX = "climax"
    FALLING_ACTION = "falling_action"
    RESOLUTION = "resolution"
    DENOUEMENT = "denouement"
```

### **3. Storyline Graph Structure**
```python
@dataclass
class StorylineGraph:
    # Event Storage
    events: Dict[str, TimelineEvent]           # event_id -> event
    chronological_sequence: List[str]          # Ordered event_ids
    
    # Relationship Networks
    causal_network: nx.DiGraph                 # Cause-effect graph
    character_timeline: Dict[str, List[str]]   # character -> ordered event_ids
    plot_threads: Dict[str, List[str]]         # thread_name -> ordered event_ids
    
    # Temporal Anchors
    temporal_markers: Dict[str, int]           # "day_1" -> chronological_order
    chapter_boundaries: List[int]              # Chronological positions
    volume_boundaries: List[int]
    
    # Validation & Coherence
    consistency_score: float                   # Overall timeline coherence
    unresolved_conflicts: List[TemporalConflict]
    confidence_distribution: Dict[str, int]    # confidence_level -> count
```

---

## 🤖 Agentic Timeline Function Calls Design

### **1. Core Timeline Functions**
```python
# Timeline Event Management
{
  "action": "create_timeline_event",
  "params": {
    "description": "事件描述",
    "event_type": "dialogue",
    "primary_actors": ["角色A", "角色B"],
    "importance_score": 0.8,
    "temporal_position": "after_previous_event",
    "causal_triggers": ["event_id_123"]
  }
}

{
  "action": "update_timeline_event", 
  "params": {
    "event_id": "evt_456",
    "new_importance": 0.9,
    "add_causal_link": "evt_789",
    "update_character_states": {
      "角色A": {"emotional_state": "determined", "new_goal": "寻找真相"}
    }
  }
}

{
  "action": "reorder_timeline_events",
  "params": {
    "event_ids": ["evt_123", "evt_456", "evt_789"],
    "new_chronological_order": [2, 1, 3],
    "reasoning": "重新分析后发现时间顺序有误"
  }
}
```

### **2. Causal Reasoning Functions**
```python
{
  "action": "establish_causality",
  "params": {
    "cause_event_id": "evt_123",
    "effect_event_id": "evt_456", 
    "causality_type": "direct_cause",
    "confidence": 0.85,
    "reasoning": "角色A的决定直接导致了事件B"
  }
}

{
  "action": "trace_motivation_chain",
  "params": {
    "character_name": "角色A",
    "starting_event": "evt_100",
    "ending_event": "evt_200",
    "motivation_hypothesis": "复仇动机驱动整个行为链"
  }
}

{
  "action": "identify_plot_catalyst",
  "params": {
    "event_description": "关键转折点",
    "affected_plot_threads": ["主线", "感情线"],
    "catalyst_type": "revelation"
  }
}
```

### **3. Character-Timeline Integration Functions**
```python
{
  "action": "update_character_timeline",
  "params": {
    "character_name": "角色A",
    "event_id": "evt_456",
    "character_role": "protagonist",
    "state_changes": {
      "knowledge_gained": "敌人的真实身份",
      "emotional_shift": "愤怒 -> 冷静决心",
      "relationship_change": {"角色B": "0.8 -> 0.3"}
    }
  }
}

{
  "action": "correlate_character_development",
  "params": {
    "character_name": "角色A", 
    "development_arc": "从无能少年到英雄",
    "key_events": ["evt_123", "evt_456", "evt_789"],
    "development_metrics": {
      "power_level": [0.2, 0.5, 0.8],
      "confidence": [0.3, 0.6, 0.9]
    }
  }
}
```

### **4. Self-Aware Query & Validation Functions**
```python
{
  "action": "query_timeline_context",
  "params": {
    "query_type": "character_motivation",
    "character_name": "角色A",
    "time_window": "last_5_events",
    "context_request": "为什么做出这个决定？"
  }
}

{
  "action": "validate_temporal_consistency",
  "params": {
    "validation_scope": "volume_3",
    "check_types": ["chronological_order", "character_state_continuity", "causal_chains"],
    "conflict_resolution": "flag_for_review"
  }
}

{
  "action": "cross_reference_events",
  "params": {
    "primary_event": "evt_456",
    "reference_scope": "all_volumes",
    "correlation_types": ["foreshadowing", "callback", "parallel_structure"]
  }
}
```

---

## 🌲 Combinatorial Causality Structure (KEY INNOVATION)

### **Problem Statement**
- **Current Issue**: Querying causality requires loading entire story context (O(n))
- **Goal**: Direct computational causality path finding (O(log n))
- **Solution**: Hybrid Forest + Graph Architecture with multi-level indexing

### **Recommended Structure: Hybrid Forest + Graph**

#### **Primary Structure: Causal Forest (Multiple Rooted Trees)**
```python
class CausalForest:
    """Multiple causality trees rooted at different catalyst events"""
    
    def __init__(self):
        self.causal_trees: Dict[str, CausalTree] = {}  # root_event_id -> tree
        self.event_to_tree_map: Dict[str, str] = {}    # event_id -> root_event_id
        self.cross_tree_bridges: List[CrossTreeEdge] = []  # Inter-tree connections
```

**Why Forest > Single Tree:**
1. **Multiple Causality Origins**: Stories have multiple independent catalyst events
2. **Parallel Storylines**: Different plot threads with minimal interaction
3. **Computational Efficiency**: Query only relevant subtree, not entire story

#### **Multi-Level Indexing System**
```python
@dataclass
class CausalIndex:
    # Primary Index: Event -> Direct Causality Path
    event_to_path: Dict[str, CausalPath] = {}
    
    # Secondary Index: Fast Lookups
    character_event_index: Dict[str, Set[str]] = {}    # character -> event_ids
    motivation_index: Dict[str, Set[str]] = {}         # motivation_type -> event_ids
    temporal_index: SortedDict[int, Set[str]] = {}     # chronological_order -> event_ids
    
    # Tertiary Index: Combinatorial Queries
    causal_depth_index: Dict[int, Set[str]] = {}       # depth_level -> event_ids
    cascade_strength_index: Dict[float, Set[str]] = {} # influence_strength -> event_ids
```

#### **Causal Path Compression**
```python
@dataclass
class CausalPath:
    """Compressed representation of causality chain"""
    target_event_id: str
    root_catalyst: str                    # Ultimate origin
    path_length: int                      # Degrees of separation
    compressed_chain: List[CausalLink]    # Only critical nodes
    influence_strength: float             # Cumulative causality strength
```

### **Optimization Structures**

#### **Structure 1: Stratified Causal DAG**
```python
class StratifiedCausalDAG:
    """Multi-layer causality with fast path finding"""
    
    def __init__(self):
        # Layer 0: Root catalysts (story origins)
        self.catalyst_layer: Dict[str, CatalystEvent] = {}
        
        # Layer N: Causality levels (depth from root)
        self.causality_layers: Dict[int, Dict[str, TimelineEvent]] = {}
        
        # Cross-layer edges: Direct causality links
        self.causal_edges: Dict[str, List[CausalEdge]] = {}
        
        # Fast lookup: Event -> Layer mapping
        self.event_layer_map: Dict[str, int] = {}
```

#### **Structure 2: Causal Trie for Motivation Chains**
```python
class MotivationTrie:
    """Efficient storage/retrieval of motivation-driven event chains"""
    
    class TrieNode:
        def __init__(self):
            self.motivation_type: Optional[str] = None
            self.character_name: Optional[str] = None  
            self.children: Dict[str, TrieNode] = {}    # next_motivation -> node
            self.terminal_events: List[str] = []       # Events at this path end
```

#### **Structure 3: Optimal Multi-Structure System**
```python
class OptimalCausalitySystem:
    """Combines best aspects of multiple structures"""
    
    def __init__(self):
        # Primary: Stratified DAG for general causality
        self.causal_dag = StratifiedCausalDAG()
        
        # Secondary: Character subgraphs for character-specific queries  
        self.character_graphs: Dict[str, nx.DiGraph] = {}
        
        # Tertiary: Motivation trie for pattern matching
        self.motivation_trie = MotivationTrie()
        
        # Quaternary: Temporal index for chronological queries
        self.temporal_index = SortedDict()  # time -> event_set
    
    def smart_query(self, query_type: str, params: Dict) -> CausalityResult:
        """Route to optimal structure based on query type"""
```

### **Context Optimization Benefits**

#### **Before: Linear Context Loading**
```
Query: "Why is Character A angry in current chunk?"
Process: Load all story summaries → Search sequentially → Find relevant events
Context Size: O(n) where n = total story length
Time Complexity: O(n)
```

#### **After: Combinatorial Direct Access**  
```
Query: "Why is Character A angry in current chunk?"
Process: 
1. Index lookup: current_event_id → character_event_index["Character A"]
2. Path compression: Find causality_path(current_event, max_depth=3)
3. Return compressed reasoning chain

Context Size: O(log n) - only relevant causal path
Time Complexity: O(log n) with preprocessing
```

---

## 🗄️ Database Schema Design

### **Extended Database Schema**
```sql
-- Timeline Events
CREATE TABLE timeline_events (
    event_id VARCHAR PRIMARY KEY,
    volume_id INTEGER,
    batch_id INTEGER,
    chronological_order INTEGER,
    event_type VARCHAR,
    description TEXT,
    importance_score REAL,
    confidence_level REAL,
    created_at TIMESTAMP
);

-- Causal Relationships  
CREATE TABLE event_causality (
    cause_event_id VARCHAR,
    effect_event_id VARCHAR,
    causality_type VARCHAR,
    confidence REAL,
    reasoning TEXT,
    PRIMARY KEY (cause_event_id, effect_event_id)
);

-- Character-Event Relationships
CREATE TABLE character_event_participation (
    character_name VARCHAR,
    event_id VARCHAR, 
    participation_type VARCHAR, -- 'primary_actor', 'affected', 'witness'
    character_state_before JSONB,
    character_state_after JSONB,
    PRIMARY KEY (character_name, event_id)
);

-- Motivation Chains
CREATE TABLE motivation_chains (
    chain_id VARCHAR PRIMARY KEY,
    character_name VARCHAR,
    motivation_type VARCHAR,
    trigger_event_id VARCHAR,
    culmination_event_id VARCHAR,
    chain_strength REAL
);
```

---

## 🚀 Implementation Roadmap

### **Phase 1: Core Infrastructure (Week 1-2)**
- ✅ Database schema extension
- ✅ Core data classes implementation (`TimelineEvent`, `StorylineGraph`, etc.)
- ✅ Database access layer with dual PostgreSQL/SQLite support

### **Phase 2: Basic Timeline Functions (Week 3-4)**
- ✅ Essential function calls: `create_timeline_event`, `update_timeline_event`, `establish_causality`
- ✅ Integration with existing `_process_function_calls()`
- ✅ Extended JSON schema in prompt engineering

### **Phase 3: Correlation Engine (Week 5-6)**
- ✅ `CharacterTimelineCorrelator` implementation
- ✅ Advanced query system with multi-dimensional correlation
- ✅ Cross-volume narrative consistency checking

### **Phase 4: Combinatorial Causality System (Week 7-8)**
- ✅ **PRIORITY**: Stratified Causal DAG implementation
- ✅ Multi-level indexing system
- ✅ Path compression and optimization algorithms
- ✅ Smart query routing system

### **Phase 5: Self-Aware Iteration (Week 9-10)**
- ✅ Validation & refinement loops
- ✅ Pattern recognition and learning systems
- ✅ Predictive analytics for character behavior and plot progression

---

## 🎯 Key Performance Targets

### **Context Efficiency**
- **Target**: Reduce context size from O(n) to O(log n)
- **Method**: Combinatorial direct access with path compression

### **Query Performance**
- **Target**: Sub-second causality path queries for stories up to 1M+ events
- **Method**: Multi-level indexing with optimal data structure selection

### **Accuracy Metrics**
- **Character Consistency**: >90% across long story arcs
- **Temporal Coherence**: >95% chronological accuracy
- **Causal Reasoning**: >85% correct cause-effect identification

---

## 🔧 Integration Points

### **Current System Extension Points**
1. **`_process_function_calls()`** - Add timeline function routing
2. **Prompt Engineering** - Extended JSON schema with timeline functions
3. **`_validate_environment()`** - Add timeline database validation
4. **Character Registry** - Bidirectional timeline correlation

### **New System Components**
1. **`TimelineManager`** - Core timeline management class
2. **`CausalityEngine`** - Combinatorial causality computation
3. **`CharacterTimelineCorrelator`** - Character-timeline integration
4. **`TimelineValidator`** - Consistency checking and conflict resolution

---

## 🎯 Success Metrics

### **Quantitative Targets**
- **Context Reduction**: 90%+ reduction in context size for causality queries
- **Query Speed**: <1s response time for complex causality chains
- **Accuracy**: >90% character consistency, >95% temporal accuracy
- **Scalability**: Handle 1M+ events without performance degradation

### **Qualitative Targets**
- **Narrative Coherence**: Maintain logical story continuity across volumes
- **Character Development**: Track believable character evolution arcs
- **Causal Logic**: Provide meaningful explanations for character actions
- **Self-Improvement**: System learns and refines understanding over time

---

## 📝 Next Steps

1. **Review and validate** this design with stakeholders
2. **Prioritize implementation phases** based on immediate needs
3. **Set up development environment** for timeline system
4. **Begin Phase 1**: Database schema and core infrastructure
5. **Prototype combinatorial causality system** for performance validation

---

**Document Status**: ✅ Complete - Ready for implementation  
**Last Updated**: 2025-09-25  
**Next Review**: Before Phase 1 implementation begins

---

*This document represents the complete brainstorming and design phase for the Timeline & Causality System. All major design decisions have been made and the system is ready for implementation.*