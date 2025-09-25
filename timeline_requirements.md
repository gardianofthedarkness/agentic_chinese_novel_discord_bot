# Timeline System Requirements Analysis

## Core Concepts

### 1. Timeline Tree Structure
- **Timeline Node**: A specific moment/state in the story with contextual information
- **Branch**: A path of connected nodes representing story progression
- **Root**: Starting point of the story/scenario
- **Leaf**: Current endpoints of different story branches
- **Merge Points**: Where different branches can converge back to similar states

### 2. Parallel Universe Logic
- **Universe State**: Complete context of characters, relationships, events at a timeline node
- **Similarity Scoring**: Algorithm to measure how similar two timeline states are
- **Auto-Selection**: When a channel opens, find the closest existing timeline branch
- **Branching Threshold**: When current conversation diverges enough to create new branch

### 3. Scenario Roleplay Features
- **Chapter Context**: Current story chapter, location, active characters, recent events
- **Background Hints**: Generated summaries for new users to understand current situation  
- **Story Continuity**: Maintain consistency across timeline branches
- **Character States**: Track character relationships, knowledge, emotions across timelines

### 4. User Character System
- **User Profiles**: Users can create/play as characters in the story
- **Role Switching**: Toggle between observer mode and character participant mode
- **Character Abilities**: Define what user characters can do in the story world
- **Story Integration**: How user actions affect the timeline and other characters

## Detailed Requirements

### Timeline Data Structure
```
TimelineNode:
  - id: unique identifier
  - parent_id: reference to parent node (null for root)
  - children: list of child node ids
  - timestamp: when this node was created
  - chapter_context: story state at this point
  - conversation_summary: key events in this conversation
  - characters_present: list of active characters
  - user_characters: list of users playing characters
  - story_tags: themes, events, locations
  - universe_fingerprint: hash of key story elements
```

### Branching Logic
```
Branch Creation Triggers:
1. Significant story choice/decision made
2. New user joins with different character concept
3. Major character action that changes story direction
4. Manual branch creation by users
5. Context similarity drops below threshold

Branch Conditions:
- Minimum conversation length before branching (5+ messages)
- Maximum similarity score for auto-branching (< 0.7)
- User confirmation for major story changes
- Character consensus for story-altering decisions
```

### Background Story System
```
Story Context Elements:
- Current chapter/arc summary
- Recent major events (last 5-10 key moments)
- Character relationship status
- Current location/setting
- Active plot threads
- Character goals and motivations
- World state (political, magical, environmental)

Hint Generation:
- Extract context using RAG from similar story moments
- Generate 2-3 sentence summaries for new users
- Provide character motivation context
- Explain current relationships and tensions
- Suggest how user characters could integrate
```

### User Character Integration
```
User Character Types:
1. Original Characters: User-created, new to story
2. Adopted Characters: Taking over existing story characters
3. Guest Characters: Temporary participants
4. Observer Mode: Read-only participation

Character Creation Requirements:
- Name and basic personality
- Background that fits story world
- Starting relationships with other characters
- Special abilities/skills relevant to story
- Integration point in current timeline
```

### Parallel Universe Matching
```
Similarity Factors:
- Character relationships (30% weight)
- Current location/setting (20% weight)  
- Recent events/plot points (25% weight)
- Character emotional states (15% weight)
- Active plot threads (10% weight)

Matching Algorithm:
1. Calculate similarity scores for all existing branches
2. Find branches above similarity threshold (> 0.6)
3. Select highest scoring compatible branch
4. If no good match, create new branch from closest root
5. Initialize new branch with background context
```

## Technical Architecture

### Database Schema
```sql
-- Timeline nodes
CREATE TABLE timeline_nodes (
    id TEXT PRIMARY KEY,
    parent_id TEXT,
    channel_id TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    chapter_title TEXT,
    location TEXT,
    summary TEXT,
    universe_fingerprint TEXT,
    FOREIGN KEY (parent_id) REFERENCES timeline_nodes(id)
);

-- Character states at timeline nodes
CREATE TABLE node_character_states (
    node_id TEXT,
    character_name TEXT,
    character_type TEXT, -- 'npc', 'user', 'system'
    user_id TEXT, -- Discord user ID if user character
    emotional_state TEXT,
    location TEXT,
    knowledge_level INTEGER,
    relationships TEXT, -- JSON
    FOREIGN KEY (node_id) REFERENCES timeline_nodes(id)
);

-- Story events at timeline nodes
CREATE TABLE node_events (
    id TEXT PRIMARY KEY,
    node_id TEXT,
    event_type TEXT, -- 'dialogue', 'action', 'plot_point', 'decision'
    event_summary TEXT,
    characters_involved TEXT, -- JSON array
    impact_score REAL, -- How significant this event is
    FOREIGN KEY (node_id) REFERENCES timeline_nodes(id)
);

-- User character definitions
CREATE TABLE user_characters (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL, -- Discord user ID
    character_name TEXT NOT NULL,
    personality TEXT,
    background TEXT,
    abilities TEXT, -- JSON
    created_at DATETIME,
    active BOOLEAN DEFAULT TRUE
);
```

### Core Classes Structure
```python
class TimelineNode:
    def __init__(self, id, parent_id, channel_id, chapter_context)
    def add_child(self, child_node)
    def get_path_to_root(self)
    def calculate_universe_fingerprint(self)
    def get_story_context(self)

class TimelineManager:
    def __init__(self, db_path)
    def create_branch(self, parent_node, context)
    def find_similar_branches(self, context, threshold=0.6)
    def auto_select_timeline(self, channel_id, context)
    def get_background_context(self, node_id)

class UserCharacter:
    def __init__(self, user_id, name, personality, background)
    def integrate_into_story(self, timeline_node)
    def get_available_actions(self, context)
    def update_relationships(self, interactions)

class StoryContextExtractor:
    def __init__(self, rag_system)
    def extract_chapter_context(self, conversation_history)
    def generate_background_hints(self, timeline_node)
    def analyze_story_progression(self, node_sequence)
```

## User Experience Flow

### New Channel Initialization
1. Bot joins channel or user activates bot
2. System analyzes any existing conversation context
3. Search for similar existing timeline branches
4. If match found (>0.6 similarity), connect to existing branch
5. If no match, create new branch from story root
6. Generate background context for new users
7. Present current story state and available characters

### User Character Creation
1. User issues character creation command
2. System analyzes current story context for integration points
3. User provides character details (name, personality, background)
4. System validates character fits story world
5. Character integrated into current timeline node
6. Relationships initialized with existing characters
7. User can switch between observer and character mode

### Story Progression
1. Users interact through Discord messages
2. System tracks significant events and decisions
3. Timeline nodes created at key story moments
4. Character states updated with each interaction
5. Background context continuously updated
6. Branch creation triggered by major divergences
7. Universe fingerprint recalculated for matching

### Timeline Branching
1. System detects significant story divergence
2. Calculates similarity to existing branches
3. If similarity drops below threshold, prompt for branching
4. Create new branch with current context
5. Initialize new universe state
6. Continue story progression on new branch
7. Track relationships between parallel branches

## Success Metrics

### System Performance
- Timeline matching accuracy (>85% user satisfaction)
- Background context relevance (>90% helpful ratings)
- Story continuity maintenance across branches
- Response time for timeline operations (<2 seconds)

### User Engagement
- User character adoption rate
- Story branch diversity and creativity
- Cross-timeline interaction frequency
- Long-term story engagement retention

### Story Quality
- Character consistency across timelines
- Plot coherence within branches
- Background hint usefulness for new users
- Integration quality of user-created characters