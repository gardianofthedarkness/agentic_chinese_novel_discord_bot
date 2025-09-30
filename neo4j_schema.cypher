// Neo4j Schema for Agentic Chinese Novel Processing
// Optimized for causality analysis and character relationship networks

// Create indexes for performance
CREATE INDEX event_id_index IF NOT EXISTS FOR (e:Event) ON (e.event_id);
CREATE INDEX chapter_index IF NOT EXISTS FOR (e:Event) ON (e.chapter_index);
CREATE INDEX character_name_index IF NOT EXISTS FOR (c:Character) ON (c.name);
CREATE INDEX location_name_index IF NOT EXISTS FOR (l:Location) ON (l.name);
CREATE INDEX timeline_order_index IF NOT EXISTS FOR (e:Event) ON (e.timeline_order);

// Create fulltext indexes for content search
CREATE FULLTEXT INDEX event_content_search IF NOT EXISTS FOR (e:Event) ON EACH [e.description, e.content_summary];
CREATE FULLTEXT INDEX character_description_search IF NOT EXISTS FOR (c:Character) ON EACH [c.description, c.personality_traits];

// Vector index for semantic similarity (requires APOC/GDS)
CALL db.index.vector.createNodeIndex(
  'event_embeddings',
  'Event',
  'embedding',
  1536,  // DeepSeek embedding dimension
  'cosine'
);

CALL db.index.vector.createNodeIndex(
  'character_embeddings',
  'Character',
  'embedding',
  1536,
  'cosine'
);

// Create constraints
CREATE CONSTRAINT event_id_unique IF NOT EXISTS FOR (e:Event) ON e.event_id;
CREATE CONSTRAINT character_name_unique IF NOT EXISTS FOR (c:Character) ON c.name;
CREATE CONSTRAINT location_name_unique IF NOT EXISTS FOR (l:Location) ON l.name;

// Node Labels and Properties Schema

// Event nodes - Core timeline events
// Properties:
// - event_id: STRING (unique identifier)
// - chapter_index: INTEGER (source chapter)
// - timeline_order: INTEGER (chronological position)
// - description: STRING (event description)
// - content_summary: STRING (detailed content)
// - emotional_intensity: FLOAT (0.0-1.0)
// - importance_score: FLOAT (0.0-1.0)
// - event_type: STRING (dialogue, action, description, etc.)
// - embedding: LIST[FLOAT] (semantic vector)
// - created_at: DATETIME
// - processed_by: STRING (processing model version)

// Character nodes - Novel characters with variants unified
// Properties:
// - name: STRING (primary name)
// - aliases: LIST[STRING] (name variants)
// - character_type: STRING (protagonist, antagonist, supporting, etc.)
// - personality_traits: LIST[STRING]
// - emotional_state: STRING (current emotional state)
// - description: STRING (physical/personality description)
// - first_appearance_chapter: INTEGER
// - confidence_score: FLOAT (detection confidence)
// - embedding: LIST[FLOAT] (personality vector)
// - created_at: DATETIME

// Location nodes - Settings and places
// Properties:
// - name: STRING (location name)
// - description: STRING (location description)
// - location_type: STRING (indoor, outdoor, building, etc.)
// - first_mentioned_chapter: INTEGER
// - embedding: LIST[FLOAT] (semantic vector)

// Concept nodes - Abstract concepts, themes, powers
// Properties:
// - name: STRING (concept name)
// - concept_type: STRING (power, theme, organization, etc.)
// - description: STRING
// - importance: FLOAT
// - embedding: LIST[FLOAT]

// Volume nodes - Novel volume structure
// Properties:
// - volume_number: INTEGER
// - title: STRING
// - chapter_range: STRING (e.g., "1-10")
// - summary: STRING

// Chapter nodes - Individual chapters
// Properties:
// - chapter_index: INTEGER
// - title: STRING
// - summary: STRING
// - word_count: INTEGER
// - key_events_count: INTEGER

// Relationship Types and Properties

// CAUSES relationship - Direct causality
// Properties:
// - causality_strength: FLOAT (0.0-1.0)
// - confidence: FLOAT (0.0-1.0)
// - causal_type: STRING (direct, indirect, enabling, preventing)
// - time_delay: INTEGER (events between cause and effect)

// INFLUENCES relationship - Indirect influence
// Properties:
// - influence_type: STRING (emotional, strategic, informational)
// - strength: FLOAT (0.0-1.0)
// - duration: STRING (temporary, lasting, permanent)

// HAPPENS_BEFORE/HAPPENS_AFTER - Temporal ordering
// Properties:
// - time_gap: INTEGER (timeline order difference)
// - certainty: FLOAT (temporal order confidence)

// PARTICIPATES_IN - Character participation in events
// Properties:
// - role: STRING (protagonist, witness, victim, etc.)
// - importance: FLOAT (how central to the event)
// - emotional_impact: FLOAT (impact on character)

// MENTIONS - Character/location/concept mentioned in event
// Properties:
// - mention_type: STRING (direct, indirect, implied)
// - frequency: INTEGER (times mentioned)
// - context: STRING (positive, negative, neutral)

// KNOWS - Character relationships
// Properties:
// - relationship_type: STRING (friend, enemy, family, etc.)
// - strength: FLOAT (0.0-1.0)
// - trust_level: FLOAT (-1.0 to 1.0)
// - first_interaction_chapter: INTEGER

// LOCATED_AT - Event location relationships
// Properties:
// - location_specificity: STRING (exact, approximate, general)

// BELONGS_TO - Chapter/Event to Volume relationships
// Properties:
// - position_in_volume: INTEGER

// Sample query patterns for optimization:

// 1. Find causality chains
// MATCH path = (start:Event)-[:CAUSES*1..5]->(end:Event)
// WHERE start.event_id = $start_id AND end.event_id = $end_id
// RETURN path, length(path) as chain_length
// ORDER BY chain_length LIMIT 10

// 2. Character influence networks
// MATCH (c:Character)-[r:PARTICIPATES_IN]->(e:Event)-[:CAUSES*1..3]->(affected:Event)
// WHERE c.name = $character_name
// RETURN c, collect(DISTINCT affected) as influenced_events

// 3. Temporal event analysis
// MATCH (e1:Event)-[:HAPPENS_BEFORE]->(e2:Event)
// WHERE e1.chapter_index = $chapter AND e2.chapter_index > $chapter
// RETURN e1, e2, e2.chapter_index - e1.chapter_index as chapter_gap

// 4. Semantic similarity search (with vector index)
// CALL db.index.vector.queryNodes('event_embeddings', 10, $query_vector)
// YIELD node, score
// RETURN node.description, score

// 5. Character emotional journey
// MATCH (c:Character)-[:PARTICIPATES_IN]->(e:Event)
// WHERE c.name = $character_name
// RETURN e.chapter_index, e.description, e.emotional_intensity
// ORDER BY e.timeline_order

// Performance optimization patterns:
// - Use timeline_order for efficient chronological queries
// - Vector indexes for semantic search replace expensive LIKE operations
// - Graph algorithms (shortest path, centrality) for complex relationship analysis
// - Batch inserts with UNWIND for large data processing