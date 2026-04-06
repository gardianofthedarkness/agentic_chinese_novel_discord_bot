"""
Skill: knowledge_graph
======================
Focus mode for querying the Neo4j knowledge graph of extracted novel data.

Database state
--------------
Neo4j is populated when novels are ingested. Until then the graph is empty
and queries will return no results (not errors).

Neo4j schema (exact property names — do NOT invent others)
----------------------------------------------------------
Node labels:
  Event
    event_id            string   unique identifier
    volume_id           integer  which volume this event belongs to
    batch_id            integer  processing batch
    chronological_order integer  ordering within a volume
    description         string   plain-text description of the event
    event_type          string   e.g. "battle", "dialogue", "discovery"
    importance_score    float    0.0 – 1.0
    confidence_level    float    0.0 – 1.0
    primary_actors      list     character names directly involved
    affected_characters list     characters indirectly affected
    caused_by_events    list     event_ids that caused this event
    causes_events       list     event_ids this event caused
    temporal_markers    list     time/location strings

  Character
    character_id        string   unique identifier
    name                string   character name (Chinese)
    volume_id           integer
    character_type      string   "protagonist" | "antagonist" | "supporting"
    aliases             list     alternative names
    personality_traits  list     trait strings
    first_appearance    integer  chronological_order of first event
    confidence_score    float

Relationships: none ingested yet (empty graph until novel loaded).

Cypher query patterns
---------------------
# Find characters by name (partial match)
MATCH (c:Character) WHERE c.name CONTAINS '萧' RETURN c.name, c.character_type LIMIT 10

# List events in a volume ordered by time
MATCH (e:Event {volume_id: 1})
RETURN e.description, e.event_type, e.chronological_order
ORDER BY e.chronological_order LIMIT 20

# Find events involving a character
MATCH (e:Event)
WHERE '萧炎' IN e.primary_actors OR '萧炎' IN e.affected_characters
RETURN e.description, e.event_type ORDER BY e.chronological_order LIMIT 10

# Count nodes to check if data is loaded
MATCH (n) RETURN labels(n)[0] as label, count(n) as count

# Full-text search on event descriptions
MATCH (e:Event)
WHERE toLower(e.description) CONTAINS toLower('斗气')
RETURN e.description LIMIT 10

IMPORTANT rules:
- Always use CONTAINS for partial string match, NOT = for names
- Do NOT use UNION unless both sides return identical column names
- Do NOT reference properties that don't exist in the schema above
- If the graph is empty, tell the user no novel has been ingested yet
"""

TOOLS = ["run_cypher", "upload_neo4j"]


class KnowledgeGraphSkill:
    name = "knowledge_graph"
    description = "Query and update the Neo4j knowledge graph of extracted novel events and characters"
    tools = TOOLS

    @classmethod
    def get_prompt(cls) -> str:
        return __doc__
