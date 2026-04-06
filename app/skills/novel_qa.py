"""
Skill: novel_qa
===============
Focus mode for answering questions about Chinese novels using the knowledge base.

Data sources available
----------------------
1. Neo4j knowledge graph (structured extracted data)
   - Event nodes: plot events with actors, causes, effects
   - Character nodes: names, types, traits
   - Use run_cypher to query (see knowledge_graph skill for schema)
   - NOTE: empty until a novel is ingested via the novel_processing skill

2. Qdrant vector store (raw novel text chunks)
   - Collection: novel_chunks
   - Contains: original text passages with page/chapter metadata
   - Use semantic_search to query by meaning
   - NOTE: empty until a novel is ingested

3. If both sources are empty, tell the user honestly that no novel has been
   loaded yet and suggest they use /chat to ask about ingesting one.

Answer strategy
---------------
1. For factual questions (who, what, when): prefer Neo4j (structured facts)
2. For passage/quote questions: use semantic_search on Qdrant
3. For causal/relationship questions: use run_cypher with path queries
4. Always cite which source you retrieved from
5. If data is missing, say so clearly — do not hallucinate plot details

Response format
---------------
- Answer in the same language the user asked in (Chinese question → Chinese answer)
- Keep answers concise unless the user asks for deep analysis
- For character questions, include character_type and key traits when available
"""

TOOLS = ["run_cypher", "semantic_search"]


class NovelQASkill:
    name = "novel_qa"
    description = "Answer questions about novel plot, characters, and events using the knowledge base"
    tools = TOOLS

    @classmethod
    def get_prompt(cls) -> str:
        return __doc__
