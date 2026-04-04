"""
RAG Skill
=========
Retrieval-Augmented Generation over the novel knowledge base.

Retrieves relevant passages from Neo4j (text search) and optionally
Qdrant (semantic search), then returns them as context for the LLM.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RAGSkill:
    """
    Wraps Neo4j + Qdrant retrieval into a simple retrieve() interface.
    """

    def __init__(self, db, qdrant=None):
        """
        Args:
            db:      DatabaseAdapter instance (Neo4j + PostgreSQL).
            qdrant:  QdrantAdapter instance (optional; enables semantic search).
        """
        self.db = db
        self.qdrant = qdrant

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        volume_id: Optional[int] = None,
    ) -> List[str]:
        """
        Retrieve context passages relevant to `query`.

        Returns a list of plain-text passages (no metadata).
        """
        results: List[str] = []

        # 1. Neo4j text search
        neo4j_results = await self._neo4j_search(query, limit, volume_id)
        results.extend(neo4j_results)

        # 2. Qdrant semantic search (if embedding available)
        if self.qdrant and self.qdrant.is_connected():
            qdrant_results = await self._qdrant_search(query, limit)
            results.extend(qdrant_results)

        return results[:limit]

    async def retrieve_for_character(
        self, character_name: str, limit: int = 5
    ) -> List[str]:
        """Return passages that mention a specific character."""
        return await self._neo4j_search(character_name, limit)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _neo4j_search(
        self, query: str, limit: int, volume_id: Optional[int] = None
    ) -> List[str]:
        if not self.db.neo4j or not self.db.neo4j.driver:
            return []
        try:
            params: Dict[str, Any] = {"q": query, "lim": limit}
            cypher = (
                "MATCH (e:Event) "
                "WHERE toLower(e.description) CONTAINS toLower($q) "
            )
            if volume_id:
                cypher += "AND e.volume_id = $vol "
                params["vol"] = volume_id
            cypher += "RETURN e.description AS text ORDER BY e.importance_score DESC LIMIT $lim"

            with self.db.neo4j.driver.session() as session:
                res = session.run(cypher, **params)
                return [r["text"] for r in res if r["text"]]
        except Exception as e:
            logger.warning(f"Neo4j RAG search failed: {e}")
            return []

    async def _qdrant_search(self, query: str, limit: int) -> List[str]:
        """
        Semantic search via Qdrant.

        Requires an embedding function — plug in your embedding model here.
        """
        # TODO: inject an embedding function (e.g. m3e-small, OpenAI embeddings)
        # Example skeleton:
        # vector = await embed(query)
        # hits = self.qdrant.search(vector, limit=limit)
        # return [h.get("text", "") for h in hits]
        logger.debug("Qdrant semantic search not yet configured (no embedding function)")
        return []
