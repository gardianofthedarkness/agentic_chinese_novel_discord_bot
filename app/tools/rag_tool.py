"""
RAG Tool
========
Semantic search against the Qdrant vector store.

The tool embeds the query, searches the configured collection, and returns
the top-k results as a JSON list so the agent can incorporate them into its
response.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import Field

from app.core.tool_base import BaseTool, ToolInput
from app.core.tool_factory import ToolFactory

logger = logging.getLogger(__name__)


class SemanticSearchInput(ToolInput):
    query: str = Field(..., description="Natural-language search query.")
    collection: str = Field("novel_chunks", description="Qdrant collection name to search.")
    limit: int = Field(5, description="Maximum number of results to return.")
    score_threshold: float = Field(0.5, description="Minimum similarity score (0–1).")


@ToolFactory.register
class SemanticSearchTool(BaseTool):
    """Semantic (vector) search over the novel knowledge base."""

    name = "semantic_search"
    description = (
        "Search the novel knowledge base using semantic similarity. "
        "Returns relevant text chunks ranked by relevance."
    )
    input_cls = SemanticSearchInput

    def __init__(self, **deps: Any) -> None:
        self._qdrant = deps.get("qdrant")
        self._llm = deps.get("llm")

    async def execute(self, inputs: dict) -> str:
        if self._qdrant is None:
            return json.dumps({"error": "No Qdrant adapter available."})

        query = inputs["query"]
        collection = inputs["collection"]
        limit = inputs["limit"]
        threshold = inputs["score_threshold"]

        # Embed the query using the LLM helper (DeepSeekClient has no embed endpoint,
        # so we fall back to a simple keyword search if embedding is unavailable).
        try:
            if hasattr(self._llm, "embed") and callable(self._llm.embed):
                vector = await self._llm.embed(query)
                results = await self._qdrant.search(
                    collection_name=collection,
                    query_vector=vector,
                    limit=limit,
                    score_threshold=threshold,
                )
            else:
                # Fallback: scroll all and do substring filter (dev/demo mode)
                all_points = await self._qdrant.scroll_all(collection)
                results = [
                    p for p in all_points
                    if query.lower() in str(p.get("payload", "")).lower()
                ][:limit]

            return json.dumps({"results": results})
        except Exception as exc:
            logger.error(f"SemanticSearchTool error: {exc}")
            return json.dumps({"error": str(exc)})
