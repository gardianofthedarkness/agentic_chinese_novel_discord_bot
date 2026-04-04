"""
Processing Chain
================
LangGraph subgraph for ingesting a novel chapter:
    read → extract events → analyze characters → analyze causality → store

Used by the novel processing skill, not the chat agent.
"""

import logging
from functools import partial
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from app.nodes.event_extraction import extract_events_node
from app.nodes.character_analysis import analyze_characters_node
from app.nodes.causality_analysis import analyze_causality_node

logger = logging.getLogger(__name__)


def create_processing_chain(llm, db=None):
    """
    Build and compile the novel-processing LangGraph subgraph.

    Args:
        llm:  DeepSeekClient instance.
        db:   DatabaseAdapter instance (optional; used in store step).

    Returns:
        Compiled LangGraph graph callable.
    """

    # Bind LLM into each node so the graph only receives state dicts
    async def _extract(state: Dict[str, Any]) -> Dict[str, Any]:
        return await extract_events_node(state, llm)

    async def _characters(state: Dict[str, Any]) -> Dict[str, Any]:
        return await analyze_characters_node(state, llm)

    async def _causality(state: Dict[str, Any]) -> Dict[str, Any]:
        return await analyze_causality_node(state, llm)

    async def _store(state: Dict[str, Any]) -> Dict[str, Any]:
        """Persist extracted data to the database."""
        if db is None:
            return {**state, "processing_stage": "stored_skipped"}

        stored_events = 0
        stored_chars = 0

        # Store events
        for ev in state.get("events", []):
            try:
                from models import TimelineEvent
                import uuid, datetime
                event = TimelineEvent(
                    event_id=ev.get("event_id") or f"evt_{uuid.uuid4().hex[:10]}",
                    volume_id=ev.get("volume_id", 1),
                    batch_id=ev.get("batch_id", 0),
                    description=ev.get("description", ""),
                    event_type=ev.get("event_type", "other"),
                    importance_score=float(ev.get("importance_score", 0.5)),
                    chronological_order=ev.get("chapter_index"),
                    primary_actors=ev.get("primary_actors", []),
                    affected_characters=ev.get("affected_characters", []),
                    confidence_level=0.8,
                    created_at=datetime.datetime.now(),
                )
                await db.store_event(event)
                stored_events += 1
            except Exception as e:
                logger.warning(f"Failed to store event: {e}")

        # Store characters
        for ch in state.get("characters", []):
            try:
                from app.tools.character_tools import upload_character
                await upload_character(ch, db)
                stored_chars += 1
            except Exception as e:
                logger.warning(f"Failed to store character: {e}")

        # Store causal links
        for link in state.get("causal_links", []):
            try:
                from db.base_adapter import CausalLink
                await db.store_causal_link(CausalLink(**link))
            except Exception as e:
                logger.warning(f"Failed to store causal link: {e}")

        logger.info(f"Stored {stored_events} events, {stored_chars} characters")
        return {
            **state,
            "processing_stage": "stored",
            "store_summary": {"events": stored_events, "characters": stored_chars},
        }

    # Build graph
    graph = StateGraph(dict)
    graph.add_node("extract_events", _extract)
    graph.add_node("analyze_characters", _characters)
    graph.add_node("analyze_causality", _causality)
    graph.add_node("store", _store)

    graph.set_entry_point("extract_events")
    graph.add_edge("extract_events", "analyze_characters")
    graph.add_edge("analyze_characters", "analyze_causality")
    graph.add_edge("analyze_causality", "store")
    graph.add_edge("store", END)

    return graph.compile()
