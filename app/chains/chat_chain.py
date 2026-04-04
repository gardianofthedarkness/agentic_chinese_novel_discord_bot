"""
Chat Chain
==========
LangGraph subgraph for answering a user message:
    retrieve_context → generate_response

The retrieval step searches Neo4j for relevant events/characters and
injects them into the state before the response node runs.
"""

import logging
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from app.nodes.chat_response import generate_chat_response_node

logger = logging.getLogger(__name__)


def create_chat_chain(llm, db=None):
    """
    Build the chat LangGraph subgraph.

    Args:
        llm:  DeepSeekClient instance.
        db:   DatabaseAdapter (used for context retrieval).

    Returns:
        Compiled LangGraph graph callable.
    """

    async def _retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pull relevant context from Neo4j for the user's message.
        Writes results into state["tool_context"].
        """
        if db is None:
            return {**state, "tool_context": []}

        user_msg: str = state.get("user_message", "")
        volume_id: Optional[int] = state.get("volume_id")
        context_parts: List[str] = []

        try:
            if db.neo4j and db.neo4j.driver:
                with db.neo4j.driver.session() as session:
                    # Search relevant events
                    params: Dict[str, Any] = {"q": user_msg, "lim": 5}
                    query = (
                        "MATCH (e:Event) "
                        "WHERE toLower(e.description) CONTAINS toLower($q) "
                    )
                    if volume_id:
                        query += "AND e.volume_id = $vol "
                        params["vol"] = volume_id
                    query += (
                        "RETURN e.description AS desc, e.event_type AS type "
                        "ORDER BY e.importance_score DESC LIMIT $lim"
                    )
                    res = session.run(query, **params)
                    events = [r["desc"] for r in res if r["desc"]]
                    if events:
                        context_parts.append("相关事件：\n" + "\n".join(f"- {e}" for e in events))

                    # Search relevant characters
                    char_res = session.run(
                        "MATCH (c:Character) "
                        "WHERE toLower(c.name) CONTAINS toLower($q) "
                        "RETURN c.name AS name, c.personality_traits AS traits LIMIT 3",
                        q=user_msg,
                    )
                    chars = [
                        f"{r['name']}（{', '.join(r['traits'] or [])}）"
                        for r in char_res
                        if r["name"]
                    ]
                    if chars:
                        context_parts.append("相关角色：\n" + "\n".join(f"- {c}" for c in chars))

        except Exception as e:
            logger.warning(f"Context retrieval failed: {e}")

        return {**state, "tool_context": context_parts}

    async def _respond(state: Dict[str, Any]) -> Dict[str, Any]:
        return await generate_chat_response_node(state, llm)

    # Build graph
    graph = StateGraph(dict)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("respond", _respond)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "respond")
    graph.add_edge("respond", END)

    return graph.compile()
