"""
Chat Workflow
=============
Wires together ChatNode → ToolNode → EndNode in a ReAct loop using
GraphBuilder, NodeFactory, and ToolFactory.

Graph topology
--------------

    ┌─────────┐
    │  START  │
    └────┬────┘
         │
         ▼
    ┌─────────┐   tool_call    ┌──────────┐
    │  chat   │ ─────────────► │   tool   │
    └─────────┘                └────┬─────┘
         │                          │ (always loops back)
         │ is_complete              ▼
         └──────────────────► ┌──────────┐
                               │   end    │
                               └────┬─────┘
                                    │
                                    ▼
                                  END

Routing logic
-------------
After ChatNode runs, `_route_after_chat` inspects the state:
- If tool_calls is non-empty and is_complete is False → go to "tool"
- Otherwise → go to "end"

After ToolNode runs there is always an unconditional edge back to "chat"
(the ReAct loop).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langgraph.graph import END

from app.core.graph_builder import GraphBuilder
from app.core.node_factory import NodeFactory
from app.core.state import AgentState

# Ensure all nodes are registered by importing the modules.
import app.nodes.chat_node   # noqa: F401
import app.nodes.tool_node   # noqa: F401
import app.nodes.end_node    # noqa: F401

# Ensure all tools are registered.
import app.tools.send_message   # noqa: F401
import app.tools.end_tool       # noqa: F401
import app.tools.file_tools     # noqa: F401
import app.tools.neo4j_tools    # noqa: F401
import app.tools.rag_tool       # noqa: F401
import app.tools.skill_tools    # noqa: F401

logger = logging.getLogger(__name__)


def _route_after_chat(state: Dict[str, Any]) -> str:
    """Decide whether to call a tool or finish."""
    if state.get("tool_calls") and not state.get("is_complete", False):
        return "tool"
    return "end"


def create_chat_workflow(
    *,
    llm: Any,
    db: Any = None,
    qdrant: Any = None,
    harness: Any = None,
    checkpointer: Any = None,
):
    """
    Build and compile the ReAct chat graph.

    Args:
        llm          : DeepSeekClient (or compatible) instance.
        db           : DatabaseAdapter — passed to tool nodes.
        qdrant       : QdrantAdapter   — passed to SemanticSearchTool.
        harness      : InfraHarness    — Redis streaming + background execution.
        checkpointer : Optional LangGraph checkpointer for conversation memory.

    Returns:
        Compiled LangGraph graph ready for .invoke() / .astream().
    """
    deps: Dict[str, Any] = {
        "llm": llm,
        "db": db,
        "qdrant": qdrant,
        "harness": harness,
    }

    nodes = NodeFactory.build_all(**deps)
    logger.info(f"chat_workflow: built nodes: {list(nodes.keys())}")

    graph = (
        GraphBuilder(AgentState)
        .add_nodes(nodes.values())
        .set_entry("chat")
        .add_conditional_edges(
            "chat",
            _route_after_chat,
            {"tool": "tool", "end": "end"},
        )
        .add_edge("tool", "chat")   # ReAct loop
        .add_terminal_edge("end")   # end → END sentinel
        .compile(checkpointer=checkpointer)
    )

    logger.info("chat_workflow: graph compiled successfully")
    return graph
