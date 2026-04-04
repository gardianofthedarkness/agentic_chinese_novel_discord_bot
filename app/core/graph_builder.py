"""
Graph Builder
=============
Fluent builder that constructs and compiles a LangGraph StateGraph.

Hides the verbosity of the raw LangGraph API behind a clean builder
interface and enforces consistent patterns (entry point set, every node
reachable, etc.).

Usage::

    graph = (
        GraphBuilder(AgentState)
        .add_node(chat_node)
        .add_node(tool_node)
        .add_node(end_node)
        .set_entry(ChatNode.node_id)
        .add_conditional_edges(
            ChatNode.node_id,
            route_after_chat,
            {
                "tool":  ToolNode.node_id,
                "end":   EndNode.node_id,
            },
        )
        .add_edge(ToolNode.node_id, ChatNode.node_id)
        .add_edge(EndNode.node_id, END)
        .compile()
    )
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, Optional, Type

from langgraph.graph import END, StateGraph

from app.core.node_base import BaseNode

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Fluent builder for a LangGraph StateGraph.

    Parameters
    ----------
    state_schema : type
        The TypedDict (or dataclass) that defines AgentState.
        Passed directly to StateGraph().
    """

    def __init__(self, state_schema: Type) -> None:
        self._graph = StateGraph(state_schema)
        self._entry: Optional[str] = None
        self._node_ids: list[str] = []
        logger.debug(f"GraphBuilder initialised with schema {state_schema.__name__}")

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_node(self, node: BaseNode) -> "GraphBuilder":
        """
        Register a BaseNode instance with the graph.

        The node's async __call__ method is used as the LangGraph callable,
        so LangGraph invokes `await node(state)` at runtime.
        """
        self._graph.add_node(node.node_id, node)
        self._node_ids.append(node.node_id)
        logger.debug(f"  + node: {node.node_id!r}")
        return self

    def add_nodes(self, nodes: Iterable[BaseNode]) -> "GraphBuilder":
        """Add multiple nodes in one call."""
        for node in nodes:
            self.add_node(node)
        return self

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def set_entry(self, node_id: str) -> "GraphBuilder":
        self._graph.set_entry_point(node_id)
        self._entry = node_id
        return self

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def add_edge(self, from_id: str, to_id: str) -> "GraphBuilder":
        """Add a deterministic edge."""
        self._graph.add_edge(from_id, to_id)
        return self

    def add_conditional_edges(
        self,
        from_id: str,
        condition: Callable[[Dict[str, Any]], str],
        mapping: Dict[str, str],
    ) -> "GraphBuilder":
        """
        Add a conditional edge.

        Args:
            from_id:    Source node id.
            condition:  Function(state) → branch key string.
            mapping:    Dict mapping branch keys to target node ids.
        """
        self._graph.add_conditional_edges(from_id, condition, mapping)
        return self

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------

    def compile(self, checkpointer=None):
        """
        Compile and return the executable graph.

        Args:
            checkpointer: Optional LangGraph checkpointer for persistence /
                          conversation memory. Pass a MemorySaver or RedisCheckpointer.

        Raises:
            RuntimeError: if no entry point has been set.
        """
        if not self._entry:
            raise RuntimeError(
                "GraphBuilder: call set_entry(node_id) before compile()."
            )
        logger.info(
            f"Compiling graph: entry={self._entry!r}, "
            f"nodes={self._node_ids}"
        )
        kwargs: Dict[str, Any] = {}
        if checkpointer is not None:
            kwargs["checkpointer"] = checkpointer
        return self._graph.compile(**kwargs)

    # ------------------------------------------------------------------
    # Convenience: terminal edge
    # ------------------------------------------------------------------

    def add_terminal_edge(self, node_id: str) -> "GraphBuilder":
        """Add a deterministic edge from node_id to the LangGraph END sentinel."""
        self._graph.add_edge(node_id, END)
        return self
