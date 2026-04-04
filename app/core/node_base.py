"""
Node Base
=========
Abstract contract that every LangGraph node must implement.

Design principles:
- Nodes are classes with a `run(state) -> state` method.
- LangGraph calls nodes as `callable(state) -> state`; BaseNode.__call__
  bridges this by delegating to the async `run()` method.
- Each node has a stable `node_id` (class-level) so GraphBuilder can
  reference nodes by identity rather than by raw function pointer.
- Constructor accepts **deps for dependency injection (same pattern as BaseTool).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict


class BaseNode(ABC):
    """
    Abstract base for all LangGraph graph nodes.

    Class-level attributes
    ----------------------
    node_id   Unique snake_case identifier used in the StateGraph.
    """

    node_id: ClassVar[str]

    def __init__(self, **deps: Any) -> None:
        """
        Accept injected dependencies.

        Subclasses pull out what they need (db, llm, tool_instances, etc.)
        and call super().__init__() — extras are silently ignored here.
        """

    # ------------------------------------------------------------------
    # Abstract run method
    # ------------------------------------------------------------------

    @abstractmethod
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the node's logic and return an updated state slice.

        LangGraph merges the returned dict into the full state using the
        reducer functions declared in AgentState.

        Args:
            state:  The current AgentState dict.

        Returns:
            A dict containing *only* the keys this node modified.
            (LangGraph merges, not replaces, so returning the full state
            is wasteful but safe.)
        """

    # ------------------------------------------------------------------
    # LangGraph compatibility shim
    # ------------------------------------------------------------------

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Make the node directly callable for LangGraph node registration."""
        return await self.run(state)

    def __repr__(self) -> str:
        return f"<Node id={self.node_id!r} class={type(self).__name__}>"
