"""
Node Factory
============
Central registry for all LangGraph graph nodes.

Mirrors ToolFactory's pattern: register with a decorator, build with deps.

Usage::

    @NodeFactory.register
    class ChatNode(BaseNode):
        node_id = "chat"
        ...

    # In workflow builder:
    nodes = NodeFactory.build_all(db=db, llm=llm, tools=tool_instances)
    graph = GraphBuilder(AgentState) \
        .add_nodes(nodes.values()) \
        .set_entry("chat") \
        ...
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Type

from app.core.node_base import BaseNode

logger = logging.getLogger(__name__)


class NodeFactory:
    """Singleton-style registry for BaseNode subclasses."""

    _registry: Dict[str, Type[BaseNode]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, node_cls: Type[BaseNode]) -> Type[BaseNode]:
        """
        Class decorator that registers node_cls.

        Usage::

            @NodeFactory.register
            class MyNode(BaseNode):
                node_id = "my_node"
        """
        if not hasattr(node_cls, "node_id") or not node_cls.node_id:
            raise ValueError(
                f"Node class {node_cls.__name__} must define a non-empty `node_id`."
            )
        cls._registry[node_cls.node_id] = node_cls
        logger.debug(f"Registered node: {node_cls.node_id!r}")
        return node_cls

    # ------------------------------------------------------------------
    # Instance creation
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, node_id: str, **deps: Any) -> BaseNode:
        """Instantiate a single node with injected dependencies."""
        if node_id not in cls._registry:
            raise KeyError(
                f"Node {node_id!r} not found. Available: {list(cls._registry)}"
            )
        return cls._registry[node_id](**deps)

    @classmethod
    def build_all(cls, **deps: Any) -> Dict[str, BaseNode]:
        """
        Instantiate every registered node.

        Returns:
            Dict mapping node_id → node instance.
        """
        instances: Dict[str, BaseNode] = {}
        for nid in cls._registry:
            try:
                instances[nid] = cls.create(nid, **deps)
            except Exception as e:
                logger.warning(f"Failed to instantiate node {nid!r}: {e}")
        return instances

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @classmethod
    def list_ids(cls) -> List[str]:
        return list(cls._registry.keys())
