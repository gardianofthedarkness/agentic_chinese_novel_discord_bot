"""
app.core
========
OOP foundation: state schema, base classes, factories, and graph builder.
"""

from app.core.state import AgentState, RunConfig, make_initial_state
from app.core.tool_base import BaseTool, ToolInput
from app.core.tool_factory import ToolFactory
from app.core.node_base import BaseNode
from app.core.node_factory import NodeFactory
from app.core.graph_builder import GraphBuilder

__all__ = [
    "AgentState",
    "RunConfig",
    "make_initial_state",
    "BaseTool",
    "ToolInput",
    "ToolFactory",
    "BaseNode",
    "NodeFactory",
    "GraphBuilder",
]
