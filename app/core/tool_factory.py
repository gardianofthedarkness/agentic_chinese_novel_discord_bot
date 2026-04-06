"""
Tool Factory
============
Central registry for all agent tools.

Usage pattern
-------------

1.  Define a tool::

        @ToolFactory.register
        class SearchEventsTool(BaseTool):
            name = "search_events"
            ...

2.  The workflow creates a fully-wired tool set::

        tools = ToolFactory.build_all(db=db, llm=llm, stream_manager=sm)

3.  The ToolNode looks up and runs a tool by name::

        tool = ToolFactory.get_instance(name, **deps)
        result = await tool.safe_execute(inputs)

Design notes
------------
- Registration is idempotent: re-registering the same name overwrites silently.
- `build_all` passes **deps to every tool constructor; each tool pulls out what
  it needs and ignores the rest (see BaseTool.__init__).
- `get_all_specs()` is called once at workflow startup to build the system prompt.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

from app.core.tool_base import BaseTool

logger = logging.getLogger(__name__)


class ToolFactory:
    """
    Singleton-style class registry for BaseTool subclasses.

    All state lives on the class, so it is shared across the process without
    needing to pass a factory instance around.
    """

    _registry: Dict[str, Type[BaseTool]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, tool_cls: Type[BaseTool]) -> Type[BaseTool]:
        """
        Class decorator that adds tool_cls to the registry.

        Usage::

            @ToolFactory.register
            class MyTool(BaseTool):
                name = "my_tool"
                ...
        """
        if not hasattr(tool_cls, "name") or not tool_cls.name:
            raise ValueError(f"Tool class {tool_cls.__name__} must define a non-empty `name`.")
        if tool_cls.name in cls._registry:
            logger.debug(f"Re-registering tool {tool_cls.name!r} (overwrite)")
        cls._registry[tool_cls.name] = tool_cls
        logger.debug(f"Registered tool: {tool_cls.name!r}")
        return tool_cls

    # ------------------------------------------------------------------
    # Instance creation
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, name: str, **deps: Any) -> BaseTool:
        """
        Create a single tool instance with injected dependencies.

        Raises:
            KeyError: if name is not registered.
        """
        if name not in cls._registry:
            raise KeyError(
                f"Tool {name!r} not found in registry. "
                f"Available: {list(cls._registry)}"
            )
        return cls._registry[name](**deps)

    @classmethod
    def build_all(cls, **deps: Any) -> Dict[str, BaseTool]:
        """
        Instantiate every registered tool with the given dependencies.

        Returns:
            Dict mapping tool name → tool instance.
        """
        instances: Dict[str, BaseTool] = {}
        for name in cls._registry:
            try:
                instances[name] = cls.create(name, **deps)
            except Exception as e:
                logger.warning(f"Failed to instantiate tool {name!r}: {e}")
        return instances

    # ------------------------------------------------------------------
    # Discovery / introspection
    # ------------------------------------------------------------------

    @classmethod
    def get_all_specs(cls) -> List[Dict[str, Any]]:
        """
        Return a list of tool specification dicts for system prompt construction.

        Each dict has: name, description, parameters (JSON Schema).
        """
        return [tool_cls.to_spec() for tool_cls in cls._registry.values()]

    @classmethod
    def list_names(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._registry

    # ------------------------------------------------------------------
    # System prompt builder
    # ------------------------------------------------------------------

    @classmethod
    def build_tool_docs(cls, active_tools: Optional[List[str]] = None) -> str:
        """
        Build a human-readable tool catalogue for the LLM system prompt.

        Args:
            active_tools: If provided, only include tools in this list.
                          If None, include all registered tools.
        """
        lines = ["## Available Tools\n"]
        for spec in cls.get_all_specs():
            if active_tools is not None and spec["name"] not in active_tools:
                continue
            lines.append(f"### {spec['name']}")
            lines.append(spec["description"])
            props = spec.get("parameters", {}).get("properties", {})
            required = set(spec.get("parameters", {}).get("required", []))
            if props:
                param_lines = []
                for pname, pmeta in props.items():
                    req_marker = " (required)" if pname in required else ""
                    ptype = pmeta.get("type", "any")
                    pdesc = pmeta.get("description", "")
                    param_lines.append(f"  - `{pname}` ({ptype}{req_marker}): {pdesc}")
                lines.append("Parameters:\n" + "\n".join(param_lines))
            lines.append("")
        return "\n".join(lines)
