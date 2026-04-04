"""
Tool Node
=========
Executes the tool call placed in state["tool_calls"] by ChatNode.

Flow
----
1. Pop the latest tool call from state["tool_calls"].
2. Look up the tool by name in ToolFactory.
3. Call tool.safe_execute(inputs) — validated, never raises.
4. Append a ToolMessage (with the result) to state["messages"].
5. Return the updated state slice so LangGraph routes back to ChatNode.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from langchain_core.messages import ToolMessage

from app.core.node_base import BaseNode
from app.core.node_factory import NodeFactory
from app.core.tool_factory import ToolFactory

logger = logging.getLogger(__name__)


@NodeFactory.register
class ToolNode(BaseNode):
    """Dispatches tool calls from state to the matching BaseTool implementation."""

    node_id = "tool"

    def __init__(self, **deps: Any) -> None:
        # Store deps so we can lazily build tool instances on first use.
        self._deps = deps
        self._harness = deps.get("harness")
        self._tool_instances: Dict[str, Any] = {}

    # ------------------------------------------------------------------

    def _get_tool(self, name: str):
        if name not in self._tool_instances:
            self._tool_instances[name] = ToolFactory.create(name, **self._deps)
        return self._tool_instances[name]

    # ------------------------------------------------------------------

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        tool_calls = state.get("tool_calls", [])
        if not tool_calls:
            logger.warning("ToolNode invoked but state['tool_calls'] is empty")
            return {
                "messages": [
                    ToolMessage(content='{"error": "no tool call in state"}', tool_call_id="none")
                ]
            }

        call = tool_calls[-1]  # most recent
        name = call.get("name", "")
        inputs = call.get("inputs", {})

        logger.debug(f"ToolNode: executing tool={name!r} inputs={inputs}")

        if not ToolFactory.is_registered(name):
            result = json.dumps({"error": f"Unknown tool: {name!r}"})
        else:
            tool = self._get_tool(name)
            run_config = state.get("run_config")
            task_id = run_config.task_id if run_config else ""

            if self._harness:
                coro = tool.safe_execute(inputs)
                raw, is_background = await self._harness.run_tool(
                    coro, tool_name=name, task_id=task_id
                )
                if is_background:
                    # Wait for the background job to complete
                    result = await self._harness.await_background_result(raw)
                else:
                    result = raw
            else:
                result = await tool.safe_execute(inputs)

        logger.debug(f"ToolNode: tool={name!r} result length={len(result)}")

        return {
            "messages": [
                ToolMessage(content=result, tool_call_id=name)
            ]
        }
