"""
End Tool
========
Signals to the agent that it should stop the ReAct loop and produce a
FINAL ANSWER.  The agent calls this tool when it has gathered enough
context and is ready to compose a reply.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.core.tool_base import BaseTool, ToolInput
from app.core.tool_factory import ToolFactory


class EndToolInput(ToolInput):
    reason: str = Field(..., description="Brief explanation of why the agent is done (for logging).")


@ToolFactory.register
class EndTool(BaseTool):
    """Signal that the agent has finished and should emit a FINAL ANSWER."""

    name = "end"
    description = (
        "Call this tool when you have all the information needed to give a complete answer. "
        "After calling it, emit FINAL ANSWER: <reply>."
    )
    input_cls = EndToolInput

    def __init__(self, **deps: Any) -> None:
        pass

    async def execute(self, inputs: dict) -> str:
        return '{"status": "ready_to_finalize"}'
