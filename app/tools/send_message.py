"""
Send Message Tool
=================
Streams a message token-by-token to the Discord bot via Redis pub/sub.

The tool publishes to the run's stream_channel so the SSE endpoint in
utils/astream.py can forward each chunk to the waiting Discord client.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.core.tool_base import BaseTool, ToolInput
from app.core.tool_factory import ToolFactory


class SendMessageInput(ToolInput):
    message: str = Field(..., description="The message text to send to the user.")
    task_id: str = Field(..., description="The current task / run ID (used to identify the stream channel).")


@ToolFactory.register
class SendMessageTool(BaseTool):
    """Stream a message to the Discord user via the Redis pub/sub channel."""

    name = "send_message"
    description = "Send an intermediate or final message to the Discord user."
    input_cls = SendMessageInput

    def __init__(self, **deps: Any) -> None:
        self._harness = deps.get("harness")

    async def execute(self, inputs: dict) -> str:
        message = inputs["message"]
        task_id = inputs["task_id"]
        channel = f"stream:{task_id}"

        if self._harness:
            # Publish without __END__ — this is an intermediate message
            await self._harness.pool.publish(channel, message)
            return '{"status": "sent"}'
        return '{"status": "no_harness", "message": "' + message.replace('"', '\\"') + '"}'
