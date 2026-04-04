"""
End Node
========
Terminal node that finalises a completed agent run.

Responsibilities
----------------
- Extract the final reply text from the last AIMessage in the conversation.
- Optionally publish the final reply to the Redis stream channel so the
  Discord bot (or any SSE subscriber) receives it.
- Mark the run as complete in state.

The node is intentionally thin — heavy post-processing (e.g. saving a summary
to the DB) should be done in a dedicated post-processing node inserted before
EndNode in the graph.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage

from app.core.node_base import BaseNode
from app.core.node_factory import NodeFactory

logger = logging.getLogger(__name__)

_FINAL_RE = re.compile(r"FINAL ANSWER:\s*(.*)", re.DOTALL)


@NodeFactory.register
class EndNode(BaseNode):
    """Finalises the agent run and publishes the reply."""

    node_id = "end"

    def __init__(self, **deps: Any) -> None:
        self._harness = deps.get("harness")

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_final_text(state: Dict[str, Any]) -> str:
        """Pull the last FINAL ANSWER text from the message history."""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage):
                match = _FINAL_RE.search(msg.content)
                if match:
                    return match.group(1).strip()
                # Fallback: return full content if no marker found
                return msg.content.strip()
        return ""

    # ------------------------------------------------------------------

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        final_text = self._extract_final_text(state)
        logger.info(f"EndNode: run complete, reply length={len(final_text)}")

        if self._harness and state.get("run_config"):
            channel = state["run_config"].stream_channel
            try:
                # stream_string_to_channel streams word-by-word then sends __END__
                await self._harness.stream_string_to_channel(channel, final_text)
                logger.debug(f"EndNode: streamed reply to channel={channel!r}")
            except Exception as exc:
                logger.warning(f"EndNode: failed to publish to stream: {exc}")

        return {"is_complete": True}
