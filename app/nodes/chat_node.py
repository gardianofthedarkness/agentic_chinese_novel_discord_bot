"""
Chat Node
=========
Primary LLM reasoning node — the "agent" in the ReAct loop.

Responsibilities
----------------
- Build the system prompt (tool catalogue + optional persona).
- Call the DeepSeek LLM with the current message history.
- Parse the response:
    * If the model emits a tool_call JSON block → populate state["tool_calls"]
      so ToolNode can execute it next.
    * If the model emits a FINAL ANSWER block → set is_complete=True so the
      graph routes to EndNode.
- Append the assistant message to state["messages"].

Tool-call protocol (text-based, no native function-calling needed)
------------------------------------------------------------------
The LLM is instructed to respond in one of two formats:

    # Tool call
    TOOL_CALL: {"name": "search_events", "inputs": {"query": "...","limit": 5}}

    # Terminal response
    FINAL ANSWER: <plain text reply>

This keeps us decoupled from any provider-specific tool API.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.core.node_base import BaseNode
from app.core.node_factory import NodeFactory
from app.core.tool_factory import ToolFactory

logger = logging.getLogger(__name__)

_TOOL_CALL_RE = re.compile(r"TOOL_CALL:\s*(\{.*\})", re.DOTALL)
_FINAL_RE = re.compile(r"FINAL ANSWER:\s*(.*)", re.DOTALL)

_SYSTEM_TEMPLATE = """\
You are a helpful assistant for a Chinese novel analysis system.
You have access to the following tools:

{tool_docs}

## Response format

To call a tool respond with EXACTLY:
TOOL_CALL: {{"name": "<tool_name>", "inputs": {{...}}}}

When you have a final answer for the user, respond with EXACTLY:
FINAL ANSWER: <your response here>

Never mix the two formats in a single turn.
Always prefer using a tool when you are unsure; only emit FINAL ANSWER
when you have enough information to respond completely.
"""


def _to_api_messages(messages: list) -> List[Dict[str, str]]:
    """Convert LangChain message objects to plain dicts for the DeepSeek API."""
    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            # Text-protocol: represent tool results as user messages
            result.append({"role": "user", "content": f"[Tool result]\n{msg.content}"})
        else:
            result.append({"role": "user", "content": str(msg.content)})
    return result


@NodeFactory.register
class ChatNode(BaseNode):
    """LLM reasoning node — heart of the ReAct loop."""

    node_id = "chat"

    def __init__(self, **deps: Any) -> None:
        self._llm = deps.get("llm")
        if self._llm is None:
            raise ValueError("ChatNode requires 'llm' dependency.")
        self._harness = deps.get("harness")

    # ------------------------------------------------------------------

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        tool_docs = ToolFactory.build_tool_docs()
        system_prompt = _SYSTEM_TEMPLATE.format(tool_docs=tool_docs)

        api_messages = _to_api_messages(
            [SystemMessage(content=system_prompt)] + list(state["messages"])
        )
        logger.debug(f"ChatNode: calling LLM with {len(api_messages)} messages")

        # Accumulate the full response by draining the stream
        response_text = ""
        async for token in self._llm.stream(api_messages):
            response_text += token

        # -- Parse response -------------------------------------------
        tool_match = _TOOL_CALL_RE.search(response_text)
        final_match = _FINAL_RE.search(response_text)

        updates: Dict[str, Any] = {
            "messages": [AIMessage(content=response_text)],
        }

        if tool_match:
            try:
                call = json.loads(tool_match.group(1))
                updates["tool_calls"] = [call]
                logger.debug(f"ChatNode: tool call → {call['name']}")
            except json.JSONDecodeError as exc:
                logger.warning(f"ChatNode: malformed tool call JSON: {exc}")
        elif final_match:
            updates["is_complete"] = True
            logger.debug("ChatNode: FINAL ANSWER detected → routing to end")
        else:
            # Model didn't follow format — treat entire response as final answer
            logger.warning("ChatNode: response matched neither format; treating as FINAL ANSWER")
            updates["is_complete"] = True

        return updates
