"""
Chat Node
=========
Primary LLM reasoning node — the "agent" in the ReAct loop.

Streaming tool-call detection
------------------------------
Rather than buffering the full response before parsing, this node uses a
state machine that scans tokens *as they arrive* from the LLM stream:

  SCANNING        → watching for "TOOL_CALL:" or "FINAL ANSWER:" prefixes
  IN_JSON         → buffering the JSON object char-by-char (brace-depth counting)
                    tool call is dispatched the moment the closing "}" is found
  STREAMING_ANSWER → everything after "FINAL ANSWER: " is published token-by-token
                     to the Redis stream channel so the Discord bot sees it live

This means:
  - Tool calls execute as soon as the LLM closes the JSON — no waiting for
    the rest of the response.
  - Final answers stream directly to the user token-by-token.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.core.node_base import BaseNode
from app.core.node_factory import NodeFactory
from app.core.tool_factory import ToolFactory

logger = logging.getLogger(__name__)

# Patterns we scan for (must fit inside rolling scan window)
_TOOL_PREFIX    = "TOOL_CALL:"
_ANSWER_PREFIX  = "FINAL ANSWER:"
# Rolling window size — long enough to detect either prefix split across tokens
_WINDOW         = max(len(_TOOL_PREFIX), len(_ANSWER_PREFIX)) + 8

_SYSTEM_TEMPLATE = """\
You are a helpful assistant for a Chinese novel analysis system.
You have access to the following tools:

{tool_docs}

## Response format

To call a tool respond with EXACTLY:
TOOL_CALL: {{"name": "<tool_name>", "reason": "<one short phrase explaining what you are looking for>", "inputs": {{...}}}}

When you have a final answer for the user, respond with EXACTLY:
FINAL ANSWER: <your response here>

Never mix the two formats in a single turn.
Always prefer using a tool when you are unsure; only emit FINAL ANSWER
when you have enough information to respond completely.
The "reason" field is shown to the user while the tool runs — keep it brief and specific, \
e.g. "Finding events where Li Mu appears" or "Checking relationships between characters".
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
            result.append({"role": "user", "content": f"[Tool result]\n{msg.content}"})
        else:
            result.append({"role": "user", "content": str(msg.content)})
    return result


def _feed_json_chars(
    chars: str,
    depth: int,
    in_str: bool,
    escape: bool,
    buf: str,
) -> Tuple[str, int, bool, bool, bool]:
    """
    Feed `chars` into a streaming JSON brace-depth counter.

    Returns (consumed_json, depth, in_str, escape, is_complete).
    is_complete=True when the outermost `}` has been consumed.
    """
    for ch in chars:
        buf += ch
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return buf, depth, in_str, escape, True
    return buf, depth, in_str, escape, False


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

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

        run_config = state.get("run_config")
        channel = run_config.stream_channel if run_config else None

        # Announce thinking immediately before the first LLM token arrives
        if self._harness and channel:
            await self._harness.publish_status(channel, "🧠 Thinking…")

        # ------------------------------------------------------------------
        # Streaming state machine
        # ------------------------------------------------------------------
        # States:
        #   "scanning"         — rolling-window scan for prefix
        #   "in_json"          — brace-depth counting; fire tool on close
        #   "streaming_answer" — publish tokens straight to Redis
        # ------------------------------------------------------------------

        machine   = "scanning"
        full_text = ""      # full LLM output (for message history)
        window    = ""      # rolling scan buffer
        json_buf  = ""      # accumulates JSON chars
        j_depth   = 0
        j_in_str  = False
        j_escape  = False
        detected_call: Dict[str, Any] | None = None
        is_answer = False

        async for token in self._llm.stream(api_messages):
            full_text += token

            # ── SCANNING ───────────────────────────────────────────────
            if machine == "scanning":
                window = (window + token)[-(_WINDOW * 4):]

                # Check for TOOL_CALL:
                if _TOOL_PREFIX in window:
                    machine = "in_json"
                    after = window[window.index(_TOOL_PREFIX) + len(_TOOL_PREFIX):]
                    after = after.lstrip()
                    # Feed any chars already past the prefix into the JSON parser
                    if after:
                        json_buf, j_depth, j_in_str, j_escape, done = _feed_json_chars(
                            after, j_depth, j_in_str, j_escape, json_buf
                        )
                        if done:
                            detected_call = _parse_call(json_buf)
                            machine = "done_tool"
                    window = ""
                    continue

                # Check for FINAL ANSWER:
                if _ANSWER_PREFIX in window:
                    machine = "streaming_answer"
                    is_answer = True
                    after = window[window.index(_ANSWER_PREFIX) + len(_ANSWER_PREFIX):]
                    tail = after.lstrip()
                    if tail and self._harness and channel:
                        await self._harness.pool.publish(channel, tail)
                    window = ""
                    continue

            # ── IN_JSON ────────────────────────────────────────────────
            elif machine == "in_json":
                json_buf, j_depth, j_in_str, j_escape, done = _feed_json_chars(
                    token, j_depth, j_in_str, j_escape, json_buf
                )
                if done:
                    detected_call = _parse_call(json_buf)
                    machine = "done_tool"
                    # Any trailing tokens after "}" are irrelevant for this turn

            # ── STREAMING_ANSWER ───────────────────────────────────────
            elif machine == "streaming_answer":
                if self._harness and channel:
                    await self._harness.pool.publish(channel, token)

        # Close the answer stream so SSE subscribers see [DONE]
        if is_answer and self._harness and channel:
            await self._harness.pool.publish_end(channel)

        # ------------------------------------------------------------------
        # Build state update
        # ------------------------------------------------------------------
        updates: Dict[str, Any] = {
            "messages": [AIMessage(content=full_text)],
        }

        if detected_call:
            updates["tool_calls"] = [detected_call]
            logger.debug(f"ChatNode: live tool call detected → {detected_call.get('name')!r}")
        elif is_answer:
            updates["is_complete"] = True
            updates["metadata"] = {"answer_streamed": True}
            logger.debug("ChatNode: FINAL ANSWER streamed live → routing to end")
        else:
            # Model didn't follow format — treat full response as final answer
            logger.warning("ChatNode: response matched neither format; treating as FINAL ANSWER")
            # Stream the whole thing now (late fallback)
            if self._harness and channel:
                await self._harness.streamer.stream_string(channel, full_text, delay=0.01)
            updates["is_complete"] = True

        return updates


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_call(json_buf: str) -> Dict[str, Any] | None:
    """Parse the buffered JSON and return the call dict, or None on failure."""
    try:
        call = json.loads(json_buf)
        if "name" not in call:
            logger.warning(f"ChatNode: tool call JSON missing 'name': {json_buf!r}")
            return None
        return call
    except json.JSONDecodeError as exc:
        logger.warning(f"ChatNode: malformed tool call JSON ({exc}): {json_buf!r}")
        return None
