"""
Token Streamer
==============
Handles all token-level streaming to/from Redis channels.

Three publishing modes
----------------------
1. stream_from_llm(channel, llm_gen)
   Consumes an async-generator of LLM tokens, scans for the
   "send_human_message" pattern on-the-fly, and publishes only the
   user-visible message content token-by-token to `channel`.
   Simultaneously publishes raw tokens to `thinking:{channel}` for
   debug/advanced UX.  Returns the complete accumulated response text.

2. stream_string(channel, text)
   Breaks a pre-formed string into word-level chunks and publishes them
   sequentially.  Used by `SendHumanMessageTool` when the LLM has already
   finished generating and the tool just needs to forward the result.

3. stream_llm_raw(channel, llm_gen)
   Publishes every LLM token without filtering.  Used by the FINAL ANSWER
   path in ChatNode (the entire tail of the response is user-visible).

Subscribing
-----------
`subscribe(channel)` returns an async generator that yields tokens until
the `__END__` sentinel is received.

On-the-fly JSON detection (send_human_message)
-----------------------------------------------
The LLM emits something like:

    TOOL_CALL: {"name": "send_human_message", "inputs": {"message": "Hello! I found..."}}

We want to extract and stream only "Hello! I found..." token by token as
the LLM generates it.  We use a lightweight state machine that tracks our
position inside the JSON string, triggering streaming once we've consumed
`"message": "` and stopping at the closing `"`.
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum, auto
from typing import AsyncGenerator, Optional

from infra.redis_client import STREAM_END, RedisPool

logger = logging.getLogger(__name__)

# Marker we look for to enter user-visible streaming mode
_TRIGGER = '"message": "'
# How many chars of the sliding window to keep for trigger detection
_WINDOW = len(_TRIGGER) + 8


class _ScanState(Enum):
    SCANNING = auto()        # looking for TOOL_CALL / send_human_message
    IN_TRIGGER_SEARCH = auto()  # found send_human_message, looking for "message": "
    STREAMING = auto()       # inside the message value — publish each token
    DONE = auto()            # hit the closing " — stop


class TokenStreamer:
    """
    Publish and subscribe to Redis token channels.

    Parameters
    ----------
    pool : RedisPool — shared Redis connection pool.
    """

    def __init__(self, pool: RedisPool) -> None:
        self._pool = pool

    # ------------------------------------------------------------------
    # Publishing — from live LLM stream
    # ------------------------------------------------------------------

    async def stream_from_llm(
        self,
        channel: str,
        llm_gen: AsyncGenerator[str, None],
        *,
        thinking_channel: Optional[str] = None,
    ) -> str:
        """
        Consume `llm_gen`, publish user-visible tokens to `channel`.

        - Tokens that belong to the message value of a `send_human_message`
          call are published immediately as they arrive.
        - If the response contains `FINAL ANSWER:`, everything after that
          prefix is streamed to `channel`.
        - Raw tokens always go to `thinking_channel` (if set).

        Returns the full accumulated response string.
        """
        full_text = ""
        window = ""       # sliding buffer for trigger detection
        state = _ScanState.SCANNING
        final_answer_buf = ""  # accumulates after "FINAL ANSWER: " is detected
        in_final_answer = False
        escape_next = False

        async for token in llm_gen:
            full_text += token

            # Always publish raw token to thinking channel
            if thinking_channel:
                await self._pool.publish(thinking_channel, token)

            # ---- State machine ----------------------------------------
            if state == _ScanState.SCANNING:
                window = (window + token)[-_WINDOW * 4:]  # keep rolling window

                # Check for send_human_message pattern
                if "send_human_message" in window:
                    state = _ScanState.IN_TRIGGER_SEARCH
                    window = ""

                # Check for FINAL ANSWER
                if not in_final_answer and "FINAL ANSWER:" in full_text:
                    in_final_answer = True
                    idx = full_text.index("FINAL ANSWER:") + len("FINAL ANSWER:")
                    # Stream everything after the marker
                    tail = full_text[idx:].lstrip()
                    if tail:
                        await self._pool.publish(channel, tail)
                    state = _ScanState.DONE  # switch to passthrough below

            elif state == _ScanState.IN_TRIGGER_SEARCH:
                window += token
                idx = window.find(_TRIGGER)
                if idx != -1:
                    # Start streaming from right after the opening quote
                    remainder = window[idx + len(_TRIGGER):]
                    if remainder:
                        await self._pool.publish(channel, remainder)
                    state = _ScanState.STREAMING
                    window = ""
                    escape_next = False

            elif state == _ScanState.STREAMING:
                # Publish each token; stop at unescaped closing quote
                if escape_next:
                    await self._pool.publish(channel, token)
                    escape_next = False
                elif token == "\\":
                    escape_next = True
                    await self._pool.publish(channel, token)
                elif '"' in token:
                    # Split at the closing quote
                    before, _, _ = token.partition('"')
                    if before:
                        await self._pool.publish(channel, before)
                    state = _ScanState.DONE
                else:
                    await self._pool.publish(channel, token)

            elif state == _ScanState.DONE:
                # In FINAL ANSWER passthrough mode — stream all remaining tokens
                if in_final_answer:
                    await self._pool.publish(channel, token)

        # Signal end to subscribers
        await self._pool.publish_end(channel)
        if thinking_channel:
            await self._pool.publish_end(thinking_channel)

        return full_text

    # ------------------------------------------------------------------
    # Publishing — pre-formed string (word-by-word)
    # ------------------------------------------------------------------

    async def stream_string(
        self,
        channel: str,
        text: str,
        *,
        chunk_size: int = 8,
        delay: float = 0.0,
    ) -> None:
        """
        Stream a pre-formed string to `channel` in small chunks.

        Splits on whitespace boundaries when possible so words arrive
        complete rather than mid-token, giving a natural typing feel.

        Args:
            channel    : target Redis channel.
            text       : the full string to stream.
            chunk_size : approximate characters per publish call.
            delay      : optional inter-chunk delay in seconds (0 = no delay).
        """
        words = text.split(" ")
        buf = ""
        for word in words:
            buf += word + " "
            if len(buf) >= chunk_size:
                await self._pool.publish(channel, buf)
                buf = ""
                if delay:
                    await asyncio.sleep(delay)
        if buf.strip():
            await self._pool.publish(channel, buf.rstrip())
        await self._pool.publish_end(channel)

    # ------------------------------------------------------------------
    # Subscribing
    # ------------------------------------------------------------------

    async def subscribe(
        self,
        channel: str,
        *,
        timeout: float = 120.0,
    ) -> AsyncGenerator[str, None]:
        """
        Async generator — yields tokens from `channel` until `__END__`.

        Args:
            channel : Redis pub/sub channel to subscribe to.
            timeout : max seconds to wait between tokens before aborting.

        Yields:
            str — each published token (excluding the __END__ sentinel).
        """
        pubsub = self._pool.pubsub()
        await pubsub.subscribe(channel)
        logger.debug(f"TokenStreamer: subscribed to {channel!r}")

        try:
            loop = asyncio.get_event_loop()
            deadline = loop.time() + timeout

            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    logger.warning(f"TokenStreamer: timeout on {channel!r}")
                    break

                try:
                    msg = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0),
                        timeout=min(remaining, 5.0),
                    )
                except asyncio.TimeoutError:
                    continue

                if msg is None:
                    continue

                data = msg.get("data", "")
                if data == STREAM_END:
                    break
                if data:
                    yield data
                    # Reset deadline on activity
                    deadline = loop.time() + timeout

        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()
