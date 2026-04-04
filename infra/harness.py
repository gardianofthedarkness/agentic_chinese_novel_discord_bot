"""
Infrastructure Harness
======================
Top-level object that owns and wires together all infra components:

    RedisPool  →  TokenStreamer
               →  BackgroundExecutor
               →  RedisTaskQueue
               →  StreamManager (legacy compat)

Every node, tool, and workflow that needs streaming or async execution
should receive an `InfraHarness` instance via dependency injection rather
than constructing its own Redis connections.

Usage (app startup)::

    harness = InfraHarness(redis_url="redis://localhost:6379/0")
    await harness.connect()

    # Pass to workflow builder
    graph = create_chat_workflow(llm=llm, db=db, harness=harness)

    # FastAPI lifespan shutdown
    await harness.close()

High-level API surface
-----------------------
harness.stream_llm_to_channel(channel, llm_gen)
    Consume the LLM stream, extract user-visible text (detects
    `send_human_message` and `FINAL ANSWER:`), publish to channel.

harness.stream_string_to_channel(channel, text)
    Stream a pre-formed string word-by-word to channel.

harness.run_tool(coro, tool_name, task_id)
    Execute a tool coroutine.  Returns result string if fast; returns
    BackgroundHandle if promoted to background (>5s).

harness.subscribe(channel)
    Async generator of tokens from a channel (for SSE endpoints).

harness.enqueue(user_message, ...)
    Enqueue a chat task; returns task_id.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Optional, Tuple

from infra.redis_client import RedisPool
from infra.token_stream import TokenStreamer
from infra.background_exec import BackgroundExecutor, BackgroundHandle

logger = logging.getLogger(__name__)


class InfraHarness:
    """
    Central infrastructure harness.

    Parameters
    ----------
    redis_url      : str   — Redis connection URL.
    bg_timeout     : float — seconds before a tool is promoted to background.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        bg_timeout: float = 5.0,
    ) -> None:
        self._redis_url = redis_url
        self._bg_timeout = bg_timeout

        self.pool = RedisPool(redis_url)
        self.streamer = TokenStreamer(self.pool)
        self.executor = BackgroundExecutor(self.pool, timeout=bg_timeout)

        # Lazy-init queue (imported here to avoid circular import)
        self._queue: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open all Redis connections."""
        await self.pool.connect()

        from utils.queue.redis_queue import RedisTaskQueue
        self._queue = RedisTaskQueue(self._redis_url)
        await self._queue.connect()

        logger.info("InfraHarness connected")

    async def close(self) -> None:
        """Close all Redis connections cleanly."""
        if self._queue:
            await self._queue.close()
        await self.pool.close()
        logger.info("InfraHarness closed")

    # ------------------------------------------------------------------
    # Queue access
    # ------------------------------------------------------------------

    @property
    def queue(self):
        if self._queue is None:
            raise RuntimeError("InfraHarness not connected — call connect() first.")
        return self._queue

    # ------------------------------------------------------------------
    # Streaming — high-level API
    # ------------------------------------------------------------------

    async def stream_llm_to_channel(
        self,
        channel: str,
        llm_gen: AsyncGenerator[str, None],
        *,
        task_id: Optional[str] = None,
    ) -> str:
        """
        Consume a live LLM token stream and route user-visible tokens to
        `channel`.  Returns the full accumulated response text.

        Simultaneously publishes raw tokens to `thinking:{task_id}` when
        `task_id` is provided.
        """
        thinking = f"thinking:{task_id}" if task_id else None
        return await self.streamer.stream_from_llm(
            channel,
            llm_gen,
            thinking_channel=thinking,
        )

    async def stream_string_to_channel(
        self,
        channel: str,
        text: str,
        *,
        chunk_size: int = 8,
    ) -> None:
        """Stream a pre-formed string to `channel` word-by-word."""
        await self.streamer.stream_string(channel, text, chunk_size=chunk_size)

    async def subscribe(
        self,
        channel: str,
        *,
        timeout: float = 120.0,
    ) -> AsyncGenerator[str, None]:
        """
        Async generator — yields tokens from `channel` until `__END__`.
        Used by the SSE endpoint in astream.py.
        """
        return self.streamer.subscribe(channel, timeout=timeout)

    # ------------------------------------------------------------------
    # Tool execution — high-level API
    # ------------------------------------------------------------------

    async def run_tool(
        self,
        coro: Any,
        *,
        tool_name: str = "unknown",
        task_id: str = "",
    ) -> Tuple[Any, bool]:
        """
        Execute a tool coroutine with background promotion.

        Returns:
            (result, is_background)
            - result is a str when is_background=False.
            - result is a BackgroundHandle when is_background=True.
        """
        return await self.executor.run(
            coro,
            tool_name=tool_name,
            task_id=task_id,
        )

    async def await_background_result(
        self,
        handle: BackgroundHandle,
        *,
        timeout: int = 300,
    ) -> str:
        """Wait for a previously promoted background job to finish."""
        return await handle.await_result(self.pool, timeout=timeout)

    # ------------------------------------------------------------------
    # Task queue — convenience pass-through
    # ------------------------------------------------------------------

    async def enqueue(
        self,
        user_message: str,
        discord_channel: str,
        *,
        character_name: Optional[str] = None,
        volume_id: Optional[int] = None,
    ) -> str:
        """Enqueue a chat task; returns task_id."""
        return await self.queue.enqueue(
            user_message,
            discord_channel,
            character_name=character_name,
            volume_id=volume_id,
        )
