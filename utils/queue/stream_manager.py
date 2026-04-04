"""
Stream Manager
==============
Thin async wrapper around Redis pub/sub for token streaming.

Usage (publisher side — EndNode / SendMessageTool)::

    await stream_manager.publish("stream:task-123", "Hello")
    await stream_manager.publish("stream:task-123", "__END__")

Usage (subscriber side — SSE endpoint in astream.py)::

    async for token in stream_manager.subscribe("stream:task-123"):
        if token == "__END__":
            break
        yield f"data: {token}\n\n"
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

_END_SENTINEL = "__END__"


class StreamManager:
    """
    Publishes and subscribes to Redis channels for real-time token streaming.

    Parameters
    ----------
    redis_url : str
        Redis connection URL (e.g. "redis://localhost:6379/0").
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        self._redis_url = redis_url
        self._client: aioredis.Redis | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        self._client = aioredis.from_url(self._redis_url, decode_responses=True)
        logger.info(f"StreamManager connected to {self._redis_url}")

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    async def publish(self, channel: str, token: str) -> None:
        """Publish a single token (or __END__) to the channel."""
        if self._client is None:
            raise RuntimeError("StreamManager is not connected. Call connect() first.")
        await self._client.publish(channel, token)

    # ------------------------------------------------------------------
    # Subscribe
    # ------------------------------------------------------------------

    async def subscribe(
        self,
        channel: str,
        timeout: float = 120.0,
    ) -> AsyncGenerator[str, None]:
        """
        Async generator that yields tokens from the channel until __END__.

        Args:
            channel:  Redis channel name.
            timeout:  Max seconds to wait for the next token before giving up.

        Yields:
            str — each published token (excluding __END__).
        """
        if self._client is None:
            raise RuntimeError("StreamManager is not connected.")

        pubsub = self._client.pubsub()
        await pubsub.subscribe(channel)
        logger.debug(f"StreamManager: subscribed to {channel!r}")

        try:
            deadline = asyncio.get_event_loop().time() + timeout
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    logger.warning(f"StreamManager: timeout waiting on {channel!r}")
                    break

                message = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0),
                    timeout=min(remaining, 5.0),
                )
                if message is None:
                    continue

                data = message.get("data", "")
                if data == _END_SENTINEL:
                    break
                if data:
                    yield data
        except asyncio.TimeoutError:
            logger.warning(f"StreamManager: outer timeout on {channel!r}")
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()
