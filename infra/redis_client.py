"""
Redis Client
============
Singleton async Redis connection pool used by the entire infra layer.

All other infra modules acquire connections through `RedisPool` rather than
creating their own — ensures a single pool is shared across the process.

Key namespaces used across the system
--------------------------------------
stream:{task_id}          pub/sub — user-visible tokens flowing to the frontend
thinking:{task_id}        pub/sub — raw LLM tokens (debug / advanced UX)
bg:job:{job_id}           hash    — background job metadata (status, tool, task_id)
bg:result:{job_id}        list    — single-element list written when job completes
                                    (BLPOP-friendly)
"""

from __future__ import annotations

import logging
from typing import Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# Sentinel published to mark the end of any stream channel
STREAM_END = "__END__"


class RedisPool:
    """
    Async Redis connection pool — one instance per process.

    Usage::

        pool = RedisPool("redis://localhost:6379/0")
        await pool.connect()

        client = pool.client          # raw aioredis.Redis
        await pool.publish("chan", "hello")
        async for msg in pool.subscribe("chan"):
            ...
        await pool.close()
    """

    def __init__(self, url: str = "redis://localhost:6379/0") -> None:
        self._url = url
        self._client: Optional[aioredis.Redis] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        self._client = aioredis.from_url(
            self._url,
            decode_responses=True,
            max_connections=20,
        )
        # Verify connectivity
        await self._client.ping()
        logger.info(f"RedisPool connected: {self._url}")

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("RedisPool closed")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def client(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("RedisPool not connected — call connect() first.")
        return self._client

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    async def publish(self, channel: str, message: str) -> None:
        await self.client.publish(channel, message)

    async def publish_end(self, channel: str) -> None:
        await self.client.publish(channel, STREAM_END)

    async def hset(self, name: str, mapping: dict) -> None:
        await self.client.hset(name, mapping=mapping)

    async def hget(self, name: str, key: str) -> Optional[str]:
        return await self.client.hget(name, key)

    async def hgetall(self, name: str) -> dict:
        return await self.client.hgetall(name)

    async def lpush(self, key: str, *values: str) -> None:
        await self.client.lpush(key, *values)

    async def blpop(self, key: str, timeout: int = 0):
        return await self.client.blpop(key, timeout=timeout)

    async def expire(self, key: str, seconds: int) -> None:
        await self.client.expire(key, seconds)

    def pubsub(self) -> aioredis.client.PubSub:
        return self.client.pubsub()
