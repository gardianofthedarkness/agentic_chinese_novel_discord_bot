"""
Redis Task Queue
================
Simple list-based task queue backed by Redis.

Tasks are JSON-serialised dicts pushed to a Redis list.  Workers
BLPOP from the list so they wake up immediately when work arrives
without busy-polling.

Task schema::

    {
        "task_id":        str,      # UUID
        "user_message":   str,      # the user's Discord message
        "character_name": str|null, # optional roleplay persona
        "volume_id":      int|null, # optional novel volume context
        "discord_channel": str,     # for routing the reply
        "stream_channel": str,      # Redis pub/sub channel for tokens
    }
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

_DEFAULT_QUEUE = "agent:tasks"
_BLPOP_TIMEOUT = 5  # seconds; 0 = block forever


class RedisTaskQueue:
    """
    Async task queue backed by a Redis list.

    Parameters
    ----------
    redis_url  : str   — Redis connection URL.
    queue_name : str   — Key name for the list (default: "agent:tasks").
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        queue_name: str = _DEFAULT_QUEUE,
    ) -> None:
        self._redis_url = redis_url
        self._queue = queue_name
        self._client: aioredis.Redis | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        self._client = aioredis.from_url(self._redis_url, decode_responses=True)
        logger.info(f"RedisTaskQueue connected: queue={self._queue!r}")

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        if self._client is None:
            raise RuntimeError("RedisTaskQueue is not connected. Call connect() first.")

    async def enqueue(
        self,
        user_message: str,
        discord_channel: str,
        *,
        character_name: Optional[str] = None,
        volume_id: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Push a new task onto the queue and return the task_id."""
        self._ensure_connected()

        task_id = str(uuid.uuid4())
        task: Dict[str, Any] = {
            "task_id": task_id,
            "user_message": user_message,
            "character_name": character_name,
            "volume_id": volume_id,
            "discord_channel": discord_channel,
            "stream_channel": f"stream:{task_id}",
        }
        if extra:
            task.update(extra)

        await self._client.rpush(self._queue, json.dumps(task))
        logger.debug(f"Enqueued task {task_id!r} → {self._queue!r}")
        return task_id

    # ------------------------------------------------------------------
    # Dequeue (worker side)
    # ------------------------------------------------------------------

    async def dequeue(self) -> Optional[Dict[str, Any]]:
        """
        Blocking pop from the queue.

        Returns a task dict, or None on timeout (so the worker loop can
        check a shutdown flag and re-enter the wait).
        """
        self._ensure_connected()
        result = await self._client.blpop(self._queue, timeout=_BLPOP_TIMEOUT)
        if result is None:
            return None
        _, payload = result
        task = json.loads(payload)
        logger.debug(f"Dequeued task {task.get('task_id')!r}")
        return task

    async def iter_tasks(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Continuously yield tasks until the connection is closed."""
        while True:
            task = await self.dequeue()
            if task is not None:
                yield task

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    async def depth(self) -> int:
        """Return the number of tasks currently waiting in the queue."""
        self._ensure_connected()
        return await self._client.llen(self._queue)
