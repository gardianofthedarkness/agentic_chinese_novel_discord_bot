"""
Background Executor
===================
Wraps async tool coroutines with a configurable timeout.

If a tool completes within the timeout → result returned immediately (fast path).
If a tool exceeds the timeout → promoted to a background asyncio task; a
`BackgroundHandle` is returned immediately so the graph can continue.

Background lifecycle
--------------------
1. `BackgroundExecutor.run(coro, job_id, task_id)` is called by ToolNode.
2. Within `timeout` seconds (default 5):
   - Tool finishes → `(result, False)` returned.
3. If timeout fires:
   - Background asyncio task is spawned.
   - `(BackgroundHandle(job_id), True)` returned.
   - Redis hash `bg:job:{job_id}` is written with status="pending".
4. When the background task finishes:
   - Result pushed to Redis list `bg:result:{job_id}`.
   - Hash updated with status="done".
   - Expiry set to 1 hour.
5. Caller retrieves the result via `BackgroundHandle.await_result()` which
   BLPOPs from `bg:result:{job_id}`.  ToolNode can either await it
   synchronously or publish a "working on it..." message and let the worker
   inject the result asynchronously.

Redis key layout
----------------
bg:job:{job_id}      HASH   {status, tool_name, task_id, error?}
bg:result:{job_id}   LIST   single element — serialised result string
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import uuid
from typing import Any, Coroutine, Optional, Tuple

from infra.redis_client import RedisPool

logger = logging.getLogger(__name__)

_BG_TTL = 3600          # seconds — how long to keep bg job keys in Redis
_DEFAULT_TIMEOUT = 5.0  # seconds before a tool is promoted to background


@dataclasses.dataclass
class BackgroundHandle:
    """
    Returned when a tool is promoted to background execution.

    Callers can either:
      - `await handle.await_result(pool)` — block until the background task
        finishes (use with a generous timeout).
      - Treat it as fire-and-forget and let the worker inject results later.
    """

    job_id: str
    tool_name: str
    task_id: str

    @property
    def result_key(self) -> str:
        return f"bg:result:{self.job_id}"

    @property
    def job_key(self) -> str:
        return f"bg:job:{self.job_id}"

    async def await_result(
        self,
        pool: RedisPool,
        *,
        timeout: int = 300,
    ) -> str:
        """
        Block until the background task writes its result to Redis.

        Args:
            pool    : RedisPool instance.
            timeout : max seconds to wait (default 5 min).

        Returns:
            The tool result string.  On timeout returns a JSON error.
        """
        result = await pool.blpop(self.result_key, timeout=timeout)
        if result is None:
            return json.dumps({"error": f"Background job {self.job_id!r} timed out waiting for result."})
        _, value = result
        return value

    def to_pending_message(self) -> str:
        """Return a user-friendly 'I'm working on it' string."""
        return json.dumps({
            "status": "background",
            "job_id": self.job_id,
            "tool": self.tool_name,
            "message": f"Tool '{self.tool_name}' is running in the background. "
                       f"Results will be available shortly (job_id={self.job_id}).",
        })


class BackgroundExecutor:
    """
    Runs coroutines with timeout-based background promotion.

    Parameters
    ----------
    pool    : RedisPool — for persisting job state and results.
    timeout : float     — seconds before promoting to background (default 5.0).
    """

    def __init__(self, pool: RedisPool, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._pool = pool
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        coro: Coroutine[Any, Any, str],
        *,
        tool_name: str = "unknown",
        task_id: str = "",
        job_id: Optional[str] = None,
    ) -> Tuple[Any, bool]:
        """
        Execute `coro` with timeout.

        Returns:
            (result, is_background)
            - is_background=False → `result` is the tool's return string.
            - is_background=True  → `result` is a `BackgroundHandle`.
        """
        job_id = job_id or str(uuid.uuid4())

        try:
            result = await asyncio.wait_for(
                asyncio.shield(asyncio.ensure_future(coro)),
                timeout=self._timeout,
            )
            logger.debug(f"BackgroundExecutor: tool={tool_name!r} completed in foreground")
            return result, False

        except asyncio.TimeoutError:
            logger.info(
                f"BackgroundExecutor: tool={tool_name!r} exceeded {self._timeout}s "
                f"— promoting to background job={job_id!r}"
            )
            handle = BackgroundHandle(
                job_id=job_id,
                tool_name=tool_name,
                task_id=task_id,
            )
            # Register job in Redis
            await self._register_job(handle)
            # The original coroutine is already running (shielded above);
            # we need to re-create it since the future may have been cancelled.
            # Spawn a new background task with the same coroutine.
            asyncio.create_task(
                self._background_task(handle, coro),
                name=f"bg-tool-{job_id}",
            )
            return handle, True

        except Exception as exc:
            logger.error(f"BackgroundExecutor: tool={tool_name!r} raised {exc}")
            return json.dumps({"error": str(exc)}), False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _register_job(self, handle: BackgroundHandle) -> None:
        await self._pool.hset(
            handle.job_key,
            {
                "status": "pending",
                "tool_name": handle.tool_name,
                "task_id": handle.task_id,
                "job_id": handle.job_id,
            },
        )
        await self._pool.expire(handle.job_key, _BG_TTL)

    async def _background_task(self, handle: BackgroundHandle, coro: Any) -> None:
        """Await the coroutine, store result in Redis, update status."""
        try:
            result = await coro
            await self._pool.hset(handle.job_key, {"status": "done"})
        except Exception as exc:
            result = json.dumps({"error": str(exc)})
            await self._pool.hset(handle.job_key, {"status": "error", "error": str(exc)})
            logger.error(f"BackgroundExecutor: background job {handle.job_id!r} failed: {exc}")
        finally:
            # Push result (BLPOP-friendly single-element list)
            await self._pool.lpush(handle.result_key, result)
            await self._pool.expire(handle.result_key, _BG_TTL)
            logger.info(f"BackgroundExecutor: job {handle.job_id!r} stored result")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    async def get_job_status(self, job_id: str) -> dict:
        """Return the current metadata for a background job."""
        return await self._pool.hgetall(f"bg:job:{job_id}")
