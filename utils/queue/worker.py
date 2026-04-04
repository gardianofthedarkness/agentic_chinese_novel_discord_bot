"""
Async Agent Worker
==================
Pulls tasks from the Redis queue, runs the LangGraph agent, and streams
token output back via Redis pub/sub.

One worker handles tasks sequentially; spawn multiple workers for
parallelism (each is an asyncio task or separate process).

Usage (from app startup)::

    worker = AsyncAgentWorker(
        queue=redis_queue,
        graph=compiled_graph,
        harness=harness,
    )
    asyncio.create_task(worker.run())
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from app.core.state import RunConfig, make_initial_state

logger = logging.getLogger(__name__)


class AsyncAgentWorker:
    """
    Continuously dequeues tasks and runs the LangGraph agent.

    Parameters
    ----------
    queue          : RedisTaskQueue  — the task queue to pull from.
    graph          : compiled LangGraph graph.
    harness        : InfraHarness    — Redis streaming + background execution.
    max_concurrent : int             — max parallel graph runs (default 1).
    """

    def __init__(
        self,
        queue: Any,
        graph: Any,
        harness: Any,
        max_concurrent: int = 1,
    ) -> None:
        self._queue = queue
        self._graph = graph
        self._harness = harness
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running = False

    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main loop — call once and keep as a background task."""
        self._running = True
        logger.info("AsyncAgentWorker started")
        async for task in self._queue.iter_tasks():
            if not self._running:
                break
            asyncio.create_task(self._handle(task))

    async def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------

    async def _handle(self, task: Dict[str, Any]) -> None:
        task_id = task.get("task_id", "unknown")
        channel = task.get("stream_channel", f"stream:{task_id}")

        async with self._semaphore:
            logger.info(f"Worker: handling task {task_id!r}")
            try:
                await self._run_agent(task)
            except Exception as exc:
                logger.error(f"Worker: task {task_id!r} failed: {exc}", exc_info=True)
                # Best-effort: notify the subscriber that something went wrong
                try:
                    await self._harness.pool.publish(channel, f"[ERROR] {exc}")
                    await self._harness.pool.publish_end(channel)
                except Exception:
                    pass

    async def _run_agent(self, task: Dict[str, Any]) -> None:
        task_id = task["task_id"]
        channel = task["stream_channel"]

        run_config = RunConfig(
            task_id=task_id,
            stream_channel=channel,
            character_name=task.get("character_name"),
            volume_id=task.get("volume_id"),
        )

        initial_state = make_initial_state(
            user_message=task["user_message"],
            run_config=run_config,
        )

        logger.debug(f"Worker: invoking graph for task {task_id!r}")

        # EndNode streams the reply and publishes __END__ via InfraHarness.
        # The worker just drives the graph to completion.
        await self._graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": task_id}},
        )

        logger.info(f"Worker: task {task_id!r} complete")
