"""
app.main
========
Application entry point.

Builds the dependency graph (LLM, DB, Qdrant, StreamManager, Redis queue)
and wires them into the compiled LangGraph workflow.

The old NovelAgent God-class has been replaced by:
  - app.core.*          — OOP foundation (state, factories, builder)
  - app.nodes.*         — ChatNode, ToolNode, EndNode
  - app.tools.*         — concrete tool implementations
  - app.workflows.*     — compiled graph factories
  - utils.queue.*       — Redis queue + StreamManager + worker
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from app.workflows.chat_workflow import create_chat_workflow
from infra.harness import InfraHarness
from utils.queue.worker import AsyncAgentWorker

logger = logging.getLogger(__name__)


class App:
    """
    Top-level application object.

    Usage::

        app = App(config)
        await app.start()           # connect all services, start worker
        # ... serve requests ...
        await app.stop()            # clean shutdown
    """

    def __init__(self, config: Any) -> None:
        self._config = config
        self.graph: Any = None
        self.harness: Optional[InfraHarness] = None
        self._db: Any = None
        self._worker: Optional[AsyncAgentWorker] = None
        self._worker_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect services, compile graph, start background worker."""
        cfg = self._config

        # --- LLM ----------------------------------------------------------
        from helpers.llm import create_llm_client
        llm = create_llm_client(cfg.deepseek_api_key)

        # --- Databases ----------------------------------------------------
        qdrant: Any = None
        try:
            from db.database_adapter import DatabaseAdapter
            self._db = DatabaseAdapter(cfg)
            await self._db.connect()
            logger.info("DatabaseAdapter connected")
        except Exception as exc:
            logger.warning(f"DB unavailable (continuing without): {exc}")

        try:
            from db.qdrant_adapter import QdrantAdapter
            qdrant = QdrantAdapter(cfg)
            await qdrant.connect()
            logger.info("QdrantAdapter connected")
        except Exception as exc:
            logger.warning(f"Qdrant unavailable (continuing without): {exc}")

        # --- Infra harness (Redis) ----------------------------------------
        redis_url = getattr(cfg, "redis_url", "redis://localhost:6379/0")
        self.harness = InfraHarness(redis_url=redis_url)
        await self.harness.connect()

        # --- Graph --------------------------------------------------------
        self.graph = create_chat_workflow(
            llm=llm,
            db=self._db,
            qdrant=qdrant,
            harness=self.harness,
        )

        # --- Worker -------------------------------------------------------
        self._worker = AsyncAgentWorker(
            queue=self.harness.queue,
            graph=self.graph,
            harness=self.harness,
        )
        self._worker_task = asyncio.create_task(self._worker.run())
        logger.info("App started — worker running")

    async def stop(self) -> None:
        if self._worker:
            await self._worker.stop()
        if self._worker_task:
            self._worker_task.cancel()
        if self.harness:
            await self.harness.close()
        logger.info("App stopped")
