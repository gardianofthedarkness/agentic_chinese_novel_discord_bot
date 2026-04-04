"""
API Server (astream)
=====================
FastAPI server that bridges the Discord bot and the LangGraph agent.

Endpoints (Discord-bot compatible):
    POST /api/agent/chat       — enqueue task; returns task_id + SSE stream URL
    GET  /api/agent/stream/:id — SSE token stream for a task
    POST /api/agent/analyze    — direct (non-streamed) analysis
    POST /api/agent/explore    — topic deep-dive via agent
    GET  /api/agent/status     — health check
    GET  /api/agent/memory     — KB statistics
    GET  /api/characters       — list characters
    GET  /api/events           — list events

Streaming flow
--------------
1. Client POSTs to /api/agent/chat (with stream=true).
2. Server enqueues the task in Redis, returns {task_id, stream_url}.
3. Client GETs /api/agent/stream/{task_id} — SSE endpoint.
4. Worker picks up the task, runs the graph, publishes tokens to Redis channel.
5. SSE endpoint subscribes to that channel and forwards each token to the client.
6. When worker publishes __END__, the SSE generator closes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy singletons (initialised in lifespan)
# ---------------------------------------------------------------------------
_app_instance: Any = None


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """Startup / shutdown lifecycle."""
    global _app_instance

    from config import config as cfg
    from app.main import App

    _app_instance = App(cfg)
    await _app_instance.start()
    logger.info("App ready")
    yield

    await _app_instance.stop()
    logger.info("Server shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Novel Agent API",
    description="LangGraph-powered Chinese novel Q&A and processing API",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message or question")
    history: List[Dict[str, str]] = Field(default=[], description="Conversation history")
    character_name: Optional[str] = Field(None, description="Character to roleplay as")
    volume_id: Optional[int] = Field(None, description="Filter to a specific volume")
    stream: bool = Field(False, description="If true, returns SSE stream instead of JSON")
    discord_channel: str = Field("direct", description="Discord channel ID for routing")


class AnalyzeRequest(BaseModel):
    type: str = Field("summary", description="summary | characters | events | causality")
    limit: int = Field(10)
    volume_id: Optional[int] = None


class ExploreRequest(BaseModel):
    topic: str
    depth: str = "medium"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_app() -> Any:
    if _app_instance is None:
        raise HTTPException(status_code=503, detail="Application not ready")
    return _app_instance


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/agent/status")
async def agent_status():
    ready = _app_instance is not None
    return {
        "status": "online" if ready else "starting",
        "agent_type": "langgraph_react_agent_v3",
        "capabilities": ["chat", "roleplay", "novel_processing", "causality_analysis", "rag"],
        "ready": ready,
    }


@app.post("/api/agent/chat")
async def chat(request: ChatRequest):
    """
    Enqueue a chat task.

    - stream=false (default): runs synchronously, returns JSON response.
    - stream=true: enqueues and returns {task_id, stream_url} immediately;
      client should then GET /api/agent/stream/{task_id}.
    """
    instance = _require_app()

    if request.stream:
        task_id = await instance.harness.queue.enqueue(
            user_message=request.message,
            discord_channel=request.discord_channel,
            character_name=request.character_name,
            volume_id=request.volume_id,
        )
        return {
            "task_id": task_id,
            "stream_url": f"/api/agent/stream/{task_id}",
        }

    # Synchronous path: invoke graph directly (no queue)
    from app.core.state import RunConfig, make_initial_state
    import uuid

    task_id = str(uuid.uuid4())
    run_config = RunConfig(
        task_id=task_id,
        stream_channel=f"stream:{task_id}",
        character_name=request.character_name,
        volume_id=request.volume_id,
    )
    initial_state = make_initial_state(
        user_message=request.message,
        run_config=run_config,
    )

    try:
        final_state = await instance.graph.ainvoke(initial_state)
    except Exception as exc:
        logger.exception("Chat invoke error")
        raise HTTPException(status_code=500, detail=str(exc))

    # Extract final reply from last AIMessage
    from langchain_core.messages import AIMessage
    import re

    response = ""
    for msg in reversed(final_state.get("messages", [])):
        if isinstance(msg, AIMessage):
            m = re.search(r"FINAL ANSWER:\s*(.*)", msg.content, re.DOTALL)
            if m:
                response = m.group(1).strip()
            else:
                response = msg.content.strip()
            break

    return {"response": response, "task_id": task_id, "query_type": "agent"}


@app.get("/api/agent/stream/{task_id}")
async def stream_task(task_id: str):
    """
    SSE endpoint — subscribe to a running task's token stream.

    Yields Server-Sent Events:
        data: <token>\\n\\n
        data: [DONE]\\n\\n   (at end)
    """
    instance = _require_app()

    async def _sse_generator():
        try:
            gen = await instance.harness.subscribe(f"stream:{task_id}")
            async for token in gen:
                yield f"data: {token}\n\n"
        except Exception as exc:
            yield f"data: [ERROR] {exc}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/agent/explore")
async def explore(request: ExploreRequest):
    instance = _require_app()

    from app.core.state import RunConfig, make_initial_state
    import uuid

    task_id = str(uuid.uuid4())
    run_config = RunConfig(task_id=task_id, stream_channel=f"stream:{task_id}")
    initial_state = make_initial_state(
        user_message=f"Tell me everything about: {request.topic}",
        run_config=run_config,
    )

    try:
        final_state = await instance.graph.ainvoke(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    from langchain_core.messages import AIMessage
    response = ""
    for msg in reversed(final_state.get("messages", [])):
        if isinstance(msg, AIMessage):
            response = msg.content
            break

    return {"topic": request.topic, "depth": request.depth, "exploration": response}


@app.post("/api/agent/analyze")
async def analyze(request: AnalyzeRequest):
    instance = _require_app()
    db = getattr(instance, "_db", None) or getattr(instance, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database not ready")
    try:
        filters = {}
        if request.volume_id:
            filters["volume_id"] = request.volume_id
        events = await db.query_events(filters)
        return {
            "results": {
                "analysis_summary": {"timeline_events": len(events)},
                "events": [e.to_dict() for e in events[: request.limit]],
            }
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/characters")
async def list_characters(volume_id: Optional[int] = None, limit: int = 20):
    instance = _require_app()
    db = getattr(instance, "_db", None)
    if db is None:
        return {"characters": []}
    try:
        if hasattr(db, "neo4j") and db.neo4j:
            with db.neo4j.driver.session() as session:
                result = session.run(
                    "MATCH (c:Character) RETURN c.name as name, c.character_type as type LIMIT $lim",
                    lim=limit,
                )
                return {"characters": [dict(r) for r in result]}
        return {"characters": []}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/events")
async def list_events(volume_id: Optional[int] = None, limit: int = 20):
    instance = _require_app()
    db = getattr(instance, "_db", None)
    if db is None:
        return {"events": []}
    try:
        filters = {}
        if volume_id:
            filters["volume_id"] = volume_id
        events = await db.query_events(filters)
        return {"events": [e.to_dict() for e in events[:limit]]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "utils.astream:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "5005")),
        reload=False,
    )
