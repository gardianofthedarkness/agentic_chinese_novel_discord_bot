"""
Agent State
===========
Single source of truth for everything that flows through the graph.

Design principles:
- Immutable snapshots: LangGraph merges via reducer functions, never mutates.
- Typed: every field has an explicit type so IDEs and mypy can catch errors early.
- Minimal: only carry what nodes actually need; derived data lives in metadata.
"""

import operator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, Sequence

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Reducer helpers
# ---------------------------------------------------------------------------

def _replace(old: Any, new: Any) -> Any:
    """Reducer that replaces the old value with the new one (last-write-wins)."""
    return new


def _append(old: List, new: List) -> List:
    """Reducer that appends new items; keeps last 200 to prevent unbounded growth."""
    combined = (old or []) + (new or [])
    return combined[-200:]


# ---------------------------------------------------------------------------
# Core graph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    State that flows through every node in the LangGraph graph.

    Fields annotated with a reducer function are *merged* across parallel
    branches; fields annotated with `_replace` are simply overwritten.
    """

    # Conversation history (LangChain messages).
    # Capped at 100 messages by the _append reducer to prevent OOM.
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Log of (tool_name, input_dict, output_str) tuples.
    tool_calls: Annotated[List[Dict[str, Any]], _append]

    # Token stream: nodes append tokens here; the stream manager publishes them.
    output_tokens: Annotated[List[str], _append]

    # Set True by EndNode / EndTool to stop the graph.
    is_complete: Annotated[bool, _replace]

    # Per-request context injected by the worker before graph.ainvoke().
    run_config: Annotated[Optional["RunConfig"], _replace]

    # Free-form metadata (debug info, metrics, etc.).
    metadata: Annotated[Dict[str, Any], _replace]


def make_initial_state(
    user_message: str,
    run_config: "RunConfig",
    history: Optional[List[Dict[str, str]]] = None,
) -> AgentState:
    """
    Factory: build a clean AgentState for a new agent run.

    Args:
        user_message:  The user's current input text.
        run_config:    Per-request configuration (task_id, character, etc.).
        history:       Prior conversation turns as role/content dicts.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    prior: List[BaseMessage] = []
    for turn in history or []:
        if turn.get("role") == "user":
            prior.append(HumanMessage(content=turn["content"]))
        elif turn.get("role") == "assistant":
            prior.append(AIMessage(content=turn["content"]))

    return AgentState(
        messages=prior + [HumanMessage(content=user_message)],
        tool_calls=[],
        output_tokens=[],
        is_complete=False,
        run_config=run_config,
        metadata={"started_at": datetime.utcnow().isoformat()},
    )


# ---------------------------------------------------------------------------
# Per-request configuration (injected, not mutable mid-run)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    """
    Immutable per-request context passed alongside AgentState.

    Frozen so it cannot be accidentally mutated by a node.
    """
    task_id: str
    stream_channel: str              # Redis pub/sub channel for token streaming
    character_name: Optional[str] = None
    volume_id: Optional[int] = None
    max_iterations: int = 20         # Hard cap on agent loop iterations
    temperature: float = 0.7

    @classmethod
    def for_task(cls, task_id: str, **kwargs) -> "RunConfig":
        return cls(
            task_id=task_id,
            stream_channel=f"stream:{task_id}",
            **kwargs,
        )
