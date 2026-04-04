"""
Tool Base
=========
Abstract contract that every agent tool must implement.

Design principles:
- Tools are classes, not bare functions. This makes them:
    - Testable (inject mocks via __init__)
    - Discoverable (ToolFactory can enumerate them)
    - Self-documenting (name + description + input_schema on the class itself)
- Tools are registered with ToolFactory via the @ToolFactory.register decorator.
- Tools return a plain string — the tool node serialises it into the message history.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Type

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Input schema base
# ---------------------------------------------------------------------------

class ToolInput(BaseModel):
    """
    Base class for tool input schemas.

    Subclass and declare fields with type annotations + docstrings.
    The schema is automatically extracted for the LLM system prompt.

    Example::

        class SearchEventsInput(ToolInput):
            query: str = Field(..., description="Search query string")
            limit: int = Field(5, description="Max results to return")
    """

    class Config:
        extra = "forbid"      # reject unknown keys from LLM output
        use_enum_values = True


# ---------------------------------------------------------------------------
# Base tool
# ---------------------------------------------------------------------------

class BaseTool(ABC):
    """
    Abstract base for all agent tools.

    Class-level attributes
    ----------------------
    name         Unique snake_case identifier used in tool calls.
    description  One-sentence description shown in the system prompt.
    input_cls    Pydantic model class that validates the LLM's input dict.

    Constructor
    -----------
    Accepts **deps keyword arguments so NodeFactory / ToolFactory can inject
    shared resources (db, llm, stream_manager, etc.) without knowing the
    exact signature of each subclass.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    input_cls: ClassVar[Type[ToolInput]]

    def __init__(self, **deps: Any) -> None:
        # Subclasses pull out what they need; extras are silently ignored.
        pass

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    @abstractmethod
    async def execute(self, inputs: Dict[str, Any]) -> str:
        """
        Run the tool and return a plain-text result.

        The result will be appended to the message history as a tool output.
        Always return a non-empty string. On error, return a JSON-encoded
        error dict rather than raising — the agent must be able to reason
        about failures.

        Args:
            inputs:  Dict of validated input values (keys match input_cls fields).

        Returns:
            str — JSON or plain text to feed back to the LLM.
        """

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    async def safe_execute(self, raw_inputs: Dict[str, Any]) -> str:
        """
        Validate inputs against input_cls, then call execute().

        Returns a JSON error string on validation failure instead of raising,
        so the agent can self-correct without crashing the graph.
        """
        try:
            validated = self.input_cls(**raw_inputs)
            return await self.execute(validated.dict())
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    # ------------------------------------------------------------------
    # Schema / spec
    # ------------------------------------------------------------------

    @classmethod
    def to_spec(cls) -> Dict[str, Any]:
        """
        Return a tool specification dict suitable for injection into the
        LLM system prompt.
        """
        schema = cls.input_cls.schema() if hasattr(cls, "input_cls") else {}
        return {
            "name": cls.name,
            "description": cls.description,
            "parameters": schema,
        }

    def __repr__(self) -> str:
        return f"<Tool name={self.name!r}>"
