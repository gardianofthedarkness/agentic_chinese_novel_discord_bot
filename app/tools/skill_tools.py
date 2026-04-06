"""
Skill Tools — Focus Mode
========================
Three tools that implement the skill/focus-mode system:

  list_skills    — show available skills with descriptions (always available)
  focus_skill    — load a skill: injects its prompt + activates its tools
  offload_skill  — unload a skill: removes its tools from the active set

Focus mode mechanics
--------------------
- `state["loaded_skills"]` tracks which skills are active. ChatNode derives
  the visible tool set from this on each turn via `compute_active_tools()`.
- When no skills are loaded, ALL tools are visible (default / unfocused state).
- Calling focus_skill activates only the tools that skill declares, plus the
  permanent core tools (list_skills, focus_skill, offload_skill, send_message, end).
- Multiple skills can be loaded simultaneously — their tool sets are unioned.
- offload_skill with no argument / "all" resets to unfocused (all tools visible).

The skill's prompt is returned as the tool result so the LLM immediately
incorporates the instructions into its next reasoning step.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import pkgutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field

from app.core.tool_base import BaseTool, ToolInput
from app.core.tool_factory import ToolFactory

logger = logging.getLogger(__name__)

_SKILLS_PKG = "app.skills"
_SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"

# Tools always visible regardless of focus mode
_CORE_TOOLS = {"list_skills", "focus_skill", "offload_skill", "send_message", "end"}


# ---------------------------------------------------------------------------
# Skill discovery (cached — skills don't change at runtime)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _discover_skills() -> Dict[str, Any]:
    """Scan app/skills/ once and return {skill_name: skill_class}."""
    skills: Dict[str, Any] = {}
    for module_info in pkgutil.iter_modules([str(_SKILLS_DIR)]):
        if module_info.name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"{_SKILLS_PKG}.{module_info.name}")
            for attr in mod.__dict__.values():
                if (
                    inspect.isclass(attr)
                    and hasattr(attr, "name")
                    and hasattr(attr, "description")
                    and hasattr(attr, "get_prompt")
                    and hasattr(attr, "tools")
                ):
                    skills[attr.name] = attr
        except Exception as exc:
            logger.warning(f"Failed to import skill module {module_info.name!r}: {exc}")
    return skills


def compute_active_tools(loaded_skills: List[str]) -> Optional[List[str]]:
    """Return union of loaded skills' tool sets + core tools, or None if no skills loaded."""
    if not loaded_skills:
        return None
    all_skills = _discover_skills()
    tools: set = set(_CORE_TOOLS)
    for skill_name in loaded_skills:
        skill_cls = all_skills.get(skill_name)
        if skill_cls:
            tools.update(getattr(skill_cls, "tools", []))
    return sorted(tools)


# ---------------------------------------------------------------------------
# ListSkillsTool
# ---------------------------------------------------------------------------

class ListSkillsInput(ToolInput):
    keyword: str = Field("", description="Optional keyword to filter skill names/descriptions.")


@ToolFactory.register
class ListSkillsTool(BaseTool):
    """List all available focus-mode skills with their descriptions and tool sets."""

    name = "list_skills"
    description = (
        "List available skills. Each skill is a focus mode that loads domain-specific "
        "instructions and activates the relevant tools. Call this before focus_skill."
    )
    input_cls = ListSkillsInput

    def __init__(self, **deps: Any) -> None:
        pass

    async def execute(self, inputs: dict) -> str:
        keyword = inputs.get("keyword", "").lower()
        all_skills = _discover_skills()
        matches = []
        for skill_name, skill_cls in all_skills.items():
            desc = getattr(skill_cls, "description", "")
            tools = getattr(skill_cls, "tools", [])
            if not keyword or keyword in skill_name.lower() or keyword in desc.lower():
                matches.append({"name": skill_name, "description": desc, "tools": tools})
        return json.dumps({"skills": matches, "total": len(matches)})


# ---------------------------------------------------------------------------
# FocusSkillTool
# ---------------------------------------------------------------------------

class FocusSkillInput(ToolInput):
    skill_name: str = Field(..., description="Name of the skill to focus on (from list_skills).")


@ToolFactory.register
class FocusSkillTool(BaseTool):
    """
    Load a skill into focus mode.
    Returns the skill's instructions and activates its tools.
    Use list_skills first to see available skill names.
    """

    name = "focus_skill"
    description = (
        "Activate a skill: loads its instructions and enables its tools. "
        "Multiple skills can be active simultaneously."
    )
    input_cls = FocusSkillInput

    def __init__(self, **deps: Any) -> None:
        pass

    async def execute(self, inputs: dict) -> str:
        # Note: state mutation is handled by ToolNode reading _state_updates
        # from the returned JSON. See ToolNode for the convention.
        skill_name = inputs["skill_name"]
        all_skills = _discover_skills()

        if skill_name not in all_skills:
            available = list(all_skills.keys())
            return json.dumps({
                "error": f"Skill {skill_name!r} not found.",
                "available": available,
            })

        skill_cls = all_skills[skill_name]
        prompt = skill_cls.get_prompt()
        tools = getattr(skill_cls, "tools", [])

        return json.dumps({
            "skill": skill_name,
            "prompt": prompt,
            "tools_activated": tools,
            # Special key: ToolNode reads this to mutate state
            "_state_updates": {
                "loaded_skills_add": skill_name,
            },
        })


# ---------------------------------------------------------------------------
# OffloadSkillTool
# ---------------------------------------------------------------------------

class OffloadSkillInput(ToolInput):
    skill_name: str = Field(
        "all",
        description=(
            "Name of the skill to unload, or 'all' to reset to unfocused mode "
            "(all tools visible)."
        ),
    )


@ToolFactory.register
class OffloadSkillTool(BaseTool):
    """
    Deactivate a skill: removes its tools from the active set.
    Use 'all' to exit focus mode entirely and restore all tools.
    """

    name = "offload_skill"
    description = (
        "Deactivate a skill and remove its tools. "
        "Use skill_name='all' to exit focus mode and restore all tools."
    )
    input_cls = OffloadSkillInput

    def __init__(self, **deps: Any) -> None:
        pass

    async def execute(self, inputs: dict) -> str:
        skill_name = inputs.get("skill_name", "all")
        return json.dumps({
            "offloaded": skill_name,
            "_state_updates": {
                "loaded_skills_remove": skill_name,
            },
        })
