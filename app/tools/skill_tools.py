"""
Skill Tools
===========
Allow the agent to discover and load skills at runtime.

SearchSkillTool  — list available skills and their descriptions.
LoadSkillTool    — load a skill's prompt/instructions into the conversation.

Skills live in app/skills/ as Python modules.  Each skill module is expected
to expose a class with:
    name: str               — unique skill id
    description: str        — one-sentence description
    get_prompt() -> str     — returns the skill's system-level instructions
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import pkgutil
from pathlib import Path
from typing import Any

from pydantic import Field

from app.core.tool_base import BaseTool, ToolInput
from app.core.tool_factory import ToolFactory

logger = logging.getLogger(__name__)

_SKILLS_PKG = "app.skills"
_SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"


def _discover_skills() -> dict[str, Any]:
    """Import all modules in app/skills and collect skill classes."""
    skills = {}
    for module_info in pkgutil.iter_modules([str(_SKILLS_DIR)]):
        if module_info.name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"{_SKILLS_PKG}.{module_info.name}")
            for _, obj in inspect.getmembers(mod, inspect.isclass):
                if hasattr(obj, "name") and hasattr(obj, "description") and hasattr(obj, "get_prompt"):
                    skills[obj.name] = obj
        except Exception as exc:
            logger.warning(f"Failed to import skill module {module_info.name!r}: {exc}")
    return skills


# ---------------------------------------------------------------------------
# SearchSkillTool
# ---------------------------------------------------------------------------

class SearchSkillInput(ToolInput):
    keyword: str = Field("", description="Optional keyword to filter skill names/descriptions.")


@ToolFactory.register
class SearchSkillTool(BaseTool):
    """List available skills, optionally filtered by keyword."""

    name = "search_skill"
    description = "Search available skills by keyword. Returns skill names and descriptions."
    input_cls = SearchSkillInput

    def __init__(self, **deps: Any) -> None:
        pass

    async def execute(self, inputs: dict) -> str:
        keyword = inputs.get("keyword", "").lower()
        all_skills = _discover_skills()

        matches = []
        for skill_name, skill_cls in all_skills.items():
            desc = getattr(skill_cls, "description", "")
            if not keyword or keyword in skill_name.lower() or keyword in desc.lower():
                matches.append({"name": skill_name, "description": desc})

        return json.dumps({"skills": matches, "total": len(matches)})


# ---------------------------------------------------------------------------
# LoadSkillTool
# ---------------------------------------------------------------------------

class LoadSkillInput(ToolInput):
    skill_name: str = Field(..., description="The unique name of the skill to load.")


@ToolFactory.register
class LoadSkillTool(BaseTool):
    """Load a skill's prompt/instructions so they can be used in the current conversation."""

    name = "load_skill"
    description = "Load a skill's detailed instructions into the conversation context."
    input_cls = LoadSkillInput

    def __init__(self, **deps: Any) -> None:
        pass

    async def execute(self, inputs: dict) -> str:
        skill_name = inputs["skill_name"]
        all_skills = _discover_skills()

        if skill_name not in all_skills:
            available = list(all_skills.keys())
            return json.dumps({"error": f"Skill {skill_name!r} not found.", "available": available})

        skill_cls = all_skills[skill_name]
        try:
            prompt = skill_cls().get_prompt()
            return json.dumps({"skill": skill_name, "prompt": prompt})
        except Exception as exc:
            return json.dumps({"error": str(exc)})
