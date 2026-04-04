"""
Event Extraction Node
======================
LangGraph node: given a chapter text, extract structured timeline events.
"""

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

_SYSTEM = (
    "你是一位专业的中文小说情节分析师。"
    "请从给定的章节文本中提取所有重要事件，并以JSON格式返回。"
)

_PROMPT_TMPL = """
请分析以下章节文本，提取所有重要事件。

章节文本：
{chapter_text}

请以JSON数组格式返回，每个事件包含以下字段：
- description: 事件描述（中文，简明扼要）
- event_type: 事件类型（battle/romance/revelation/dialogue/travel/other）
- importance_score: 重要程度（0.0-1.0）
- primary_actors: 主要参与角色名称列表
- affected_characters: 受影响角色名称列表
- temporal_markers: 时间标记（如有）

只返回JSON数组，不要其他文字。
"""


async def extract_events_node(state: Dict[str, Any], llm) -> Dict[str, Any]:
    """
    LangGraph node: extract events from state["chapter_text"].

    Writes extracted events into state["events"].
    """
    chapter_text = state.get("chapter_text", "")
    if not chapter_text.strip():
        return {**state, "events": [], "errors": state.get("errors", []) + ["Empty chapter text"]}

    prompt = _PROMPT_TMPL.format(chapter_text=chapter_text[:8000])  # guard against huge inputs

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": prompt},
    ]

    try:
        from helpers.llm import DeepSeekClient  # avoid circular if called standalone
        result = await llm.chat(messages, temperature=0.2, max_tokens=3000)
        if not result["success"]:
            raise RuntimeError(result.get("error", "LLM call failed"))

        raw = result["response"]
        # Strip markdown fences if present
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        events = json.loads(raw)

        chapter_index = state.get("chapter_index", 0)
        volume_id = state.get("volume_id", 1)

        # Annotate with chapter/volume context
        for i, ev in enumerate(events):
            ev.setdefault("chapter_index", chapter_index)
            ev.setdefault("volume_id", volume_id)
            ev.setdefault("batch_id", 0)

        logger.info(f"Extracted {len(events)} events from chapter {chapter_index}")
        return {**state, "events": state.get("events", []) + events, "processing_stage": "events_extracted"}

    except json.JSONDecodeError as e:
        logger.warning(f"Event extraction JSON parse error: {e}")
        return {**state, "errors": state.get("errors", []) + [f"JSON parse error: {e}"]}
    except Exception as e:
        logger.error(f"extract_events_node failed: {e}")
        return {**state, "errors": state.get("errors", []) + [str(e)]}
