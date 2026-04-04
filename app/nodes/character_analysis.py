"""
Character Analysis Node
========================
LangGraph node: identify and profile characters in a chapter.
"""

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

_SYSTEM = "你是中文小说角色分析专家，擅长识别角色特征、性格和关系网络。"

_PROMPT_TMPL = """
请分析以下章节文本，识别其中的所有角色。

章节文本：
{chapter_text}

请以JSON数组格式返回，每个角色包含：
- name: 角色名称
- character_type: 角色类型（protagonist/antagonist/supporting/minor）
- personality_traits: 性格特征列表
- aliases: 别名列表（如有）
- role_in_chapter: 本章中的作用（简短描述）

只返回JSON数组。
"""


async def analyze_characters_node(state: Dict[str, Any], llm) -> Dict[str, Any]:
    """
    LangGraph node: extract characters from state["chapter_text"].

    Writes results into state["characters"].
    """
    chapter_text = state.get("chapter_text", "")
    if not chapter_text.strip():
        return {**state, "characters": []}

    prompt = _PROMPT_TMPL.format(chapter_text=chapter_text[:8000])
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": prompt},
    ]

    try:
        result = await llm.chat(messages, temperature=0.2, max_tokens=2000)
        if not result["success"]:
            raise RuntimeError(result.get("error", "LLM call failed"))

        raw = result["response"].strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        characters = json.loads(raw)
        logger.info(f"Found {len(characters)} characters in chapter {state.get('chapter_index', 0)}")
        return {
            **state,
            "characters": state.get("characters", []) + characters,
            "processing_stage": "characters_analyzed",
        }
    except Exception as e:
        logger.error(f"analyze_characters_node failed: {e}")
        return {**state, "errors": state.get("errors", []) + [str(e)]}
