"""
Causality Analysis Node
========================
LangGraph node: infer causal relationships between extracted events.
"""

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_SYSTEM = "你是中文小说情节因果关系分析专家。请分析事件之间的因果逻辑。"

_PROMPT_TMPL = """
以下是从同一章节提取的事件列表（按顺序排列）：

{events_text}

请分析这些事件之间的因果关系，返回JSON数组，每条因果链包含：
- from_event_index: 原因事件的索引（0-based）
- to_event_index: 结果事件的索引（0-based）
- causality_type: 因果类型（direct/indirect/contributing）
- strength: 因果强度（0.0-1.0）
- reasoning: 简短推理说明

只返回JSON数组，若无明确因果关系则返回 []。
"""


async def analyze_causality_node(state: Dict[str, Any], llm) -> Dict[str, Any]:
    """
    LangGraph node: infer causal links between events in state["events"].

    Writes results into state["causal_links"].
    """
    events: List[Dict] = state.get("events", [])
    if len(events) < 2:
        return {**state, "causal_links": []}

    # Condense event list for the prompt
    lines = [f"{i}. {ev.get('description', '')}" for i, ev in enumerate(events)]
    events_text = "\n".join(lines[:30])  # cap at 30 events per batch

    prompt = _PROMPT_TMPL.format(events_text=events_text)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": prompt},
    ]

    try:
        result = await llm.chat(messages, temperature=0.1, max_tokens=2000)
        if not result["success"]:
            raise RuntimeError(result.get("error", "LLM call failed"))

        raw = result["response"].strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        raw_links = json.loads(raw)

        # Resolve event_index → event_id where possible
        causal_links = []
        for link in raw_links:
            from_i = link.get("from_event_index", -1)
            to_i = link.get("to_event_index", -1)
            if 0 <= from_i < len(events) and 0 <= to_i < len(events):
                causal_links.append({
                    "from_event": events[from_i].get("event_id", f"ev_{from_i}"),
                    "to_event": events[to_i].get("event_id", f"ev_{to_i}"),
                    "causality_type": link.get("causality_type", "contributing"),
                    "strength": float(link.get("strength", 0.5)),
                    "reasoning": link.get("reasoning", ""),
                    "confidence": float(link.get("strength", 0.5)),
                })

        logger.info(f"Found {len(causal_links)} causal links")
        return {
            **state,
            "causal_links": state.get("causal_links", []) + causal_links,
            "processing_stage": "causality_analyzed",
        }
    except Exception as e:
        logger.error(f"analyze_causality_node failed: {e}")
        return {**state, "errors": state.get("errors", []) + [str(e)]}
