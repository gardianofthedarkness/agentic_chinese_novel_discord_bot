"""
Causality Tools
===============
Agent-callable tool that leverages the LLM to reason about causal links
between events, then persists the result in Neo4j.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def analyze_causality(event_id_a: str, event_id_b: str, db, llm) -> str:
    """
    Determine whether event A caused event B and store the link.

    Steps:
    1. Fetch descriptions of both events from Neo4j.
    2. Ask the LLM to evaluate the causal relationship.
    3. If causal, persist a CAUSES edge in Neo4j.

    Args:
        event_id_a:  Source event ID.
        event_id_b:  Target event ID.
        db:          DatabaseAdapter instance.
        llm:         DeepSeekClient instance.

    Returns:
        JSON string with analysis result.
    """
    logger.info(f"analyze_causality({event_id_a!r}, {event_id_b!r})")
    try:
        # 1. Fetch events
        desc_a = desc_b = None
        if db.neo4j and db.neo4j.driver:
            with db.neo4j.driver.session() as session:
                res = session.run(
                    "MATCH (e:Event) WHERE e.event_id IN [$a, $b] "
                    "RETURN e.event_id AS id, e.description AS desc",
                    a=event_id_a,
                    b=event_id_b,
                )
                for r in res:
                    if r["id"] == event_id_a:
                        desc_a = r["desc"]
                    else:
                        desc_b = r["desc"]

        if not desc_a or not desc_b:
            return json.dumps({"error": "One or both events not found"})

        # 2. LLM analysis
        prompt = (
            "你是一位中文小说情节分析专家。请判断以下两个事件之间是否存在因果关系。\n\n"
            f"事件A：{desc_a}\n"
            f"事件B：{desc_b}\n\n"
            "请返回严格的JSON格式（不要添加其他文字）：\n"
            '{"is_causal": true/false, "causality_type": "direct/indirect/contributing", '
            '"strength": 0.0-1.0, "reasoning": "简短说明"}'
        )
        raw = await llm.generate(prompt, temperature=0.2)

        # 3. Parse and persist
        try:
            analysis = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            import re
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            analysis = json.loads(m.group()) if m else {"is_causal": False, "reasoning": raw}

        if analysis.get("is_causal") and db.neo4j and db.neo4j.driver:
            from db.base_adapter import CausalLink
            link = CausalLink(
                from_event=event_id_a,
                to_event=event_id_b,
                causality_type=analysis.get("causality_type", "contributing"),
                strength=float(analysis.get("strength", 0.5)),
                reasoning=analysis.get("reasoning", ""),
                confidence=float(analysis.get("strength", 0.5)),
            )
            await db.store_causal_link(link)

        return json.dumps(
            {
                "event_a": event_id_a,
                "event_b": event_id_b,
                "analysis": analysis,
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as e:
        logger.error(f"analyze_causality failed: {e}")
        return json.dumps({"error": str(e)})
