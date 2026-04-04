"""
Event Tools
============
Agent-callable tools for creating and querying novel events in the knowledge graph.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


async def search_events(query: str, db, limit: int = 5) -> str:
    """
    Search for events by keyword or semantic similarity.

    Args:
        query:  Natural-language search string.
        db:     DatabaseAdapter instance.
        limit:  Max results to return.

    Returns:
        JSON string — list of matching event dicts.
    """
    logger.info(f"search_events(query={query!r}, limit={limit})")
    try:
        # 1. Try Neo4j text scan (fast enough for moderate collections)
        if db.neo4j and db.neo4j.driver:
            with db.neo4j.driver.session() as session:
                res = session.run(
                    """
                    MATCH (e:Event)
                    WHERE toLower(e.description) CONTAINS toLower($q)
                    RETURN e.event_id AS id, e.description AS description,
                           e.event_type AS type, e.importance_score AS importance,
                           e.chronological_order AS `order`
                    ORDER BY e.importance_score DESC
                    LIMIT $lim
                    """,
                    q=query,
                    lim=limit,
                )
                results = [dict(r) for r in res]
                if results:
                    return json.dumps(results, ensure_ascii=False, indent=2)

        # 2. Fallback: PostgreSQL full-text
        if db.postgres and db.postgres.is_connected():
            events = await db.query_events({})
            matched = [
                e.to_dict()
                for e in events
                if query.lower() in e.description.lower()
            ][:limit]
            return json.dumps(matched, ensure_ascii=False, indent=2)

        return json.dumps([], ensure_ascii=False)
    except Exception as e:
        logger.error(f"search_events failed: {e}")
        return json.dumps({"error": str(e)})


async def upload_event(event_data: Dict[str, Any], db) -> str:
    """
    Create or update an Event node in Neo4j.

    Required keys in event_data:
        description (str)  — human-readable event description.
    Optional keys:
        event_id, event_type, importance_score, participants (List[str]),
        chapter_index, volume_id.

    Returns:
        JSON string with {"status": "success", "event_id": "..."}.
    """
    logger.info("upload_event")
    try:
        if not db.neo4j or not db.neo4j.driver:
            return json.dumps({"error": "Neo4j not connected"})

        if "event_id" not in event_data:
            event_data["event_id"] = f"evt_{uuid.uuid4().hex[:10]}"
        event_data.setdefault("created_at", datetime.now().isoformat())
        event_data.setdefault("source", "agent")

        participants = event_data.pop("participants", [])

        with db.neo4j.driver.session() as session:
            session.run(
                "MERGE (e:Event {event_id: $props.event_id}) SET e += $props",
                props=event_data,
            )
            for name in participants:
                session.run(
                    """
                    MATCH (e:Event {event_id: $eid})
                    MERGE (c:Character {name: $name})
                    MERGE (c)-[:PARTICIPATES_IN]->(e)
                    """,
                    eid=event_data["event_id"],
                    name=name,
                )

        return json.dumps({"status": "success", "event_id": event_data["event_id"]}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"upload_event failed: {e}")
        return json.dumps({"error": str(e)})


async def modify_event(event_id: str, updates: Dict[str, Any], db) -> str:
    """
    Update properties of an existing Event node.

    Args:
        event_id:  The event to update.
        updates:   Dict of property → new value.
        db:        DatabaseAdapter instance.

    Returns:
        JSON status string.
    """
    logger.info(f"modify_event(event_id={event_id!r})")
    try:
        if not db.neo4j or not db.neo4j.driver:
            return json.dumps({"error": "Neo4j not connected"})

        with db.neo4j.driver.session() as session:
            exists = session.run(
                "MATCH (e:Event {event_id: $eid}) RETURN count(e) AS c", eid=event_id
            ).single()["c"]
            if not exists:
                return json.dumps({"error": f"Event {event_id!r} not found"})
            session.run(
                "MATCH (e:Event {event_id: $eid}) SET e += $updates RETURN e",
                eid=event_id,
                updates=updates,
            )

        return json.dumps({"status": "success", "updated": event_id}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"modify_event failed: {e}")
        return json.dumps({"error": str(e)})
