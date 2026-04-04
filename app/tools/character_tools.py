"""
Character Tools
===============
Agent-callable tools for querying and creating characters.
"""

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


async def search_characters(name_query: str, db) -> str:
    """
    Find characters whose name contains the query string.

    Args:
        name_query: Partial or full character name.
        db:         DatabaseAdapter instance.

    Returns:
        JSON list of character dicts.
    """
    logger.info(f"search_characters(query={name_query!r})")
    try:
        if not db.neo4j or not db.neo4j.driver:
            return json.dumps({"error": "Neo4j not connected"})

        with db.neo4j.driver.session() as session:
            res = session.run(
                """
                MATCH (c:Character)
                WHERE toLower(c.name) CONTAINS toLower($name)
                RETURN c.name AS name, c.character_id AS id,
                       c.character_type AS type,
                       c.personality_traits AS traits
                LIMIT 10
                """,
                name=name_query,
            )
            return json.dumps([dict(r) for r in res], ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"search_characters failed: {e}")
        return json.dumps({"error": str(e)})


async def upload_character(character_data: Dict[str, Any], db) -> str:
    """
    Create or update a Character node in Neo4j.

    Required keys:
        name (str)  — character name (used as unique key).
    Optional keys:
        character_id, character_type, aliases (List[str]),
        personality_traits (List[str]), background.

    Returns:
        JSON status string.
    """
    logger.info(f"upload_character(name={character_data.get('name')!r})")
    try:
        if not db.neo4j or not db.neo4j.driver:
            return json.dumps({"error": "Neo4j not connected"})

        if "name" not in character_data:
            return json.dumps({"error": "Character 'name' is required"})

        character_data.setdefault(
            "character_id",
            f"char_{abs(hash(character_data['name'])) % 1_000_000}",
        )

        with db.neo4j.driver.session() as session:
            session.run(
                "MERGE (c:Character {name: $props.name}) SET c += $props",
                props=character_data,
            )

        return json.dumps(
            {"status": "success", "character_id": character_data["character_id"]},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"upload_character failed: {e}")
        return json.dumps({"error": str(e)})
