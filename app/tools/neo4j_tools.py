"""
Neo4j Tools
===========
Two tools for graph database interaction:

1. UploadNeo4jTool  — Upload a JSON event/entity payload as a Neo4j node.
2. RunCypherTool    — Execute an arbitrary (read-only by default) Cypher query.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import Field

from app.core.tool_base import BaseTool, ToolInput
from app.core.tool_factory import ToolFactory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UploadNeo4jTool
# ---------------------------------------------------------------------------

class UploadNeo4jInput(ToolInput):
    label: str = Field(..., description="Neo4j node label (e.g. 'Event', 'Character').")
    properties: dict = Field(..., description="Dict of property key/value pairs for the node.")


@ToolFactory.register
class UploadNeo4jTool(BaseTool):
    """Create or merge a node in Neo4j with the given label and properties."""

    name = "upload_neo4j"
    description = "Upload a node (event, character, etc.) to the Neo4j graph database."
    input_cls = UploadNeo4jInput

    def __init__(self, **deps: Any) -> None:
        self._db = deps.get("db")

    async def execute(self, inputs: dict) -> str:
        if self._db is None:
            return json.dumps({"error": "No database adapter available."})

        label = inputs["label"]
        props = inputs["properties"]

        # Build a MERGE query using the 'id' property if present, else CREATE
        if "id" in props:
            cypher = (
                f"MERGE (n:{label} {{id: $id}}) "
                "SET n += $props "
                "RETURN id(n) AS internal_id"
            )
            params = {"id": props["id"], "props": props}
        else:
            cypher = f"CREATE (n:{label} $props) RETURN id(n) AS internal_id"
            params = {"props": props}

        try:
            result = await self._db.run_cypher(cypher, params)
            return json.dumps({"status": "ok", "result": result})
        except Exception as exc:
            logger.error(f"UploadNeo4jTool error: {exc}")
            return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# RunCypherTool
# ---------------------------------------------------------------------------

class RunCypherInput(ToolInput):
    query: str = Field(..., description="Cypher query to execute.")
    params: dict = Field(default_factory=dict, description="Query parameters dict.")
    read_only: bool = Field(True, description="If true, only SELECT-like queries are allowed (no MERGE/CREATE/DELETE).")


_WRITE_KEYWORDS = {"CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DETACH"}


@ToolFactory.register
class RunCypherTool(BaseTool):
    """Execute a Cypher query against Neo4j and return the results as JSON."""

    name = "run_cypher"
    description = "Run a Cypher query on Neo4j. Set read_only=false to allow write operations."
    input_cls = RunCypherInput

    def __init__(self, **deps: Any) -> None:
        self._db = deps.get("db")

    async def execute(self, inputs: dict) -> str:
        if self._db is None:
            return json.dumps({"error": "No database adapter available."})

        query = inputs["query"]
        params = inputs.get("params", {})
        read_only = inputs.get("read_only", True)

        if read_only:
            upper = query.upper()
            for kw in _WRITE_KEYWORDS:
                if kw in upper:
                    return json.dumps({
                        "error": f"Query contains write keyword '{kw}' but read_only=true."
                    })

        try:
            result = await self._db.run_cypher(query, params)
            return json.dumps({"status": "ok", "result": result})
        except Exception as exc:
            logger.error(f"RunCypherTool error: {exc}")
            return json.dumps({"error": str(exc)})
