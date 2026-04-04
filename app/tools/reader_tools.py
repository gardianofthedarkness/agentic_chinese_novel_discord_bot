"""
Reader Tools
============
Agent-callable tools for reading novel files and creating hierarchical events.
"""

import json
import logging
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reader state (one per agent instance)
# ---------------------------------------------------------------------------

class ReaderState:
    """Holds the cursor position for the agent's sequential novel reader."""

    def __init__(self):
        self.file_path: Optional[str] = None
        self.cursor: int = 0
        self.total_size: int = 0


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

async def initialize_reader(file_path: str, state: ReaderState) -> str:
    """
    Open a novel file (plain text or EPUB) and reset the read cursor.

    Args:
        file_path:  Absolute path to the file.
        state:      ReaderState object (mutated in-place).

    Returns:
        JSON status string.
    """
    logger.info(f"initialize_reader(path={file_path!r})")
    try:
        if not os.path.exists(file_path):
            return json.dumps({"error": f"File not found: {file_path}"})

        target = file_path

        # Convert EPUB to temp plain-text file
        if file_path.lower().endswith(".epub"):
            from helpers.epub_reader import EpubReader  # lazy import
            try:
                reader = EpubReader(file_path)
                content = reader.get_content()
                reader.close()
                base = os.path.splitext(os.path.basename(file_path))[0]
                target = os.path.join(tempfile.gettempdir(), f"{base}_converted.txt")
                with open(target, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.info(f"EPUB converted to: {target}")
            except Exception as e:
                return json.dumps({"error": f"EPUB conversion failed: {e}"})

        state.file_path = target
        state.cursor = 0
        state.total_size = os.path.getsize(target)

        return json.dumps(
            {
                "status": "success",
                "file": os.path.basename(file_path),
                "total_bytes": state.total_size,
            }
        )
    except Exception as e:
        logger.error(f"initialize_reader failed: {e}")
        return json.dumps({"error": str(e)})


async def read_novel_chunk(chunk_size: int = 2000, state: Optional[ReaderState] = None) -> str:
    """
    Read the next chunk of text from the novel file.

    Args:
        chunk_size:  Number of bytes to read.
        state:       ReaderState object.

    Returns:
        JSON with {"chunk_text", "progress_percent", "is_eof"}.
    """
    logger.info(f"read_novel_chunk(size={chunk_size})")
    if state is None or state.file_path is None:
        return json.dumps({"error": "Reader not initialised — call initialize_reader first."})
    try:
        with open(state.file_path, "r", encoding="utf-8") as f:
            f.seek(state.cursor)
            text = f.read(chunk_size)
            state.cursor = f.tell()

        progress = (state.cursor / state.total_size * 100) if state.total_size else 0.0

        return json.dumps(
            {
                "status": "success",
                "chunk_text": text,
                "cursor": state.cursor,
                "progress_percent": round(progress, 2),
                "is_eof": len(text) < chunk_size,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"read_novel_chunk failed: {e}")
        return json.dumps({"error": str(e)})


async def extract_gap_text(
    chapter_text: str,
    start_snippet: str,
    end_snippet: str,
    limit_chars: int = 1000,
) -> str:
    """
    Extract a passage between two anchor snippets in a chapter text.

    Useful for retrieving dialogue or descriptive passages skipped during
    event extraction.

    Args:
        chapter_text:   Full text of the chapter (string, not a file path).
        start_snippet:  First ~10 characters identifying the start position.
        end_snippet:    Last ~10 characters identifying the end position.
        limit_chars:    Safety truncation limit.

    Returns:
        JSON with {"content": "..."} or {"error": "..."}.
    """
    logger.info(f"extract_gap_text(start={start_snippet!r}, end={end_snippet!r})")
    try:
        start_idx = chapter_text.find(start_snippet)
        if start_idx == -1:
            return json.dumps({"error": "Start snippet not found"})

        end_idx = chapter_text.find(end_snippet, start_idx + len(start_snippet))
        if end_idx == -1:
            return json.dumps({"error": "End snippet not found after start"})

        extracted = chapter_text[start_idx : end_idx + len(end_snippet)]

        if len(extracted) > limit_chars:
            return json.dumps(
                {
                    "warning": f"Truncated ({len(extracted)} → {limit_chars} chars)",
                    "content": extracted[:limit_chars] + "...",
                },
                ensure_ascii=False,
            )

        return json.dumps({"status": "success", "content": extracted}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"extract_gap_text failed: {e}")
        return json.dumps({"error": str(e)})


async def create_meta_event(
    summary: str,
    event_type: str,
    child_event_ids: List[str],
    db,
) -> str:
    """
    Create a hierarchical meta-event that groups atomic events.

    Args:
        summary:         Description of the meta-event.
        event_type:      e.g. "arc", "battle_sequence", "romance_arc".
        child_event_ids: List of event_ids to attach via HAS_CHILD edges.
        db:              DatabaseAdapter instance.

    Returns:
        JSON with {"meta_event_id": "..."}.
    """
    logger.info(f"create_meta_event(type={event_type!r}, children={len(child_event_ids)})")
    try:
        if not db.neo4j or not db.neo4j.driver:
            return json.dumps({"error": "Neo4j not connected"})

        meta_id = f"meta_{uuid.uuid4().hex[:10]}"

        with db.neo4j.driver.session() as session:
            session.run(
                """
                MERGE (m:Event {event_id: $mid})
                SET m.description = $summary,
                    m.event_type   = $etype,
                    m.is_meta      = true,
                    m.created_at   = datetime()
                WITH m
                UNWIND $children AS child_id
                MATCH (c:Event {event_id: child_id})
                MERGE (m)-[:HAS_CHILD]->(c)
                """,
                mid=meta_id,
                summary=summary,
                etype=event_type,
                children=child_event_ids,
            )

        return json.dumps({"status": "success", "meta_event_id": meta_id}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"create_meta_event failed: {e}")
        return json.dumps({"error": str(e)})
