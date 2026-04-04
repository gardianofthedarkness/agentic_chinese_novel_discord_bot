"""
Novel Processing Skill
======================
High-level skill that ingests an entire novel (EPUB or directory of EPUBs)
by iterating chapters through the processing chain.
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NovelProcessingSkill:
    """
    Orchestrates end-to-end novel ingestion:
        1. Read EPUB(s) → chapter list
        2. Run each chapter through the processing chain (extract events,
           analyze characters, analyze causality, store to DB)
        3. Return a summary report
    """

    def __init__(self, llm, db):
        self.llm = llm
        self.db = db
        self._chain = None  # lazy-compiled

    def _get_chain(self):
        if self._chain is None:
            from app.chains.processing_chain import create_processing_chain
            self._chain = create_processing_chain(self.llm, self.db)
        return self._chain

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def process_epub(self, epub_path: str, volume_id: int = 1) -> Dict[str, Any]:
        """
        Process a single EPUB file.

        Args:
            epub_path:  Absolute path to the .epub file.
            volume_id:  Volume number to tag events/characters with.

        Returns:
            Summary dict with counts of stored entities.
        """
        from helpers.epub_reader import EpubReader
        from helpers.chapter_parser import ChapterParser

        logger.info(f"Processing EPUB: {epub_path} (volume {volume_id})")

        with EpubReader(epub_path) as reader:
            raw_text = reader.get_content()

        parser = ChapterParser()
        structure = parser.parse(raw_text)
        chapters = ChapterParser.flat_chapters(structure)

        return await self._process_chapters(chapters, volume_id)

    async def process_directory(self, dir_path: str) -> Dict[str, Any]:
        """
        Process all .epub files in a directory, auto-detecting volume numbers.

        Returns:
            Aggregated summary dict.
        """
        from helpers.epub_reader import extract_volume_number

        epub_files = sorted(
            f for f in os.listdir(dir_path) if f.lower().endswith(".epub")
        )
        if not epub_files:
            raise FileNotFoundError(f"No .epub files in {dir_path}")

        total: Dict[str, int] = {"volumes": 0, "chapters": 0, "events": 0, "characters": 0}

        for filename in epub_files:
            vol_id = extract_volume_number(filename) or (total["volumes"] + 1)
            path = os.path.join(dir_path, filename)
            result = await self.process_epub(path, volume_id=vol_id)
            total["volumes"] += 1
            total["chapters"] += result.get("chapters_processed", 0)
            total["events"] += result.get("events_stored", 0)
            total["characters"] += result.get("characters_stored", 0)

        return total

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _process_chapters(self, chapters: List[Any], volume_id: int) -> Dict[str, Any]:
        chain = self._get_chain()
        events_total = 0
        chars_total = 0

        for chapter in chapters:
            initial_state = {
                "chapter_text": chapter.content,
                "chapter_index": chapter.chapter_index,
                "volume_id": volume_id,
                "events": [],
                "characters": [],
                "causal_links": [],
                "errors": [],
            }
            try:
                final_state = await chain.ainvoke(initial_state)
                summary = final_state.get("store_summary", {})
                events_total += summary.get("events", 0)
                chars_total += summary.get("characters", 0)
                if final_state.get("errors"):
                    for err in final_state["errors"]:
                        logger.warning(f"Chapter {chapter.chapter_index} error: {err}")
            except Exception as e:
                logger.error(f"Chapter {chapter.chapter_index} failed: {e}")

        return {
            "chapters_processed": len(chapters),
            "events_stored": events_total,
            "characters_stored": chars_total,
            "volume_id": volume_id,
        }
