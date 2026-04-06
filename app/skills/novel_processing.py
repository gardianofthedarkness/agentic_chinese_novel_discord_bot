"""
Skill: novel_processing
=======================
Focus mode for ingesting novel files into the knowledge base.

Pipeline overview
-----------------
1. Read the epub/txt file with read_file
2. Parse chapters and extract events using the LLM
3. Store extracted Event nodes in Neo4j with upload_neo4j
4. Store Character nodes in Neo4j with upload_neo4j

Neo4j upload format
-------------------
Use upload_neo4j with label="Event" and these exact property names:
{
  "event_id":            "<unique string, e.g. vol1_ch3_evt2>",
  "volume_id":           1,
  "batch_id":            1,
  "chronological_order": 42,
  "description":         "萧炎在山洞中发现了一枚戒指",
  "event_type":          "discovery",
  "importance_score":    0.7,
  "primary_actors":      ["萧炎"],
  "affected_characters": [],
  "caused_by_events":    [],
  "causes_events":       [],
  "temporal_markers":    ["第三章"]
}

Use upload_neo4j with label="Character":
{
  "character_id":     "xiao_yan_vol1",
  "name":             "萧炎",
  "volume_id":        1,
  "batch_id":         1,
  "character_type":   "protagonist",
  "aliases":          ["小炎"],
  "personality_traits": ["倔强", "勤奋", "热血"],
  "first_appearance": 1
}

Processing rules
----------------
- Always check if the file exists with read_file before starting
- Process in batches of 5-10 chapters to avoid timeouts
- Use meaningful event_ids that encode volume/chapter/sequence
- Set importance_score based on plot significance (0.3 minor, 0.7 major, 1.0 climax)
- Extract at most 3-5 events per chapter — focus on significant plot points
- After ingesting, run a verification query with run_cypher:
  MATCH (n) RETURN labels(n)[0] as label, count(n) as count
"""

import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

TOOLS = ["read_file", "upload_neo4j", "run_cypher"]


class NovelProcessingSkill:
    name = "novel_processing"
    description = "Ingest novel files (epub/txt) into Neo4j knowledge graph"
    tools = TOOLS

    @classmethod
    def get_prompt(cls) -> str:
        return __doc__

    # -----------------------------------------------------------------------
    # Legacy programmatic interface (used by chains/processing_chain)
    # -----------------------------------------------------------------------

    def __init__(self, llm=None, db=None):
        self.llm = llm
        self.db = db
        self._chain = None

    def _get_chain(self):
        if self._chain is None:
            from app.chains.processing_chain import create_processing_chain
            self._chain = create_processing_chain(self.llm, self.db)
        return self._chain

    async def process_epub(self, epub_path: str, volume_id: int = 1) -> Dict[str, Any]:
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
        from helpers.epub_reader import extract_volume_number
        epub_files = sorted(f for f in os.listdir(dir_path) if f.lower().endswith(".epub"))
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

    async def _process_chapters(self, chapters: List[Any], volume_id: int) -> Dict[str, Any]:
        chain = self._get_chain()
        events_total = 0
        chars_total = 0
        for chapter in chapters:
            initial_state = {
                "chapter_text": chapter.content,
                "chapter_index": chapter.chapter_index,
                "volume_id": volume_id,
                "events": [], "characters": [], "causal_links": [], "errors": [],
            }
            try:
                final_state = await chain.ainvoke(initial_state)
                summary = final_state.get("store_summary", {})
                events_total += summary.get("events", 0)
                chars_total += summary.get("characters", 0)
            except Exception as e:
                logger.error(f"Chapter {chapter.chapter_index} failed: {e}")
        return {
            "chapters_processed": len(chapters),
            "events_stored": events_total,
            "characters_stored": chars_total,
            "volume_id": volume_id,
        }
