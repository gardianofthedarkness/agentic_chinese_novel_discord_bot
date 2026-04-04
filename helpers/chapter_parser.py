"""
Chapter Parser
==============
Splits raw novel text into a hierarchical volume → chapter structure.

The original HierarchicalChapterParser is iCloud-evicted; this is a clean
reimplementation that provides the same interface.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChapterNode:
    chapter_id: str
    title: str
    content: str
    volume_id: int
    chapter_index: int   # 0-based within volume
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.content)


@dataclass
class VolumeNode:
    volume_id: int
    title: str
    chapters: List[ChapterNode] = field(default_factory=list)

    @property
    def chapter_count(self) -> int:
        return len(self.chapters)

    @property
    def total_words(self) -> int:
        return sum(c.word_count for c in self.chapters)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Common chapter heading patterns (Chinese + English)
_CHAPTER_PATTERNS = [
    r"^第\s*[零一二三四五六七八九十百千\d]+\s*章",   # 第X章
    r"^Chapter\s+\d+",                                # Chapter N
    r"^\d+\s*[\.、]\s*\S",                            # 1. Title / 1、Title
    r"^【.+?】",                                       # 【Chapter Title】
]

_VOLUME_PATTERNS = [
    r"^第\s*[零一二三四五六七八九十百千\d]+\s*卷",   # 第X卷
    r"^Volume\s+\d+",
    r"^Vol\.?\s*\d+",
]

_CHAPTER_RE = re.compile("|".join(_CHAPTER_PATTERNS), re.MULTILINE)
_VOLUME_RE = re.compile("|".join(_VOLUME_PATTERNS), re.MULTILINE)


class ChapterParser:
    """
    Split raw novel text into volume → chapter hierarchy.

    Usage::

        parser = ChapterParser()
        structure = parser.parse(raw_text)
        # structure: Dict[int, VolumeNode]

        flat = parser.flat_chapters(structure)
        # flat: List[ChapterNode]  (sorted by volume then chapter)
    """

    def parse(self, text: str, default_volume_title: str = "卷一") -> Dict[int, VolumeNode]:
        """
        Parse raw text into volumes and chapters.

        Returns a dict keyed by volume_id (1-based).
        """
        lines = text.splitlines(keepends=True)
        segments = self._segment(lines)
        return self._build_hierarchy(segments, default_volume_title)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _segment(self, lines: List[str]) -> List[Dict]:
        """
        Identify volume/chapter boundaries.

        Returns list of dicts:
            {"kind": "volume"|"chapter"|"content", "text": str, "line_no": int}
        """
        segments = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if _VOLUME_RE.match(stripped):
                segments.append({"kind": "volume", "text": stripped, "line_no": i})
            elif _CHAPTER_RE.match(stripped):
                segments.append({"kind": "chapter", "text": stripped, "line_no": i})
            else:
                segments.append({"kind": "content", "text": line, "line_no": i})
        return segments

    def _build_hierarchy(self, segments: List[Dict], default_volume_title: str) -> Dict[int, VolumeNode]:
        volumes: Dict[int, VolumeNode] = {}
        current_vol_id = 1
        current_vol_title = default_volume_title
        current_chap_idx = 0
        current_chap_title: Optional[str] = None
        current_chap_lines: List[str] = []

        def _flush_chapter():
            nonlocal current_chap_idx, current_chap_lines, current_chap_title
            if current_chap_title is None and not current_chap_lines:
                return
            title = current_chap_title or f"第{current_chap_idx + 1}章"
            content = "".join(current_chap_lines).strip()
            if content:
                if current_vol_id not in volumes:
                    volumes[current_vol_id] = VolumeNode(current_vol_id, current_vol_title)
                chap = ChapterNode(
                    chapter_id=f"v{current_vol_id}_c{current_chap_idx}",
                    title=title,
                    content=content,
                    volume_id=current_vol_id,
                    chapter_index=current_chap_idx,
                )
                volumes[current_vol_id].chapters.append(chap)
            current_chap_title = None
            current_chap_lines = []
            current_chap_idx += 1

        def _flush_volume():
            nonlocal current_vol_id, current_vol_title, current_chap_idx
            _flush_chapter()
            current_chap_idx = 0

        for seg in segments:
            if seg["kind"] == "volume":
                _flush_volume()
                current_vol_id += 1
                current_vol_title = seg["text"]
            elif seg["kind"] == "chapter":
                _flush_chapter()
                current_chap_title = seg["text"]
            else:
                current_chap_lines.append(seg["text"])

        _flush_chapter()

        # Edge case: no volume markers → put everything under vol 1
        if not volumes and current_chap_lines:
            volumes[1] = VolumeNode(1, default_volume_title)

        return volumes

    @staticmethod
    def flat_chapters(structure: Dict[int, VolumeNode]) -> List[ChapterNode]:
        """Return all chapters as a flat list sorted by volume then chapter index."""
        result: List[ChapterNode] = []
        for vol_id in sorted(structure):
            result.extend(structure[vol_id].chapters)
        return result
