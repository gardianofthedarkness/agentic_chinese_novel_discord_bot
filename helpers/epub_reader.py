"""
EPUB Reader
===========
Extracts plain text from .epub files using only stdlib.
"""

import os
import re
import zipfile
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from typing import Optional


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self._parts.append(text)

    def get_text(self) -> str:
        return "\n".join(self._parts)


_NS = {
    "n": "urn:oasis:names:tc:opendocument:xmlns:container",
    "pkg": "http://www.idpf.org/2007/opf",
    "ncx": "http://www.daisy.org/z3986/2005/ncx/",
}


class EpubReader:
    """
    Read an EPUB file and extract its text content in reading order.

    Usage::

        reader = EpubReader("novel.epub")
        text = reader.get_content()
        reader.close()
    """

    def __init__(self, epub_path: str):
        if not os.path.exists(epub_path):
            raise FileNotFoundError(f"EPUB not found: {epub_path}")
        self._path = epub_path
        self._zf = zipfile.ZipFile(epub_path, "r")

    def close(self) -> None:
        self._zf.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_xml(self, path: str) -> ET.Element:
        with self._zf.open(path) as f:
            return ET.fromstring(f.read())

    def _html_to_text(self, html_bytes: bytes) -> str:
        parser = _HTMLTextExtractor()
        parser.feed(html_bytes.decode("utf-8", errors="replace"))
        return parser.get_text()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_content(self) -> str:
        """Return full novel text extracted in spine order."""
        # 1. Locate OPF via container.xml
        container = self._read_xml("META-INF/container.xml")
        root_file = container.find(".//n:rootfile", _NS)
        opf_path = root_file.attrib["full-path"]
        opf_dir = os.path.dirname(opf_path)

        # 2. Parse OPF manifest + spine
        opf = self._read_xml(opf_path)
        manifest: dict[str, str] = {}
        for item in opf.findall(".//pkg:manifest/pkg:item", _NS):
            manifest[item.attrib["id"]] = item.attrib["href"]

        spine_items = [
            item.attrib["idref"]
            for item in opf.findall(".//pkg:spine/pkg:itemref", _NS)
        ]

        # 3. Extract text from each spine item
        parts: list[str] = []
        for ref in spine_items:
            href = manifest.get(ref)
            if not href:
                continue
            full_path = os.path.join(opf_dir, href).replace("\\", "/")
            try:
                with self._zf.open(full_path) as f:
                    raw = f.read()
                parts.append(self._html_to_text(raw))
            except KeyError:
                continue

        return "\n\n".join(parts)

    def get_metadata(self) -> dict:
        """Return basic metadata (title, author, language)."""
        try:
            container = self._read_xml("META-INF/container.xml")
            root_file = container.find(".//n:rootfile", _NS)
            opf = self._read_xml(root_file.attrib["full-path"])

            def _find(tag: str) -> Optional[str]:
                el = opf.find(f".//{{{_NS['pkg']}}}{tag}")
                return el.text if el is not None else None

            return {
                "title": _find("title"),
                "author": _find("creator"),
                "language": _find("language"),
            }
        except Exception:
            return {}


def extract_volume_number(filename: str) -> Optional[int]:
    """Heuristic: extract volume number from an EPUB filename."""
    for pattern in [
        r"(?:Volume|Vol\.?|v|卷)\s*(\d+)",
        r"^(\d+)",
        r"(\d+)",
    ]:
        m = re.search(pattern, filename, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None
