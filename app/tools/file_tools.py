"""
File Tools
==========
Read and grep local files — useful for novel analysis and code inspection.

Security note: paths are restricted to the project root (PROJECT_ROOT).
Any attempt to read outside that directory returns an error.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from pydantic import Field

from app.core.tool_base import BaseTool, ToolInput
from app.core.tool_factory import ToolFactory

# Restrict all file access to the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _safe_path(raw: str) -> Path | None:
    """Return resolved Path if it is inside PROJECT_ROOT, else None."""
    try:
        resolved = (PROJECT_ROOT / raw).resolve()
        resolved.relative_to(PROJECT_ROOT)  # raises ValueError if outside
        return resolved
    except (ValueError, Exception):
        return None


# ---------------------------------------------------------------------------
# ReadFileTool
# ---------------------------------------------------------------------------

class ReadFileInput(ToolInput):
    path: str = Field(..., description="Relative path to the file (from project root).")
    start_line: int = Field(1, description="First line to read (1-indexed).")
    end_line: int = Field(200, description="Last line to read (inclusive).")


@ToolFactory.register
class ReadFileTool(BaseTool):
    """Read a local file and return its contents (or a slice of it)."""

    name = "read_file"
    description = "Read a local file by path. Optionally specify start_line and end_line."
    input_cls = ReadFileInput

    def __init__(self, **deps: Any) -> None:
        pass

    async def execute(self, inputs: dict) -> str:
        path = _safe_path(inputs["path"])
        if path is None:
            return json.dumps({"error": "Path is outside project root or invalid."})
        if not path.exists():
            return json.dumps({"error": f"File not found: {inputs['path']}"})
        if not path.is_file():
            return json.dumps({"error": f"Not a file: {inputs['path']}"})

        start = max(1, inputs["start_line"])
        end = inputs["end_line"]

        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        selected = lines[start - 1 : end]
        return "\n".join(f"{start + i:>4}  {line}" for i, line in enumerate(selected))


# ---------------------------------------------------------------------------
# GrepFileTool
# ---------------------------------------------------------------------------

class GrepFileInput(ToolInput):
    pattern: str = Field(..., description="Regex pattern to search for.")
    path: str = Field(".", description="Relative path to a file or directory to search in.")
    context_lines: int = Field(2, description="Number of context lines to show around each match.")


@ToolFactory.register
class GrepFileTool(BaseTool):
    """Search for a regex pattern in local files using ripgrep (falls back to grep)."""

    name = "grep_file"
    description = "Search files for a regex pattern. Returns matching lines with context."
    input_cls = GrepFileInput

    def __init__(self, **deps: Any) -> None:
        pass

    async def execute(self, inputs: dict) -> str:
        search_path = _safe_path(inputs["path"])
        if search_path is None:
            return json.dumps({"error": "Path is outside project root or invalid."})

        pattern = inputs["pattern"]
        ctx = inputs["context_lines"]

        # Try ripgrep first, fall back to grep
        for cmd in [
            ["rg", "--no-heading", "-n", f"-C{ctx}", pattern, str(search_path)],
            ["grep", "-rn", f"--context={ctx}", pattern, str(search_path)],
        ]:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(PROJECT_ROOT),
                )
                if result.returncode in (0, 1):  # 1 = no matches (not an error)
                    output = result.stdout.strip()
                    if not output:
                        return "No matches found."
                    # Truncate to avoid flooding the context window
                    lines = output.splitlines()
                    if len(lines) > 100:
                        lines = lines[:100] + [f"... ({len(lines) - 100} more lines truncated)"]
                    return "\n".join(lines)
            except FileNotFoundError:
                continue  # try next command
            except subprocess.TimeoutExpired:
                return json.dumps({"error": "grep timed out"})

        return json.dumps({"error": "Neither rg nor grep is available."})
