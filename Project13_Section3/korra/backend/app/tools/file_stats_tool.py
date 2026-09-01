from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import tool


@tool
def analyze_file_statistics(file_path: str) -> str:
    """Analyze a local text file and return simple statistics.

    Use this when the user asks for line count, word count, character count,
    or a brief file summary for a local file path.
    """
    path = Path(file_path)
    if not path.exists():
        return f"File not found: {file_path}"
    if not path.is_file():
        return f"Path is not a file: {file_path}"

    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    words = text.split()
    chars = len(text)

    return (
        f"File: {path.name}\n"
        f"Lines: {len(lines)}\n"
        f"Words: {len(words)}\n"
        f"Characters: {chars}"
    )
