"""Whitespace compaction for source excerpts (shared by graph + ast_index)."""

from __future__ import annotations

import re


def minify_source_line(line: str) -> str:
    """Strip leading/trailing whitespace (indent gone); collapse any run of whitespace inside to one space."""
    s = line.strip()
    if not s:
        return ""
    return re.sub(r"\s+", " ", s)


def join_numbered_minified(pairs: list[tuple[int, str]]) -> str:
    """One physical line: `NNNN|core` fragments separated by a single space (no newlines)."""
    parts = [f"{num}|{core}" for num, core in pairs if core]
    return " ".join(parts)
