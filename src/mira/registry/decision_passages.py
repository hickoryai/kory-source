"""Cheap, high-recall passage nomination for downstream decision review.

This module deliberately does not decide whether a passage records a decision.
It selects source-faithful passages which a later classifier can evaluate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_PASSAGE_CHARS = 6000

_CUE = re.compile(
    r"\b(?:because|instead\s+of|we\s+(?:chose|choose|use|keep)|do\s+not|never|always|"
    r"must|trade[ -]?off|rationale|why|decided|ratified|approved\s+by|decision)\b",
    re.IGNORECASE,
)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(?P<title>.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s{0,3}(?:```|~~~)")
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


@dataclass(frozen=True)
class Passage:
    """An exact source slice nominated for later classification."""

    start: int
    end: int
    text: str
    section: str = ""


def nominate(
    text: str,
    *,
    markdown: bool = False,
    force: bool = False,
    section: str = "",
    offset: int = 0,
) -> list[Passage]:
    """Nominate exact, bounded passages without determining their meaning.

    Plain text requires a lightweight decision/rationale cue unless ``force`` is
    set. Markdown is treated as structured source: every nonempty paragraph in
    every section is nominated, with its current heading retained as metadata.
    Offsets are translated by ``offset`` so callers can pass a source slice.
    """
    if not text:
        return []

    blocks = _markdown_blocks(text, section) if markdown else _plain_blocks(text, section)
    passages: list[Passage] = []
    seen: set[tuple[int, int]] = set()
    for start, end, block_section in blocks:
        block = text[start:end]
        if not force and not markdown and _CUE.search(block) is None:
            continue
        for chunk_start, chunk_end in _chunks(text, start, end):
            absolute_start = offset + chunk_start
            absolute_end = offset + chunk_end
            key = (absolute_start, absolute_end)
            if key in seen:
                continue
            seen.add(key)
            passages.append(
                Passage(absolute_start, absolute_end, text[chunk_start:chunk_end], block_section)
            )
    return passages


def _plain_blocks(text: str, section: str) -> list[tuple[int, int, str]]:
    return [(start, end, section) for start, end in _paragraph_ranges(text)]


def _markdown_blocks(text: str, initial_section: str) -> list[tuple[int, int, str]]:
    """Return paragraph ranges outside fenced code, carrying their heading."""
    blocks: list[tuple[int, int, str]] = []
    current_section = initial_section
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    fenced = False
    position = 0

    def flush() -> None:
        nonlocal paragraph_start, paragraph_end
        if paragraph_start is not None and paragraph_end is not None:
            blocks.append((paragraph_start, paragraph_end, current_section))
        paragraph_start = None
        paragraph_end = None

    for line in text.splitlines(keepends=True):
        line_end = position + len(line)
        body = line.rstrip("\r\n")
        if _FENCE.match(body):
            flush()
            fenced = not fenced
        elif fenced:
            pass
        else:
            heading = _HEADING.match(body)
            if heading:
                flush()
                current_section = heading.group("title").strip()
            elif body.strip():
                if re.match(r"\s*(?:[-*+] |\d+[.)] )", body):
                    flush()
                content_start = position + len(body) - len(body.lstrip())
                content_end = position + len(body.rstrip())
                if paragraph_start is None:
                    paragraph_start = content_start
                paragraph_end = content_end
            else:
                flush()
        position = line_end
    flush()
    return blocks


def _paragraph_ranges(text: str) -> list[tuple[int, int]]:
    """Return nonblank paragraphs as exact, trimmed source ranges."""
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    end: int | None = None
    position = 0
    for line in text.splitlines(keepends=True):
        line_end = position + len(line)
        body = line.rstrip("\r\n")
        if body.strip():
            content_start = position + len(body) - len(body.lstrip())
            content_end = position + len(body.rstrip())
            if start is None:
                start = content_start
            end = content_end
        elif start is not None and end is not None:
            ranges.append((start, end))
            start = None
            end = None
        position = line_end
    if start is not None and end is not None:
        ranges.append((start, end))
    return ranges


def _chunks(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Split a source range at sentence boundaries, then whitespace if needed."""
    chunks: list[tuple[int, int]] = []
    cursor = start
    while end - cursor > MAX_PASSAGE_CHARS:
        limit = cursor + MAX_PASSAGE_CHARS
        window = text[cursor:limit]
        sentence_ends = [match.end() for match in _SENTENCE_END.finditer(window)]
        split = cursor + sentence_ends[-1] if sentence_ends else -1
        if split <= cursor:
            whitespace = max(window.rfind(" "), window.rfind("\t"), window.rfind("\n"))
            split = cursor + whitespace if whitespace > 0 else limit
        chunks.append((cursor, split))
        cursor = split
        while cursor < end and text[cursor].isspace():
            cursor += 1
    if cursor < end:
        chunks.append((cursor, end))
    return chunks
