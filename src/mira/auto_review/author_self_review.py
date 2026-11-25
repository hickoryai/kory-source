"""Author self-review carried on a pull request.

The posted shape, in the description and in a comment, is::

    **Author self-review**

    Head `<40-character sha>`
    Role: self-review

The ``Head`` line is the recorded commit. A review includes the section only
when that sha is the commit under review. A missing section does not delay
the review: nothing here sleeps, retries, or waits.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from mira.providers.review_classification import normalize_login

logger = logging.getLogger(__name__)

HEADING = "**Author self-review**"
_HEAD_LINE = re.compile(r"^Head\s+`?([0-9a-fA-F]{40})`?\s*$")
_ROLE_LINE = re.compile(r"^Role: self-review\s*$")
_ATX_HEADING = re.compile(r"^#{1,6}\s+\S")
_FENCE = re.compile(r"^(`{3,}|~{3,})")
_SHA40 = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{40}(?![0-9a-fA-F])")
_SECONDARY_LABEL = "Kory Source (secondary reviewer)"


@dataclass(frozen=True)
class AuthorSelfReview:
    """One parsed section. ``text`` is the heading through the findings."""

    head_sha: str
    text: str


def format_author_self_review(head_sha: str, findings: str = "") -> str:
    """The posted shape. ``head_sha`` is the 40-character commit."""
    text = f"{HEADING}\n\nHead `{head_sha}`\nRole: self-review"
    body = findings.strip()
    if body:
        text += "\n\n" + body
    return text


def author_self_reviews(text: str) -> list[AuthorSelfReview]:
    """Every valid section in ``text``, in order."""
    if not text:
        return []
    lines = text.splitlines()
    found: list[AuthorSelfReview] = []
    index = 0
    while index < len(lines):
        if lines[index].strip() != HEADING:
            index += 1
            continue
        parsed = _parse_at(lines, index)
        if parsed is None:
            index += 1
            continue
        section, end = parsed
        found.append(section)
        index = end
    return found


def is_author_self_review(text: str) -> bool:
    """True when ``text`` is only an author self-review section.

    A review that quotes a section and also says something else is not one.
    """
    if not author_self_reviews(text):
        return False
    return strip_author_self_reviews(text) == ""


def section_for_head(text: str, head_sha: str) -> AuthorSelfReview | None:
    """The last section whose recorded head is ``head_sha``.

    A different head is not a match. The comparison is the full 40-character
    sha, ignoring hex case.
    """
    wanted = (head_sha or "").strip().lower()
    if len(wanted) != 40:
        return None
    match: AuthorSelfReview | None = None
    for section in author_self_reviews(text):
        if section.head_sha == wanted:
            match = section
    return match


def strip_author_self_reviews(text: str) -> str:
    """Remove every author self-review section from ``text``."""
    if not text:
        return ""
    lines = text.splitlines()
    drop: set[int] = set()
    index = 0
    while index < len(lines):
        if lines[index].strip() != HEADING:
            index += 1
            continue
        parsed = _parse_at(lines, index)
        if parsed is None:
            index += 1
            continue
        _section, end = parsed
        start = index
        drop.update(range(start, end))
        index = end
    if not drop:
        return text
    kept = [line for number, line in enumerate(lines) if number not in drop]
    cleaned = "\n".join(kept).strip()
    return re.sub(r"\n{3,}", "\n\n", cleaned)


def prepare_author_self_review(
    description: str,
    head_sha: str,
    *,
    explicit: str | None = None,
) -> tuple[str, str]:
    """Section to show, and the description with every section removed.

    ``explicit`` is the runner's choice, including an empty string when it
    looked and found no section for this head. ``None`` reads the description.
    """
    raw = description or ""
    cleaned = strip_author_self_reviews(raw)
    if explicit is not None:
        return explicit, cleaned
    section = section_for_head(raw, head_sha)
    return (section.text if section is not None else ""), cleaned


def is_prior_kory_source_review(body: str, author: str = "") -> bool:
    """Whether a posted body counts as a prior Kory Source review.

    Markers are read after every author self-review section is removed. A
    review counts when that remainder contains ``Reviewed head`` and a
    40-character sha and the author is ``kory-source`` or ``kory-source[bot]``,
    or when it contains ``Kory Source (secondary reviewer)``. A body that is
    only an author self-review never counts, including when its findings
    quote those markers.
    """
    text = strip_author_self_reviews(body or "")
    if not text:
        return False
    if _SECONDARY_LABEL in text:
        return True
    return (
        normalize_login(author) == "kory-source"
        and "Reviewed head" in text
        and _SHA40.search(text) is not None
    )


async def load_author_self_review(reader: Any, snapshot: Any) -> str:
    """Section for this snapshot's head, from the description or one comment read.

    The description wins when its recorded head matches. Otherwise one listing
    of issue comments is read. That route returns comments oldest first, so the
    last comment by the pull request author whose recorded head matches is the
    newest. A missing method, a failed read, or no match returns an empty
    string. The review still runs.
    """
    meta = snapshot.meta
    head = meta.head_sha or ""
    matched = section_for_head(meta.description or "", head)
    if matched is not None:
        return matched.text
    fetch = getattr(reader, "issue_comments", None)
    if fetch is None or not meta.number:
        return ""
    try:
        comments = await fetch(meta.number)
    except Exception:
        logger.info(
            "author self-review comments unavailable %s/%s#%s",
            meta.owner,
            meta.repo,
            meta.number,
        )
        return ""
    author = (meta.author or "").casefold()
    if not author:
        return ""
    found = ""
    for comment in comments or []:
        login = str(comment.get("login") or "").casefold()
        if login != author:
            continue
        section = section_for_head(str(comment.get("body") or ""), head)
        if section is not None:
            found = section.text
    return found


def _parse_at(lines: list[str], start: int) -> tuple[AuthorSelfReview, int] | None:
    """Parse one section starting at ``start``, or return None."""
    cursor = _skip_blank(lines, start + 1)
    if cursor >= len(lines):
        return None
    head = _HEAD_LINE.match(lines[cursor].strip())
    if head is None:
        return None
    cursor = _skip_blank(lines, cursor + 1)
    if cursor >= len(lines) or _ROLE_LINE.match(lines[cursor].strip()) is None:
        return None
    end = cursor + 1
    in_fence = False
    while end < len(lines):
        stripped = lines[end].strip()
        if _FENCE.match(stripped):
            in_fence = not in_fence
            end += 1
            continue
        if not in_fence and (stripped == HEADING or _ATX_HEADING.match(lines[end])):
            break
        end += 1
    body = lines[start:end]
    while body and body[-1].strip() == "":
        body.pop()
    section = AuthorSelfReview(head.group(1).lower(), "\n".join(body))
    return section, end


def _skip_blank(lines: list[str], index: int) -> int:
    while index < len(lines) and lines[index].strip() == "":
        index += 1
    return index
