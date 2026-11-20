"""Prompt builder for verifying whether review issues have been fixed."""

from __future__ import annotations

from mira.llm.utils import strip_code_fences, strip_think_blocks
from mira.models import UnresolvedThread

# Markers that signal the start of noise sections in formatted review comments.
# Everything from these markers onward is stripped before inclusion in prompts.
_BODY_NOISE_MARKERS = ("**Suggested fix:**", "```suggestion", "<details>")

_MAX_DESCRIPTION_LENGTH = 300


def _extract_issue_description(body: str) -> str:
    """Extract the core issue description from a formatted review comment body.

    Current comments, and comments posted before the wording change, look like::

        HUMAN ATTENTION: ...          (optional; kept)
        **{category_label}**          (older comments led with an emoji)
        {severity_badge}

        **{title}**

        Fix `path:line`.
        {instructions}

        **Suggested fix:**            (older comments)
        ```suggestion ...```

        <details>Fix instructions ...</details>

    This function strips the badge header, suggestion blocks, and the fenced
    fix-instruction details, and keeps a leading ``HUMAN ATTENTION:`` line.
    """
    text = body

    # Cut off suggestion blocks and agent prompt sections
    for marker in _BODY_NOISE_MARKERS:
        pos = text.find(marker)
        if pos != -1:
            text = text[:pos]

    # Strip markdown bold markers
    text = text.replace("**", "")

    # Split into paragraphs (double-newline separated)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Skip the compact badge paragraph when more content follows. A leading
    # HUMAN ATTENTION line is the instruction, so skip the badge after it.
    idx = 0
    if paragraphs and paragraphs[0].startswith("HUMAN ATTENTION:"):
        idx = 1
    if len(paragraphs) > idx + 1 and len(paragraphs[idx]) < 80:
        paragraphs = paragraphs[:idx] + paragraphs[idx + 1 :]

    result = " ".join(paragraphs).strip()
    if not result:
        # Fallback: use the cleaned full text
        result = " ".join(text.split()).strip()

    if len(result) > _MAX_DESCRIPTION_LENGTH:
        result = result[:_MAX_DESCRIPTION_LENGTH].rsplit(" ", 1)[0] + "…"
    return result


def build_verify_fixes_prompt(
    file_groups: list[tuple[str, str, list[UnresolvedThread]]],
) -> list[dict[str, str]]:
    """Build a prompt asking the LLM which review issues have been fixed.

    Each entry in *file_groups* is a ``(path, file_content, threads)`` tuple
    where *file_content* is the current code (full file or relevant sections)
    and *threads* lists the unresolved review comments in that file.
    """
    sections: list[str] = []
    for path, content, threads in file_groups:
        issue_lines: list[str] = []
        for idx, t in enumerate(threads, 1):
            line_label = f"Line {t.line}" if t.line > 0 else "Location unknown (outdated comment)"
            outdated_tag = " [OUTDATED — code has changed]" if t.is_outdated else ""
            issue_lines.append(
                f'{idx}. (id: "{t.thread_id}") {line_label}{outdated_tag}: '
                f"{_extract_issue_description(t.body)}"
            )
        issues = "\n".join(issue_lines)
        sections.append(
            f"File: {path}\n```\n{content}\n```\n\nIssues to verify in this file:\n{issues}"
        )

    user_content = "\n\n---\n\n".join(sections)

    return [
        {
            "role": "system",
            "content": (
                "You are verifying whether code review issues have been fixed.\n\n"
                "For each issue below, you will see the current file content "
                "(full or relevant sections) with line numbers, and a list of "
                "previously flagged issues.\n\n"
                "For each issue:\n"
                "1. Look at the referenced line number in the current code.\n"
                "2. Check if the EXACT problematic code pattern described in "
                "the issue is still present at or near that line.\n"
                "3. Mark as fixed (true) if ANY of these apply:\n"
                "   - The problematic pattern has been removed or replaced.\n"
                "   - The issue description does not match the actual code "
                "(i.e. the issue was wrong or outdated).\n"
                "   - The code at the referenced location has changed such "
                "that the concern no longer applies.\n"
                "4. Mark as not fixed (false) ONLY if the exact problematic "
                "pattern described in the issue is still clearly present.\n\n"
                "Issues tagged [OUTDATED] have been flagged by GitHub as having "
                "changed code around them — these are very likely fixed.\n\n"
                "Respond with ONLY the JSON object below, no other text:\n"
                '{"results": [{"id": "<thread_id>", "fixed": true/false}, ...]}'
            ),
        },
        {"role": "user", "content": user_content},
    ]


def parse_verify_fixes_response(raw: str) -> list[str]:
    """Parse the LLM response and return thread IDs confirmed as fixed."""
    import json

    try:
        data = json.loads(strip_think_blocks(strip_code_fences(raw)))
    except (json.JSONDecodeError, TypeError):
        return []

    results = data.get("results")
    if not isinstance(results, list):
        return []

    fixed_ids: list[str] = []
    for entry in results:
        if not isinstance(entry, dict):
            continue
        if entry.get("fixed") is True and isinstance(entry.get("id"), str):
            fixed_ids.append(entry["id"])
    return fixed_ids
