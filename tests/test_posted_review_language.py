"""Posted review text is instructions for the fixing agent, under the Kory Source name."""

from __future__ import annotations

import re

from mira.config import MiraConfig
from mira.llm.prompts.review import build_review_prompt
from mira.models import FileChangeType, FileDiff, HunkInfo, KeyIssue, ReviewComment, Severity
from mira.providers.formatting import (
    REVIEW_SUMMARY_HEADING,
    format_comment_body,
    format_key_issues,
    is_review_summary_body,
)


def test_summary_and_inline_comment_do_not_say_mira() -> None:
    comment = ReviewComment(
        path="src/app.py",
        line=12,
        end_line=14,
        severity=Severity.WARNING,
        category="bug",
        title="Reject a null id",
        body=(
            "HUMAN ATTENTION: ask which status code to return.\n\n"
            "Return 400 when the id is null.\n"
            "Expected: the handler rejects a null id.\n"
            "Verify: call the handler with id=None."
        ),
        confidence=0.9,
        suggestion="if user_id is None:\n    raise ValueError('id')",
        agent_prompt="In src/app.py lines 12-14, reject a null id before the query.",
    )
    inline = format_comment_body(comment)
    summary = (
        f"{REVIEW_SUMMARY_HEADING}\n\n"
        "Fix src/app.py:12. Reject a null id before the query.\n"
        + format_key_issues(
            [
                KeyIssue(
                    issue="HUMAN ATTENTION: ask which status code to return",
                    path="src/app.py",
                    line=12,
                )
            ]
        )
    )
    for text in (inline, summary):
        # The posting login stays `miracodeai`. The reviewer name must not.
        scrubbed = re.sub(r"miracodeai", "", text, flags=re.IGNORECASE)
        assert "mira" not in scrubbed.lower()
    assert inline.startswith("HUMAN ATTENTION:")
    assert "Fix `src/app.py:12-14`." in inline
    assert "Expected: the handler rejects a null id." in inline
    assert "reply `@miracodeai reject` on this comment." in inline
    assert is_review_summary_body(summary)
    assert is_review_summary_body("**Mira Review Summary**\n\nOlder review.")

    files = [
        FileDiff(
            path="src/app.py",
            change_type=FileChangeType.MODIFIED,
            hunks=[HunkInfo(1, 5, 1, 5, "content")],
            language="python",
            added_lines=1,
            deleted_lines=1,
        )
    ]
    system = build_review_prompt(files, MiraConfig())[0]["content"]
    assert "You are Kory Source" in system
    assert "HUMAN ATTENTION:" in system
    assert "Mira" not in system
