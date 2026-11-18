"""Carry unmodified findings between immutable pull-request snapshots."""

from __future__ import annotations

from dataclasses import replace

import unidiff

from mira.models import ReviewResult


def carry_findings(previous: ReviewResult, diff_text: str, paths: set[str]) -> ReviewResult:
    """Drop changed lines and remap surviving ranges through the commit delta."""
    patch = unidiff.PatchSet(diff_text)
    changed = {f.source_file.removeprefix("a/"): f for f in patch}

    def location(path: str, start: int, end: int) -> tuple[str, int, int] | None:
        file = changed.get(path)
        target = file.path if file is not None else path
        if target not in paths or (file is not None and file.is_removed_file):
            return None
        if file is None:
            return target, start, end
        if file.is_binary_file:
            return None

        def line_number(line: int) -> int | None:
            offset = 0
            for hunk in file:
                if not hunk.source_length:
                    if line > hunk.source_start:
                        offset += hunk.target_length
                    continue
                if line < hunk.source_start:
                    break
                if line >= hunk.source_start + hunk.source_length:
                    offset += hunk.target_length - hunk.source_length
                    continue
                for entry in hunk:
                    if entry.source_line_no == line:
                        return None if entry.is_removed else entry.target_line_no
            return line + offset

        mapped = [line_number(line) for line in range(start, end + 1)]
        if any(line is None for line in mapped):
            return None
        first, last = mapped[0], mapped[-1]
        assert first is not None and last is not None
        if last - first != end - start:
            return None
        return target, first, last

    comments = []
    for comment in previous.comments:
        mapped = location(comment.path, comment.line, comment.end_line or comment.line)
        if mapped is not None:
            path, start, end = mapped
            comments.append(
                replace(
                    comment,
                    path=path,
                    line=start,
                    end_line=end if comment.end_line is not None else None,
                )
            )
    issues = []
    for issue in previous.key_issues:
        mapped = location(issue.path, issue.line, issue.line)
        if mapped is not None:
            issues.append(replace(issue, path=mapped[0], line=mapped[1]))
    return replace(previous, comments=comments, key_issues=issues)
