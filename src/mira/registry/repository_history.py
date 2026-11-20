"""Read-only git history import: prose, data rationales, comments and commit messages.

From a local checkout of a manifest repository, at a frozen commit:

- Markdown, data, SQL and template revisions along their history (bounded by
  ``max_commits``), so a passage that was later rewritten or deleted is kept;
- comments and docstrings in supported languages at the frozen commit,
  anchored to the enclosing symbol from the deterministic extractor;
- commit messages, whose trailers can state an agent co-author, model, or
  invoker.

Git author and committer labels are stored as source accounts. Decision authors
fall back to the line's Git author as an explicitly inferred identity, never an
approver or an invoking person.

``GitReader`` runs a fixed allowlist of read-only git subcommands. Nothing
here writes to the repository.

Run: ``python -m mira.registry.repository_history import --repo owner/name
--checkout PATH [--commit SHA] [--apply]`` with ``REGISTRY_IMPORT_OPERATOR``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
import subprocess
import sys
import time
from bisect import bisect_right
from collections import Counter, OrderedDict
from dataclasses import replace
from itertools import chain
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml

from mira.registry import attribution as attr
from mira.registry import decision_llm
from mira.registry.access import AccessContext, AccessDenied, require_import
from mira.registry.decision_passages import Passage, nominate
from mira.registry.imports import (
    Candidate,
    ImportRuns,
    ReadOnlyViolation,
    extract_decisions,
    materialize,
    read_cursors,
    record_cursor,
)
from mira.registry.provenance import ProvenanceStore, SourceRevision

logger = logging.getLogger(__name__)

_ERROR_PREVIEW_COUNT = 20
_ERROR_PREVIEW_CHARS = 300


def _record_error(errors: list[str], message: str) -> None:
    """Append the full error and log it. Stdout later shows only a short preview."""
    text = str(message)
    errors.append(text)
    logger.warning("%s", text)


def _error_preview(errors: list[str]) -> dict[str, Any]:
    def kind(message):
        if message.startswith("retryable failures:"):
            return "retryable_stop"
        if "workstation_outage" in message or "Router workstation outage" in message:
            return "workstation_outage"
        if "no_worker_before_deadline" in message:
            return "no_worker_before_deadline"
        if "time budget reached" in message:
            return "time_budget"
        if "git " in message and " failed:" in message:
            return "git_failure"
        # A later batch prefixes the unit key. The group text still names the type.
        if "classification group:" in message:
            return "classification_failure"
        if message.startswith("data:"):
            return "data_parse_or_store_failure"
        return "source_failure"

    return {
        "error_count": len(errors),
        "errors_by_type": dict(Counter(kind(item) for item in errors)),
        "errors": [item[:_ERROR_PREVIEW_CHARS] for item in errors[:_ERROR_PREVIEW_COUNT]],
    }


# A group that failed for one of these reasons failed in transport, not on its
# content. A later batch retries it, so it never enters checkpoint.failed.
_RETRYABLE_MARKERS = (
    "workstation_outage",
    "Router workstation outage",
    "Router completion failed",
    "router_acp_",
    "import worker:",
    "no_worker_before_deadline",
    "import_deadline",
    "waiter_ended",
    "seat_exhausted",
    "OperationalError",
    "connection is lost",
    "server closed the connection",
)
# Consecutive retryable group failures that end a batch early, so an outage
# cannot spend the whole time budget re-failing groups.
_MAX_CONSECUTIVE_RETRYABLE = 3


def _retryable(message: str) -> bool:
    return any(marker in message for marker in _RETRYABLE_MARKERS)


SOURCE_SYSTEM = "git"
IMPORTER = "kory-source.repository-history"
IMPORTER_VERSION = "5"
# A group is the most a killed process can drop. The following unit pages
# commit the cursor, so earlier groups stay finished.
_GROUP_PASSAGES = 32
_GROUP_FILES = 8
_READ_ONLY = {"log", "show", "ls-tree", "rev-parse", "cat-file", "diff", "blame", "merge-base"}
_DEFAULT_MAX_COMMITS = 200
_MARKDOWN = (".md", ".mdx", ".markdown")
# Comment syntax per language. Anything else is reported as unsupported.
_LINE_COMMENT = {
    "python": "#",
    "ruby": "#",
    "shell": "#",
    "yaml": "#",
    "terraform": "#",
    "javascript": "//",
    "typescript": "//",
    "go": "//",
    "rust": "//",
    "java": "//",
    "kotlin": "//",
    "c": "//",
    "cpp": "//",
    "csharp": "//",
    "swift": "//",
}
_EXT = {
    ".py": "python",
    ".rb": "ruby",
    ".sh": "shell",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".tf": "terraform",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".sql": "sql",
    ".css": "css",
    ".html": "html",
    ".svg": "html",
    ".j2": "jinja",
}
_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
_TOML_KEY = re.compile(r"[A-Za-z0-9_-]+")
_DOLLAR_QUOTE = re.compile(r"\$(?:[A-Za-z_][A-Za-z_0-9]*)?\$")
_BLOCK = re.compile(r"/\*.*?\*/", re.S)
_DOCSTRING = re.compile(r"(?P<q>\"\"\"|''')(?P<body>.*?)(?P=q)", re.S)
_SEP = "\x1e"


def language_of(path: str) -> str:
    return _EXT.get(Path(path).suffix.lower(), "")


_DATA = {".json", ".yaml", ".yml", ".csv", ".toml"}
_TEMPLATE = {".j2", ".html", ".svg", ".css", ".txt"}
_REASON_KEYS = {"reason", "rationale", "why", "decision", "_comment"}
_SKIP_DIRS = {"node_modules", "vendor", "dist", "build", ".git", "__pycache__", ".venv"}
_LOCKFILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "uv.lock",
    "cargo.lock",
    "composer.lock",
    "pipfile.lock",
}


def file_kind(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if path.lower().endswith(_MARKDOWN):
        return "md"
    if suffix == ".sql":
        return "sql"
    if suffix in _DATA:
        return "data"
    if suffix in _TEMPLATE:
        return "template"
    return "code" if language_of(path) else ""


def skip_reason(path: str, text: str | None = None) -> str:
    parts = {p.lower() for p in Path(path).parts}
    name = Path(path).name.lower()
    if name in {
        "changelog.md",
        "changelog.txt",
        "changes.md",
        "history.md",
        "license",
        "license.md",
        "license.txt",
        "copying",
        "copying.md",
        "release-notes.md",
        "release_notes.md",
    }:
        return "boilerplate"
    if parts & _SKIP_DIRS:
        return "dependency_or_generated_directory"
    if name in _LOCKFILES or name.endswith((".lock", ".min.js", ".min.css", ".map")):
        return "lockfile_or_minified"
    if text is not None:
        if "\x00" in text:
            return "binary"
        header = text[:1000].lower()
        if any(
            marker in header
            for marker in (
                "@generated",
                "automatically generated",
                "do not edit: generated",
                "code generated",
            )
        ):
            return "generated"
    return ""


def inventory(
    reader: GitReader, frozen: str, *, paths: list[str] | None = None
) -> tuple[list[tuple[str, str]], Counter[str]]:
    """Same file selection for import, dry-run and coverage reconciliation."""
    units: list[tuple[str, str]] = []
    skipped: Counter[str] = Counter()
    for path in reader.files(frozen):
        if paths and not any(
            path == chosen or path.startswith(chosen.rstrip("/") + "/") for chosen in paths
        ):
            skipped["excluded_path_filter"] += 1
            continue
        reason = skip_reason(path)
        kind = file_kind(path)
        if not reason and kind:
            try:
                body = reader.show(frozen, path)
                reason = "unavailable" if body is None else skip_reason(path, body)
            except UnicodeError:
                reason = "binary"
        if reason or not kind:
            skipped[reason or "unsupported"] += 1
        else:
            units.append((kind, path))
    return units, skipped


def _pointer(parts: list[str]) -> str:
    return "/" + "/".join(p.replace("~", "~0").replace("/", "~1") for p in parts)


def _json_values(text: str):
    """Parse JSONC without YAML semantics, retaining original lexical offsets.

    Mask comments and trailing commas with spaces, never delete bytes. JSON's
    decoder validates the document and decodes each scalar at its source offset.
    """
    tokens = re.compile(r'"(?:\\.|[^"\\])*"|//[^\r\n]*|/\*.*?\*/', re.S)
    clean = tokens.sub(
        lambda m: m[0] if m[0].startswith('"') else re.sub(r"[^\r\n]", " ", m[0]), text
    )
    clean = re.sub(
        r'"(?:\\.|[^"\\])*"|,(?=\s*[}\]])',
        lambda m: " " if m[0] == "," else m[0],
        clean,
    )
    json.loads(clean)
    decoder = json.JSONDecoder()
    position = 0

    def whitespace():
        nonlocal position
        while position < len(clean) and clean[position].isspace():
            position += 1

    def walk(parts, key=""):
        nonlocal position
        whitespace()
        start = position
        if clean[position] in "{[":
            mapping = clean[position] == "{"
            closing = "}" if mapping else "]"
            position += 1
            whitespace()
            index = 0
            while clean[position] != closing:
                name = str(index)
                if mapping:
                    name, position = decoder.raw_decode(clean, position)
                    whitespace()
                    position += 1  # colon, validated above
                yield from walk([*parts, name], name if mapping else key)
                whitespace()
                if clean[position] == ",":
                    position += 1
                    whitespace()
                index += 1
            position += 1
        else:
            value, position = decoder.raw_decode(clean, position)
            if isinstance(value, str):
                yield key, value, start, position, parts

    yield from walk([])


def data_candidates(text: str, suffix: str, *, nominate_values: bool = False) -> list[Candidate]:
    """Reason values with exact lexical quotes and JSON-pointer-style key paths.

    Ordinary descriptions need an explicit commitment. Reason keys identify
    rationale, but questions and hypothetical/negated decisions are excluded.
    Parsing is non-executing; malformed data fails the file rather than vanishing.
    """
    from mira.registry.uncited import _HYPOTHETICAL

    out = []

    def add(key: str, value: str, start: int, end: int, parts: list[str]) -> None:
        if nominate_values:
            if value.strip() and (key.lower() in _REASON_KEYS or nominate(value)):
                out.append(
                    Candidate(start, end, text[start:end], _pointer(parts), "nominated data value")
                )
            return
        if not value.strip() or value.strip().endswith("?"):
            return
        if re.fullmatch(r"(?:none|n/?a|unknown|todo|tbd|pending)[.!]?", value.strip(), re.I):
            return
        explicit = extract_decisions(value)
        if not explicit and (
            key.lower() not in _REASON_KEYS
            or _HYPOTHETICAL.search(value)
            or re.search(r"\b(?:undecided|not\s+(?:yet\s+)?decided|no\s+decision)\b", value, re.I)
        ):
            return
        out.append(
            Candidate(
                start,
                end,
                text[start:end],
                _pointer(parts),
                "decision rationale value at " + _pointer(parts),
            )
        )

    if suffix == ".json":
        for key, value, start, end, parts in _json_values(text):
            add(key, value, start, end, parts)
    elif suffix in {".yaml", ".yml"}:
        root = yaml.compose(text, Loader=_YAML_LOADER)

        def walk(node, parts: list[str], key: str = "", ancestors=frozenset()):
            if node is None or id(node) in ancestors:
                return
            ancestors = ancestors | {id(node)}
            if isinstance(node, yaml.ScalarNode):
                if node.tag == "tag:yaml.org,2002:str":
                    add(key, node.value, node.start_mark.index, node.end_mark.index, parts)
            elif isinstance(node, yaml.MappingNode):
                for k, v in node.value:
                    walk(v, [*parts, k.value], k.value, ancestors)
            elif isinstance(node, yaml.SequenceNode):
                for i, v in enumerate(node.value):
                    walk(v, [*parts, str(i)], key, ancestors)

        walk(root, [])
    elif suffix == ".csv":
        # csv.reader handles quoted embedded newlines. Parse each raw field again
        # to retain its lexical offsets (including escapes and quote delimiters).
        fields = list(re.finditer(r'(?:"(?:[^"]|"")*"|[^,\r\n]*)(?:,|\r?\n|$)', text))
        rows = csv.reader(io.StringIO(text), strict=True)
        headers = next(rows, [])
        header_counts = Counter(headers)
        cursor = 0
        for row_no, row in enumerate(chain([headers], rows)):
            if not row:
                cursor += 1  # csv.reader returns [] for an empty physical record.
                continue
            for col, value in enumerate(row):
                if cursor >= len(fields):
                    raise ValueError("CSV field offsets unavailable")
                m = fields[cursor]
                cursor += 1
                end = m.end()
                if text[end - 1 : end] == ",":
                    end -= 1
                else:
                    while end > m.start() and text[end - 1] in "\r\n":
                        end -= 1
                if row_no:
                    key = headers[col] if col < len(headers) else str(col)
                    parts = ["rows", str(row_no - 1), key]
                    if header_counts[key] > 1:
                        parts.extend(["columns", str(col)])
                    add(key, value, m.start(), end, parts)
    elif suffix == ".toml":
        import tomllib

        tomllib.loads(text)  # Validate before the lexical walk; never execute values.
        position = 0
        section: list[str] = []
        table_counts: Counter[tuple[str, ...]] = Counter()

        def whitespace():
            nonlocal position
            while position < len(text):
                if text[position].isspace():
                    position += 1
                elif text[position] == "#":
                    end = text.find("\n", position)
                    position = len(text) if end < 0 else end + 1
                else:
                    break

        def string_token():
            nonlocal position
            start = position
            delimiter = text[position : position + 3]
            if delimiter not in {'"""', "'''"}:
                delimiter = text[position]
            position += len(delimiter)
            while position < len(text):
                if text.startswith(delimiter, position):
                    position += len(delimiter)
                    if len(delimiter) == 3:
                        while position < len(text) and text[position] == delimiter[0]:
                            position += 1
                    raw = text[start:position]
                    return start, position, tomllib.loads("value = " + raw)["value"]
                if delimiter[0] == '"' and text[position] == "\\":
                    position += 1
                position += 1
            raise ValueError("unterminated TOML string")

        def keys():
            nonlocal position
            parts = []
            while True:
                whitespace()
                if text[position] in "\"'":
                    _, _, part = string_token()
                else:
                    match = _TOML_KEY.match(text, position)
                    if not match:
                        raise ValueError("invalid TOML key")
                    part = match[0]
                    position += len(part)
                parts.append(part)
                whitespace()
                if text[position : position + 1] != ".":
                    return parts
                position += 1

        def scan_value(parts, key):
            nonlocal position
            whitespace()
            char = text[position : position + 1]
            if char in {"'", '"'}:
                start, end, decoded = string_token()
                add(key, decoded, start, end, parts)
            elif char == "{":
                position += 1
                whitespace()
                while text[position : position + 1] != "}":
                    nested = keys()
                    if text[position : position + 1] != "=":
                        raise ValueError("invalid TOML inline assignment")
                    position += 1
                    scan_value([*parts, *nested], nested[-1])
                    whitespace()
                    if text[position : position + 1] != ",":
                        break
                    position += 1
                position += 1
            elif char == "[":
                position += 1
                index = 0
                whitespace()
                while text[position : position + 1] != "]":
                    scan_value([*parts, str(index)], key)
                    index += 1
                    whitespace()
                    if text[position : position + 1] != ",":
                        break
                    position += 1
                    whitespace()
                position += 1
            else:
                while position < len(text) and text[position] not in ",]}\r\n#":
                    position += 1

        while position < len(text):
            whitespace()
            if position >= len(text):
                break
            if text[position] == "[":
                array = text.startswith("[[", position)
                position += 2 if array else 1
                named = keys()
                section = []
                for index, part in enumerate(named):
                    section.append(part)
                    group = tuple(named[: index + 1])
                    if index == len(named) - 1 and array:
                        table_counts[group] += 1
                    if table_counts[group]:
                        section.append(str(table_counts[group] - 1))
                position += 2 if array else 1
            else:
                named = keys()
                if text[position : position + 1] != "=":
                    raise ValueError("invalid TOML assignment")
                position += 1
                scan_value([*section, *named], named[-1])
    return out


def candidate_passages(text: str, path: str) -> list[Passage]:
    """High-recall nominations only; classification determines decisions."""
    kind = file_kind(path)
    if kind == "md":
        return nominate(text, markdown=True)
    if Path(path).suffix.lower() == ".txt":
        return nominate(text)
    out = []
    grouped: list[tuple[int, int]] = []
    for start, end, _ in sorted(comment_spans(text, language_of(path))):
        if grouped and text[grouped[-1][1] : start].strip() in {"#", "//", "--"}:
            grouped[-1] = (grouped[-1][0], end)
        else:
            grouped.append((start, end))
    for start, end in grouped:
        out.extend(nominate(text[start:end], offset=start))
    if kind == "data":
        for cand in data_candidates(text, Path(path).suffix.lower(), nominate_values=True):
            # A rationale-keyed value is nominated in its entirety, even when
            # it contains no explicit decision marker or is ultimately rejected.
            out.extend(nominate(cand.quote, force=True, section=cand.section, offset=cand.start))
    return out


class GitReader:
    """Allowlisted read-only git commands against one checkout."""

    def __init__(self, checkout: str | Path) -> None:
        self.checkout = Path(checkout)
        self._commit_cache: dict[str, dict] = {}
        self._blame_cache: OrderedDict[tuple[str, str], dict[int, dict]] = OrderedDict()

    def git(self, *args: str, allow_no_match: bool = False) -> str:
        if not args or args[0] not in _READ_ONLY:
            raise ReadOnlyViolation(f"git {args[0] if args else ''} is not a read-only command")
        proc = subprocess.run(
            ["git", "-C", str(self.checkout), *args],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if proc.returncode != 0 and not (allow_no_match and proc.returncode == 1):
            raise RuntimeError(f"git {args[0]} failed: {proc.stderr.strip()[:300]}")
        return proc.stdout

    def rev_parse(self, rev: str) -> str:
        return self.git("rev-parse", "--verify", "--end-of-options", f"{rev}^{{commit}}").strip()

    def is_ancestor(self, ancestor: str, commit: str) -> bool:
        return self.git("merge-base", ancestor, commit, allow_no_match=True).strip() == ancestor

    def changed_history(self, revisions: str) -> dict[str, list[str]]:
        """Map changed paths to commits inside the range, newest first."""
        out = self.git(
            "log",
            "--format=%x00%H",
            "-z",
            "--name-only",
            "--no-renames",
            "--diff-merges=first-parent",
            revisions,
            "--",
        )
        history: dict[str, list[str]] = {}
        for record in out.removeprefix("\0").split("\0\0"):
            sha, _, paths = record.partition("\0\n")
            for path in paths.rstrip("\0").split("\0"):
                if path:
                    history.setdefault(path, []).append(sha)
        return history

    def files(self, commit: str) -> list[str]:
        return [p for p in self.git("ls-tree", "-r", "--name-only", commit).splitlines() if p]

    def show(self, commit: str, path: str) -> str | None:
        try:
            return self.git("show", f"{commit}:{path}")
        except RuntimeError:
            # Distinguish a legitimate deletion/rename boundary from a broken
            # revision, corrupt object, permission failure or other git error.
            self.git("rev-parse", "--verify", f"{commit}^{{commit}}")
            if not self.git("ls-tree", "-z", commit, "--", path):
                return None
            raise

    def commits(
        self, commit: str, paths: list[str] | None = None, max_count: int = 200
    ) -> list[dict]:
        fmt = _SEP.join(["%H", "%an", "%ae", "%at", "%cn", "%ce", "%ct", "%B"]) + "\x1d"
        args = [
            "log",
            f"--max-count={max_count}" if max_count else "",
            "--follow" if paths and len(paths) == 1 else "",
            f"--format={fmt}",
            commit,
        ]
        args = [a for a in args if a]
        if paths:
            args += ["--", *paths]
        out = []
        for chunk in self.git(*args).split("\x1d"):
            parts = chunk.strip("\n").split(_SEP)
            if len(parts) < 8:
                continue
            sha, an, ae, at, cn, ce, ct, body = parts[:8]
            out.append(
                {
                    "sha": sha,
                    "author_name": an,
                    "author_email": ae,
                    "author_date": float(at or 0),
                    "committer_name": cn,
                    "committer_email": ce,
                    "committer_date": float(ct or 0),
                    "message": body,
                }
            )
        self._commit_cache.update({str(entry["sha"]): entry for entry in out})
        return out

    def path_at(self, commit: str, path: str, max_count: int = 0) -> list[tuple[str, str]]:
        """``(commit, path)`` pairs for a file's history, following renames."""
        args = ["log", "--follow", "--name-only", "--format=%x1e%H"]
        if max_count:
            args.append(f"--max-count={max_count}")
        out = self.git(*args, commit, "--", path)
        pairs: list[tuple[str, str]] = []
        for block in out.split("\x1e"):
            lines = [line for line in block.strip().splitlines() if line]
            if len(lines) >= 2:
                pairs.append((lines[0], lines[-1]))
        return pairs

    def blame_many(self, commit: str, path: str, lines: set[int]) -> dict[int, dict]:
        """One blame invocation for all decision lines in a file revision."""
        key = (commit, path)
        cached = self._blame_cache.setdefault(key, {})
        self._blame_cache.move_to_end(key)
        missing = sorted(lines - cached.keys())
        if missing:
            ranges: list[tuple[int, int]] = []
            for line in missing:
                if ranges and line == ranges[-1][1] + 1:
                    ranges[-1] = (ranges[-1][0], line)
                else:
                    ranges.append((line, line))
            args = ["blame", "--line-porcelain"]
            for start, end in ranges:
                args += ["-L", f"{start},{end}"]
            raw = self.git(*args, commit, "--", path)
            sha, final_line = "", 0
            for entry in raw.splitlines():
                header = re.match(r"^([0-9a-f]{40,64}) (\d+) (\d+)(?: \d+)?$", entry)
                if header:
                    sha, final_line = header[1], int(header[3])
                elif entry.startswith("\t"):
                    if sha not in self._commit_cache:
                        found = self.commits(sha, max_count=1)
                        if not found:
                            raise RuntimeError("blame origin commit metadata unavailable")
                    cached[final_line] = self._commit_cache[sha]
            if not lines <= cached.keys():
                raise RuntimeError("blame omitted requested decision lines")
        while len(self._blame_cache) > 64:
            self._blame_cache.popitem(last=False)
        return {line: cached[line] for line in lines}

    def blame(self, commit: str, path: str, line: int) -> dict:
        return self.blame_many(commit, path, {line})[line]


def comment_spans(text: str, language: str) -> list[tuple[int, int, int]]:
    """``(start, end, line)`` of each comment or docstring body in ``text``."""
    spans: list[tuple[int, int, int]] = []
    newlines = [m.start() for m in re.finditer("\n", text)]
    if language in {"sql", "css", "html", "jinja"}:
        delimiters = {
            "sql": [("--", "\n"), ("/*", "*/")],
            "css": [("/*", "*/")],
            "html": [("<!--", "-->")],
            "jinja": [("{#", "#}")],
        }[language]
        i = 0
        while i < len(text):
            matched = False
            for opening, closing in delimiters:
                if text.startswith(opening, i):
                    end = text.find(closing, i + len(opening))
                    if end < 0:
                        end = len(text)
                    spans.append((i + len(opening), end, bisect_right(newlines, i - 1) + 1))
                    i = end + len(closing)
                    matched = True
                    break
            if matched:
                continue
            # SQL strings/identifiers and dollar-quoted function bodies are
            # executable values, never comments. CSS strings are likewise inert.
            if language in {"sql", "css"} and text[i] in "'\"":
                q = text[i]
                i += 1
                while i < len(text):
                    if text[i] == q:
                        if i + 1 < len(text) and text[i + 1] == q:
                            i += 2
                            continue
                        i += 1
                        break
                    if text[i] == "\\":
                        i += 1
                    i += 1
            elif language == "sql" and text[i] == "$" and (m := _DOLLAR_QUOTE.match(text, i)):
                end = text.find(m[0], i + len(m[0]))
                i = len(text) if end < 0 else end + len(m[0])
            else:
                i += 1
        return spans
    marker = _LINE_COMMENT.get(language)
    if marker:
        offset = 0
        for no, line in enumerate(text.splitlines(keepends=True), start=1):
            idx = line.find(marker)
            if idx >= 0 and line[:idx].strip() == "":
                start = offset + idx + len(marker)
                end = offset + len(line.rstrip("\r\n"))
                spans.append((start, end, no))
            offset += len(line)
    if language in (
        "javascript",
        "typescript",
        "go",
        "rust",
        "java",
        "kotlin",
        "c",
        "cpp",
        "csharp",
        "swift",
    ):
        for m in _BLOCK.finditer(text):
            spans.append((m.start() + 2, m.end() - 2, bisect_right(newlines, m.start() - 1) + 1))
    if language == "python":
        for m in _DOCSTRING.finditer(text):
            spans.append(
                (m.start("body"), m.end("body"), bisect_right(newlines, m.start() - 1) + 1)
            )
    return spans


class RepositoryHistoryImporter:
    def __init__(
        self,
        registry: Any,
        reader: GitReader,
        *,
        repo: str,
        access: AccessContext,
        commit: str = "HEAD",
        max_commits: int = _DEFAULT_MAX_COMMITS,
        classifier: Any = None,
        paths: list[str] | None = None,
        max_seconds: float = 0,
        max_units: int = 0,
        incremental: bool = False,
    ) -> None:
        if max_seconds < 0:
            raise ValueError("max_seconds must be nonnegative")
        if max_units < 0:
            raise ValueError("max_units must be nonnegative (0 means no unit cap)")
        self.max_seconds = max_seconds
        self.max_units = max_units
        self.paths = sorted(set(paths or []))
        self.classifier = classifier
        self.registry = registry
        self.reader = reader
        self.repo = repo
        self.access = access
        self.commit = commit
        if max_commits < 0:
            raise ValueError("max_commits must be nonnegative (0 means unbounded)")
        if incremental and (max_commits != _DEFAULT_MAX_COMMITS or self.paths):
            raise ValueError("incremental history does not accept a commit limit or path filter")
        self.max_commits = max_commits
        self.incremental = incremental
        self._range = ""
        self._history: dict[str, list[str]] = {}
        self._range_commits: dict[str, dict] = {}
        self.store = ProvenanceStore(registry)
        self.runs = ImportRuns(registry)

    def _pending_range(self, policy, stream, since):
        for run in reversed(self.runs.coverage(self.access, unfinished_stream=stream)):
            if (
                run["source_system"] == SOURCE_SYSTEM
                and run["corpus_id"] == policy.corpus_id
                and run["policy_version"] == policy.policy_version
                and run["importer_version"] == IMPORTER_VERSION
            ):
                scope = json.loads(run["scope_manifest"])
                if scope.get("since") == since:
                    return scope
        return None

    def run(self, *, apply: bool = False) -> dict:
        started_run = time.monotonic()
        self.stages = {kind: Counter() for kind in ("md", "code", "sql", "data", "template", "log")}
        self.attempt_estimates: Counter[str] = Counter()
        policy = require_import(self.access)
        if not policy.admits_repository(self.repo):
            raise AccessDenied("the repository is not in the approved corpus manifest")
        stream = f"repository:{self.repo}"
        since = ""
        pending = None
        self._range = ""
        self._history = {}
        self._range_commits = {}
        if self.incremental:
            since = read_cursors(self.registry, SOURCE_SYSTEM).get(stream, {}).get("cursor", "")
            if not since:
                return {"apply": apply, "status": "waiting_for_backfill"}
            frozen = self.reader.rev_parse(self.commit)
            try:
                cursor_found = self.reader.rev_parse(since) == since
            except RuntimeError:
                cursor_found = False
            if not cursor_found or not self.reader.is_ancestor(since, frozen):
                return {
                    "apply": apply,
                    "status": "cursor_not_ancestor",
                    "since": since,
                    "commit": frozen,
                }
            if since == frozen:
                return {"apply": apply, "status": "up_to_date", "since": since, "commit": frozen}
            pending = self._pending_range(policy, stream, since)
            if pending:
                until = self.reader.rev_parse(pending["commit"])
                if not self.reader.is_ancestor(until, frozen):
                    return {
                        "apply": apply,
                        "status": "pending_range_ahead",
                        "since": since,
                        "commit": until,
                    }
                frozen = until
            self._range = f"{since}..{frozen}"
        if self.classifier is None:
            self.classifier = decision_llm.default_classifier(
                self.registry, corpus_id=policy.corpus_id
            )
        if apply:
            self.classifier.validate_configuration()
        if not self.incremental:
            frozen = self.reader.rev_parse(self.commit)
        self.classifier.deadline = started_run + self.max_seconds if self.max_seconds else None
        stats_before = Counter(getattr(self.classifier, "stats", {}))
        self._stats_flushed = Counter(stats_before)
        if self.classifier.corpus_id != policy.corpus_id:
            raise AccessDenied("classifier cache belongs to another corpus")
        scope = {
            "repo": self.repo,
            "paths": self.paths,
            "commit": frozen,
            "max_commits": 0 if self.incremental else self.max_commits,
            "importer_version": IMPORTER_VERSION,
            "extractor_version": decision_llm.EXTRACTOR_VERSION,
            "extractor_model": self.classifier.model,
        }
        if self.incremental:
            scope["since"] = since
        if pending and pending != scope:
            raise ValueError(
                "resume the saved incremental range with its original extractor configuration"
            )
        if self.incremental:
            self._history = self.reader.changed_history(self._range)
            selected = []
            skipped: Counter[str] = Counter()
            for path in sorted(self._history):
                reason, kind = skip_reason(path), file_kind(path)
                if reason or not kind:
                    skipped[reason or "unsupported"] += 1
                else:
                    selected.append((kind, path))
        else:
            selected, skipped = inventory(self.reader, frozen, paths=self.paths)
        range_details = {"since": since} if self.incremental else {}
        if not apply:
            totals = Counter(kind for kind, _ in selected)
            estimates: Counter[str] = Counter()
            seen: set[str] = set()
            planning_errors: list[str] = []
            for group in self._prepared_groups(
                frozen, selected + [("log", "")], set(), planning_errors
            ):
                prepared = [revision for _, _, revisions, _ in group for revision in revisions]
                _, counts = self._classify_prepared(prepared, apply=False, seen=seen)
                estimates.update(counts)
                for _, _, _, observations in group:
                    estimates.update(observations)
            return {
                "apply": False,
                "commit": frozen,
                **range_details,
                **{
                    f"{label}_files": totals[kind]
                    for kind, label in (
                        ("md", "markdown"),
                        ("code", "code"),
                        ("sql", "sql"),
                        ("data", "data"),
                        ("template", "template"),
                    )
                },
                "candidate_passages": estimates["candidate_passages"],
                "uncached_passages": estimates["uncached_passages"],
                "estimated_llm_calls": estimates["estimated_llm_calls"],
                "extractor_model": self.classifier.model,
                "batch_size": getattr(self.classifier, "batch_size", decision_llm.BATCH_SIZE),
                "batch_chars": getattr(self.classifier, "batch_chars", decision_llm.BATCH_CHARS),
                "max_concurrency": getattr(self.classifier, "max_concurrency", 2),
                "paths": self.paths,
                "max_commits_per_file": scope["max_commits"],
                **_error_preview(planning_errors),
                "estimate_complete": not planning_errors,
                "history_skipped": {k: v for k, v in estimates.items() if k.startswith("skipped_")},
                "unsupported_files": skipped["unsupported"],
                "skipped": dict(skipped),
            }
        previous = (
            ()
            if self.incremental
            else self.registry.db.execute(
                "SELECT run_id FROM registry_import_runs WHERE corpus_id = ? AND policy_version = ? "
                "AND source_system = ? AND stream = ? AND importer_version = ? AND status = 'complete' "
                "ORDER BY finished_at DESC",
                (
                    policy.corpus_id,
                    policy.policy_version,
                    SOURCE_SYSTEM,
                    stream,
                    IMPORTER_VERSION,
                ),
            )
        )
        for (run_id,) in previous:
            prior = self.runs.get(run_id)
            if json.loads(prior["scope_manifest"]) == scope:
                return {
                    "run_id": run_id,
                    "status": "complete",
                    "commit": frozen,
                    "coverage": prior["coverage"],
                    "reused_completed_run": True,
                    **_error_preview([]),
                }
        run = self.runs.start(
            self.access,
            source_system=SOURCE_SYSTEM,
            stream=stream,
            importer=IMPORTER,
            importer_version=IMPORTER_VERSION,
            scope=scope,
            extractor=decision_llm.EXTRACTOR,
            extractor_version=decision_llm.EXTRACTOR_VERSION,
            extractor_model=self.classifier.model,
        )
        self.classifier.import_run_id = run["run_id"]
        done = set((run["checkpoint"] or {}).get("done") or [])
        failed = dict((run["checkpoint"] or {}).get("failed") or {})
        coverage_before = Counter(run["coverage"])
        errors: list[str] = []
        # An older batch may have recorded a transport failure as permanent.
        # Retry those files. The run's errors keep the original text.
        for key in sorted(failed):
            if _retryable(failed[key]):
                message = failed.pop(key)
                if not self.incremental:
                    errors.append(f"{key}: {message}")
        consecutive_retryable = 0
        units_read: Counter[str] = Counter()
        units = selected + [("log", "")]
        stop = False
        for group in self._prepared_groups(frozen, units, done, errors, failed):
            prepared = [revision for _, _, revisions, _ in group for revision in revisions]
            for kind, _, revisions, _ in group:
                self.stages[kind]["grouped"] += sum(len(r[4]) for r in revisions)
            try:
                classified, estimates = self._classify_prepared(prepared, apply=True)
            except (RuntimeError, ValueError) as exc:
                message = f"classification group: {exc}"
                if not self.incremental or _retryable(message) or "time budget reached" in message:
                    _record_error(errors, message)
                # A budget stop is resumable. Completed batches stay cached, so
                # the same files are nominated again. So is a transport failure.
                # A content failure is recorded and skipped, or one file would
                # consume every later batch.
                if "time budget reached" in str(exc):
                    self._flush_stats(run)
                    continue
                if _retryable(message):
                    self._flush_stats(run)
                    consecutive_retryable += 1
                    if consecutive_retryable >= _MAX_CONSECUTIVE_RETRYABLE:
                        _record_error(
                            errors,
                            "retryable failures: stopping batch after "
                            f"{consecutive_retryable} consecutive groups",
                        )
                        break
                    continue
                consecutive_retryable = 0
                for kind, path, _, _ in group:
                    failed[f"{kind}:{path}"] = message
                    if self.incremental:
                        logger.warning("%s:%s: %s", kind, path, message)
                self._commit_cursor(run, done, failed)
                continue
            finally:
                # Finished batches are cached even if a sibling batch failed.
                # Report those successes and the unclassified remainder, rather
                # than losing the entire failed group's nomination estimates.
                cached = self.classifier.classification_outcomes
                for kind, _, revisions, _ in group:
                    passages = [p for r in revisions for p in r[4]]
                    outcomes = [cached.get(decision_llm.passage_hash(p.text)) for p in passages]
                    skipped_outcomes = sum(
                        r is not None and r.get("outcome") == "skipped" for r in outcomes
                    )
                    ok = sum(r is not None for r in outcomes) - skipped_outcomes
                    self.stages[kind]["classified_ok"] += ok
                    self.stages[kind]["classified_skipped"] += skipped_outcomes
                    self.stages[kind]["classified_failed"] += len(passages) - ok - skipped_outcomes
            consecutive_retryable = 0
            offset = 0
            for kind, path, revisions, observations in group:
                key = f"{kind}:{path}"
                kept = classified[offset : offset + len(revisions)]
                offset += len(revisions)
                done.add(key)
                run["next_checkpoint"] = {**(run.get("checkpoint") or {}), "done": sorted(done)}
                try:
                    with self.runs.page(run, self.access) as counts:
                        counts.update(self._stat_delta())
                        counts[f"units_read_{kind}"] += 1
                        counts.update(observations)
                        for sha, old_path, text, commit, candidates in kept:
                            if kind == "log":
                                self._log_record(run, commit, candidates, counts)
                            else:
                                identity = f"{self.repo}:{path}" + (
                                    ":code" if kind == "code" else ""
                                )
                                self._file_record(
                                    run,
                                    sha,
                                    old_path,
                                    text,
                                    commit,
                                    counts,
                                    identity=identity,
                                    candidates=candidates,
                                )
                    self._mark_stats_flushed()
                    units_read[kind] += 1
                    if self.max_units and sum(units_read.values()) >= self.max_units:
                        stop = True
                        break
                except (RuntimeError, ValueError, UnicodeError, yaml.YAMLError, csv.Error) as exc:
                    done.discard(key)
                    run["next_checkpoint"] = {**(run.get("checkpoint") or {}), "done": sorted(done)}
                    self._unit_error(key, str(exc), errors, failed)
            if stop:
                break
        llm_calls_delta = (
            Counter(getattr(self.classifier, "stats", {}))["llm_calls"] - stats_before["llm_calls"]
        )
        batch = {
            "units_read": dict(units_read),
            "llm_calls_delta": llm_calls_delta,
            "errors_by_type": _error_preview(errors)["errors_by_type"],
        }
        with self.runs.page(run, self.access) as counts:
            counts.update(self._stat_delta())
            counts.update(self.attempt_estimates)
            for kind, stages in self.stages.items():
                counts.update({f"{kind}_passages_{stage}": n for stage, n in stages.items()})
            checkpoint = dict(run.get("checkpoint") or {})
            if not checkpoint.get("skips_recorded"):
                counts["excluded_scope"] += sum(skipped.values())
                counts.update({f"skipped_{key}": value for key, value in skipped.items()})
                checkpoint["skips_recorded"] = True
            checkpoint["done"] = sorted(done)
            if failed:
                checkpoint["failed"] = {key: failed[key] for key in sorted(failed)}
            else:
                checkpoint.pop("failed", None)
            checkpoint["last_batch"] = batch
            run["next_checkpoint"] = checkpoint
        self._mark_stats_flushed()
        reported_errors = list(errors)
        for key, message in sorted(failed.items()):
            line = f"{key}: {message}"
            if line not in reported_errors and message not in reported_errors:
                reported_errors.append(line)
        finished = set(done) | set(failed)
        remaining = any(f"{kind}:{path}" not in finished for kind, path in units)
        if self.incremental:
            complete = not remaining and not errors
            watermark = (
                float(self.reader.git("show", "-s", "--format=%ct", frozen).strip())
                if complete
                else 0
            )
            with self.registry.db.transaction():
                if complete and not record_cursor(
                    self.registry,
                    SOURCE_SYSTEM,
                    stream,
                    watermark=watermark,
                    cursor=frozen,
                    expected_cursor=since,
                ):
                    self.runs.finish(run, complete=False, errors=["cursor changed during import"])
                    return {
                        "apply": True,
                        "status": "cursor_changed",
                        "commit": frozen,
                        **range_details,
                    }
                final = self.runs.finish(run, complete=complete, errors=errors)
        else:
            final = self.runs.finish(
                run,
                complete=(not remaining)
                and not failed
                and self.max_commits == 0
                and not self.paths,
                errors=reported_errors,
                unknown_totals=(["history_before_max_commits"] if self.max_commits else [])
                + (["unselected_paths"] if self.paths else []),
            )
            if final["status"] == "complete":
                watermark = float(self.reader.git("show", "-s", "--format=%ct", frozen).strip())
                record_cursor(
                    self.registry, SOURCE_SYSTEM, stream, watermark=watermark, cursor=frozen
                )
        logger.info(
            "repository history batch run_id=%s status=%s units_read=%s "
            "llm_calls_delta=%s errors_by_type=%s",
            final["run_id"],
            final["status"],
            dict(units_read),
            llm_calls_delta,
            batch["errors_by_type"],
        )
        return {
            "run_id": final["run_id"],
            "status": final["status"],
            "commit": frozen,
            **range_details,
            "coverage": final["coverage"],
            "slice_coverage": {
                key: value - (coverage_before[key] or 0)
                for key, value in final["coverage"].items()
                if isinstance(value, (int, float))
            },
            "per_source": {
                kind: {
                    **{
                        stage: stages[stage]
                        for stage in (
                            "nominated",
                            "grouped",
                            "classified_ok",
                            "classified_failed",
                            "classified_skipped",
                        )
                    },
                    "stored": final["coverage"].get(f"{kind}_stored_passages", 0)
                    - coverage_before[f"{kind}_stored_passages"],
                    "stored_evidence": final["coverage"].get(f"{kind}_stored_evidence", 0)
                    - coverage_before[f"{kind}_stored_evidence"],
                }
                for kind, stages in self.stages.items()
            },
            "units_read": dict(units_read),
            "llm_calls_delta": llm_calls_delta,
            **_error_preview(reported_errors),
        }

    def _stat_delta(self) -> Counter[str]:
        current = Counter(getattr(self.classifier, "stats", {}) or {})
        return Counter(
            {key: value for key, value in (current - self._stats_flushed).items() if value > 0}
        )

    def _mark_stats_flushed(self) -> None:
        self._stats_flushed = Counter(getattr(self.classifier, "stats", {}) or {})

    def _flush_stats(self, run: dict) -> None:
        """Commit classifier counters that a failed group would otherwise drop."""
        delta = self._stat_delta()
        if not delta:
            return
        with self.runs.page(run, self.access) as counts:
            counts.update(delta)
        self._mark_stats_flushed()

    def _commit_cursor(self, run: dict, done: set[str], failed: dict[str, str]) -> None:
        """Record a failed group before the next file is nominated.

        The keys stay out of ``done`` so nothing is stored for them. Resume
        skips them. A kill after this page cannot nominate the same files again.
        """
        checkpoint = dict(run.get("checkpoint") or {})
        checkpoint["done"] = sorted(done)
        checkpoint["failed"] = {key: failed[key] for key in sorted(failed)}
        run["next_checkpoint"] = checkpoint
        delta = self._stat_delta()
        with self.runs.page(run, self.access) as counts:
            if delta:
                counts.update(delta)
        self._mark_stats_flushed()

    def _git_attrs(self, eid: str, commit: dict) -> list[attr.Attribution]:
        found = attr.markers(commit.get("message") or "")
        attributions = [*attr.git_author(eid, commit), *attr.from_markers(eid, found)]
        if not any(a.role == "decision_author" for a in attributions):
            account = next((a for a in attributions if a.role == "source_account"), None)
            if account:
                inferred = attr.Attribution(
                    "decision_author",
                    eid,
                    reason="inferred from source git author; identity does not establish human approval",
                )
                for name in ("original_identity", "normalized_identity", "occurred_at"):
                    if account.values.get(name):
                        inferred.set(name, account.values[name], "inferred", pointer=f"git/{name}")
                attributions.append(inferred)
        return attributions

    def _collect_unit(self, frozen, kind, path):
        """Classify/cache outside the atomic evidence page transaction."""
        observations: Counter[str] = Counter()
        prepared = []
        if self._range and kind != "code":
            if not self._range_commits:
                self._range_commits = {
                    c["sha"]: c for c in self.reader.commits(self._range, max_count=0)
                }
            shas = self._range_commits if kind == "log" else self._history[path]
            versions = [(sha, path, self._range_commits[sha]) for sha in shas]
        elif kind == "log":
            versions = [
                (c["sha"], "", c) for c in self.reader.commits(frozen, None, self.max_commits)
            ]
        elif kind == "code":
            if frozen not in self.reader._commit_cache:
                self.reader.commits(frozen, max_count=1)
            versions = [(frozen, path, self.reader._commit_cache.get(frozen, {"sha": frozen}))]
        else:
            history = self.reader.path_at(frozen, path, self.max_commits)
            if self.max_commits:
                history = history[: self.max_commits]
            meta = {c["sha"]: c for c in self.reader.commits(frozen, [path], self.max_commits)}
            versions = [(sha, old, meta.get(sha, {"sha": sha})) for sha, old in history]
        for sha, old_path, commit in versions:
            if (
                self.classifier.deadline is not None
                and time.monotonic() >= self.classifier.deadline
            ):
                raise RuntimeError("nomination time budget reached; resume this scope")
            try:
                text = commit["message"] if kind == "log" else self.reader.show(sha, old_path)
            except UnicodeError:
                observations["skipped_history_binary"] += 1
                observations["excluded_scope"] += 1
                continue
            if text is None:
                observations["skipped_history_absent_path"] += 1
                logger.info("path absent at that commit: %s:%s", sha, old_path)
                continue
            reason = skip_reason(old_path, text) if kind != "log" else ""
            if reason:
                observations[f"skipped_history_{reason}"] += 1
                observations["excluded_scope"] += 1
                continue
            passages = nominate(text) if kind == "log" else candidate_passages(text, old_path)
            self.stages[kind]["nominated"] += len(passages)
            prepared.append((sha, old_path, text, commit, passages))
        return prepared, observations

    def _classify_prepared(self, prepared, *, apply, seen=None):
        """Pack disjoint source revisions across files without losing offsets."""
        observations: Counter[str] = Counter()
        # Translate offsets into one disjoint virtual file for classification,
        # so all historical revisions share bounded batches and identical text
        # is classified only once. Then restore each revision's real offsets.
        flat = []
        bases = []
        base = 0
        for _, _, text, _, passages in prepared:
            bases.append(base)
            flat.extend(replace(p, start=p.start + base, end=p.end + base) for p in passages)
            base += len(text) + 1
        estimate = self.classifier.estimate(flat, seen=seen)
        observations.update(
            {
                key: estimate[key]
                for key in ("candidate_passages", "uncached_passages", "estimated_llm_calls")
            }
        )
        if apply:
            self.attempt_estimates.update(observations)
        extracted = self.classifier.classify(flat) if apply else []
        result = []
        for (sha, old_path, text, commit, _passages), base in zip(prepared, bases, strict=True):
            candidates = [
                replace(c, start=c.start - base, end=c.end - base)
                for c in extracted
                if base <= c.start and c.end <= base + len(text)
            ]
            values = (
                data_candidates(text, Path(old_path).suffix.lower(), nominate_values=True)
                if candidates and file_kind(old_path) == "data"
                else []
            )
            for index, cand in enumerate(candidates):
                value = next(
                    (
                        p
                        for p in values
                        if p.section == cand.section and p.start <= cand.start and cand.end <= p.end
                    ),
                    None,
                )
                if value:
                    candidates[index] = replace(
                        cand, start=value.start, end=value.end, quote=value.quote
                    )
            result.append((sha, old_path, text, commit, candidates))
        prepared = result
        return prepared, observations

    def _prepare_unit(self, frozen, kind, path, *, apply, seen=None):
        prepared, observations = self._collect_unit(frozen, kind, path)
        result, estimates = self._classify_prepared(prepared, apply=apply, seen=seen)
        observations.update(estimates)
        return result, observations

    def _prepared_groups(self, frozen, units, done, errors, failed=None):
        """Pack a short wave, then let the caller checkpoint it."""
        failed = {} if failed is None else failed
        group, count, chars = [], 0, 0
        for number, (kind, path) in enumerate(units, 1):
            key = f"{kind}:{path}"
            if key in done or key in failed:
                continue
            if (
                self.classifier.deadline is not None
                and time.monotonic() >= self.classifier.deadline
            ):
                _record_error(errors, "time budget reached; resume this scope")
                break
            logger.info("Nominate unit %s/%s: %s", number, len(units), key)
            try:
                prepared, observations = self._collect_unit(frozen, kind, path)
            except (RuntimeError, ValueError, UnicodeError, yaml.YAMLError, csv.Error) as exc:
                self._unit_error(key, str(exc), errors, failed if self.incremental else None)
                continue
            group.append((kind, path, prepared, observations))
            count += sum(len(passages) for _, _, _, _, passages in prepared)
            chars += sum(len(text) for _, _, text, _, _ in prepared)
            if count >= _GROUP_PASSAGES or len(group) >= _GROUP_FILES or chars >= 8 * 1024 * 1024:
                yield group
                group, count, chars = [], 0, 0
        if group:
            yield group

    def _unit_error(self, key, message, errors, failed):
        if (
            self.incremental
            and failed is not None
            and not _retryable(message)
            and "time budget reached" not in message
            and not message.startswith("git ")
        ):
            failed[key] = message
            logger.warning("%s: %s", key, message)
        else:
            _record_error(errors, f"{key}: {message}")

    def _record_timeout_skips(self, run, sha, path, text, counts):
        skipped = getattr(self.classifier, "skipped", {})
        if not skipped:
            return
        passages = candidate_passages(text, path) if path else nominate(text)
        records = list(
            (run.get("next_checkpoint") or run.get("checkpoint") or {}).get(
                "classification_skips", []
            )
        )
        known = {(r["commit"], r["path"], r["start"], r["end"]) for r in records}
        hashes = {r["passage_sha256"] for r in records}
        for passage in passages:
            digest = decision_llm.passage_hash(passage.text)
            key = (sha, path, passage.start, passage.end)
            if digest not in skipped or key in known:
                continue
            records.append(
                {
                    "commit": sha,
                    "path": path,
                    "start": passage.start,
                    "end": passage.end,
                    "section": passage.section,
                    "passage_sha256": digest,
                    "reason": skipped[digest]["skip_reason"],
                }
            )
            known.add(key)
            counts["skipped_timeout_passages"] += 1
            counts["skipped_single_passage_timeout"] += 1
            if digest not in hashes:
                hashes.add(digest)
                counts["skipped_timeout_unique_passages"] += 1
        run.setdefault("next_checkpoint", dict(run.get("checkpoint") or {}))[
            "classification_skips"
        ] = records

    def _file_record(self, run, sha, path, text, commit, counts, *, identity, candidates):
        from mira.index.extract import extract_symbols

        kind = file_kind(path)
        language = language_of(path)
        self._record_timeout_skips(run, sha, path, text, counts)
        if self.incremental and not candidates:
            return
        rev = SourceRevision(
            SOURCE_SYSTEM,
            "git_markdown" if kind == "md" else "git_source_file",
            f"{self.repo}:{path}",
            text,
            source_url=f"https://github.com/{self.repo}/blob/{sha}/{quote(path, safe='/')}",
            source_labels={"path": path, "language": language, "kind": kind},
            source_version=sha,
            source_created_at=float(commit.get("author_date") or 0),
            repository=self.repo,
            commit_sha=sha,
        )
        if not self.incremental:
            eid, created = self.store.record_revision(
                rev, corpus=(run["corpus_id"], run["policy_version"]), run_id=run["run_id"]
            )
            counts["stored_evidence" if created else "duplicate_evidence"] += 1
            counts[f"{kind}_stored_evidence" if created else f"{kind}_duplicate_evidence"] += 1
        counts["discovered"] += 1
        counts["fetched_versions"] += 1
        counts[f"{kind}_stored_passages"] += len(candidates)
        counts[f"{kind}_versions"] += 1
        symbols = extract_symbols(text, language) if kind == "code" else []
        newlines = [m.start() for m in re.finditer("\n", text)]

        def line_of(offset: int) -> int:
            return bisect_right(newlines, offset - 1) + 1

        origins = (
            self.reader.blame_many(sha, path, {line_of(c.start) for c in candidates})
            if candidates
            else {}
        )
        for cand in candidates:
            line = line_of(cand.start)
            line_end = line_of(max(cand.start, cand.end - 1))
            origin = origins[line]
            if self.incremental:
                excerpt = replace(
                    rev,
                    original_text=text[cand.start : cand.end],
                    source_labels={
                        **rev.source_labels,
                        "excerpt_start": cand.start,
                    },
                )
                eid, created = self.store.record_revision(
                    excerpt, corpus=(run["corpus_id"], run["policy_version"]), run_id=run["run_id"]
                )
                counts["stored_evidence" if created else "duplicate_evidence"] += 1
                counts[f"{kind}_stored_evidence" if created else f"{kind}_duplicate_evidence"] += 1
                cand = replace(cand, start=0, end=len(excerpt.original_text))
            attrs = self._git_attrs(eid, origin)
            self.store.record_attributions(attrs, corpus_id=run["corpus_id"])
            enclosing = [s for s in symbols if s.start_line <= line <= s.end_line]
            sym = min(enclosing, key=lambda s: s.end_line - s.start_line) if enclosing else None
            anchor = {
                "repository": self.repo,
                "observed_commit": sha,
                "path": path,
                "heading": cand.section,
                "line_start": line,
                "line_end": line_end,
                "qualified_name": (sym.qualified_name or sym.name) if sym else "",
                "symbol_kind": sym.kind if sym else "",
                "resolution_method": "deterministic_extractor" if sym else "path_only",
                "resolution_status": "exact" if sym else "path_only",
                "resolution_confidence": "high" if sym else "medium",
            }
            materialize(
                self.registry,
                run=run,
                ctx=self.access,
                source_system=SOURCE_SYSTEM,
                evidence_id=eid,
                source_url=f"{rev.source_url}#L{line}-L{line_end}",
                repo=self.repo,
                candidates=[cand],
                attributions=attrs,
                counts=counts,
                anchor=anchor,
                occurred_at=origin["author_date"],
                identity=f"{identity}:{cand.section}" if kind == "data" else identity,
            )

    def _log_record(self, run, commit, candidates, counts):
        self._record_timeout_skips(run, commit["sha"], "", commit["message"], counts)
        if self.incremental and not candidates:
            return
        rev = SourceRevision(
            SOURCE_SYSTEM,
            "git_commit_message",
            f"{self.repo}@{commit['sha']}",
            commit["message"],
            source_url=f"https://github.com/{self.repo}/commit/{commit['sha']}",
            source_version=commit["sha"],
            source_created_at=commit["author_date"],
            repository=self.repo,
            commit_sha=commit["sha"],
        )
        before = counts["stored_evidence"]
        self._record(
            run, rev, self._git_attrs(rev.evidence_id, commit), counts, candidates=candidates
        )
        counts["log_stored_passages"] += len(candidates)
        counts["log_stored_evidence"] += counts["stored_evidence"] - before

    def _record(
        self,
        run: dict,
        rev: SourceRevision,
        attrs: list[attr.Attribution],
        counts: Counter[str],
        *,
        anchor_path: str = "",
        commit: str = "",
        identity: str = "",
        candidates: list[Candidate] | None = None,
    ) -> None:
        eid, created = self.store.record_revision(
            rev, corpus=(run["corpus_id"], run["policy_version"]), run_id=run["run_id"]
        )
        counts["discovered"] += 1
        counts["fetched_versions"] += 1
        counts["stored_evidence" if created else "duplicate_evidence"] += 1
        self.store.record_attributions(attrs, corpus_id=run["corpus_id"])
        candidates = candidates if candidates is not None else extract_decisions(rev.original_text)
        for cand in candidates:
            anchor = None
            if anchor_path:
                line = rev.original_text.count("\n", 0, cand.start) + 1
                anchor = {
                    "repository": self.repo,
                    "observed_commit": commit,
                    "path": anchor_path,
                    "heading": cand.section,
                    "line_start": line,
                    "line_end": rev.original_text.count("\n", 0, cand.end) + 1,
                    "resolution_method": "markdown_section",
                    "resolution_status": "path_only",
                    "resolution_confidence": "medium",
                }
            materialize(
                self.registry,
                run=run,
                ctx=self.access,
                source_system=SOURCE_SYSTEM,
                evidence_id=eid,
                source_url=rev.source_url,
                repo=self.repo,
                candidates=[cand],
                attributions=attrs,
                counts=counts,
                anchor=anchor,
                occurred_at=rev.source_created_at,
                identity=identity or f"{rev.source_type}:{rev.source_id}",
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mira.registry.repository_history")
    parser.add_argument("command", choices=("import",))
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument(
        "--max-commits",
        type=int,
        default=_DEFAULT_MAX_COMMITS,
        help="Commit limit; 0 walks all history",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0,
        help="Optional run budget; incomplete apply runs resume",
    )
    parser.add_argument(
        "--max-units",
        type=int,
        default=0,
        help="Stop after this many stored units; 0 means no unit cap. An unfinished apply resumes.",
    )
    parser.add_argument("--path", action="append", help="Limit to a file or directory; repeatable")
    parser.add_argument(
        "--incremental", action="store_true", help="Read changes after the backfill cursor"
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if args.incremental and (args.max_commits != _DEFAULT_MAX_COMMITS or args.path):
        parser.error("--incremental does not accept a commit limit or --path")
    logging.basicConfig(level=logging.INFO)
    from mira.registry import get_registry
    from mira.registry.access import operator_context

    access = operator_context()
    registry = get_registry()
    assert registry is not None
    out = RepositoryHistoryImporter(
        registry,
        GitReader(args.checkout),
        repo=args.repo,
        access=access,
        commit=args.commit,
        max_commits=args.max_commits,
        paths=args.path,
        max_seconds=args.max_seconds,
        max_units=args.max_units,
        incremental=args.incremental,
    ).run(apply=args.apply)
    json.dump(out, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
