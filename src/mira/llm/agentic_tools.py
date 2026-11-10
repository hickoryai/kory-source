"""Tools the reviewer LLM can call during review (`read_file`, `grep_repo`).

On unindexed repos the tools cover what JIT pre-fetch can't reach
(Java/Go import resolution); on indexed repos they let the model trace
callers and dispatch points beyond the pre-fetched index context.

This module owns the tool *schemas* and a per-review *executor* that
dispatches calls, caches results, and hard-caps total output. The agentic
loop itself lives in ``passes.py:agentic_review_loop``.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass, field

from mira.index.context import SourceFetcher

logger = logging.getLogger(__name__)


READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read a file from the repository at the PR head. Use this to verify "
            "cross-file claims (does function X exist? what does the caller pass?) "
            "before filing a comment. Prefer reading specific files over guessing. "
            "Returns the file content with line numbers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Repo-relative path of the file to read, e.g. `src/auth/middleware.py`."
                    ),
                },
            },
            "required": ["path"],
        },
    },
}


GREP_REPO_TOOL = {
    "type": "function",
    "function": {
        "name": "grep_repo",
        "description": (
            "Search the repo for files whose path or content matches a pattern. "
            "Use this to find where a symbol is defined or called when you don't "
            "already know the file path. Returns up to 30 matching paths (path "
            "search) or up to 30 line-level hits (content search)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "What to look for. Treated as a regex against file content "
                        "unless `path_only` is true. Keep it specific — short or "
                        "common patterns return too many matches and get capped."
                    ),
                },
                "path_glob": {
                    "type": ["string", "null"],
                    "description": (
                        "Optional glob to restrict the search, e.g. `**/*.java` "
                        "or `src/auth/*`. Defaults to all files."
                    ),
                },
                "path_only": {
                    "type": ["boolean", "null"],
                    "description": (
                        "If true, only match paths (don't read file contents). "
                        "Faster and cheaper. Use when you're hunting for a file "
                        "by name."
                    ),
                },
            },
            "required": ["pattern"],
        },
    },
}


# Hard caps to keep tool output from blowing up the prompt or the API budget.
_MAX_FILE_BYTES = 12_000  # ~3k tokens of source; truncate larger files
_MAX_GREP_HITS = 30
_MAX_GREP_BYTES_PER_HIT = 240
_MAX_TOTAL_OUTPUT_BYTES = 50_000  # all tool calls in one review combined
# Each scanned file is a network round-trip via source_fetcher; keep tight.
_MAX_GREP_FILES = 15


# A model-supplied glob like `{a,b}{c,d}...` expands multiplicatively; refuse
# anything past this many concrete patterns instead of enumerating it.
_MAX_GLOB_ALTERNATIVES = 64


class GlobTooBroad(ValueError):
    """The glob expands to more than ``_MAX_GLOB_ALTERNATIVES`` patterns."""


def _cap(patterns: list[str]) -> list[str]:
    if len(patterns) > _MAX_GLOB_ALTERNATIVES:
        raise GlobTooBroad(f"more than {_MAX_GLOB_ALTERNATIVES} alternatives")
    return patterns


def _expand_braces(glob: str) -> list[str]:
    """`**/*.{py,ts}` -> [`**/*.py`, `**/*.ts`]. fnmatch has no brace syntax.

    Groups are expanded left to right and the running total is checked after
    each one, so an oversized glob fails before it is enumerated. Nested
    braces are not supported; an unmatched brace is a literal.
    """
    out = [""]
    rest = glob
    while True:
        start = rest.find("{")
        end = rest.find("}", start + 1)
        if start < 0 or end < 0:
            return [o + rest for o in out]
        head, alts = rest[:start], rest[start + 1 : end].split(",")
        out = _cap([o + head + alt for o in out for alt in alts])
        rest = rest[end + 1 :]


def _zero_dir_variants(pattern: str) -> list[str]:
    """Each `**/` independently matches zero or more directories.

    fnmatch's `*` already crosses `/`, so `**/` matches one or more directories;
    adding the variant with that segment removed covers zero. Every subset of
    `**/` segments is dropped, so `a/**/b/**/*.py` also matches `a/b/x/y.py`.
    """
    parts = pattern.split("**/")
    variants = [parts[0]]
    for part in parts[1:]:
        variants = _cap([v + sep + part for v in variants for sep in ("**/", "")])
    return variants


def _glob_patterns(glob: str) -> list[str]:
    """The concrete fnmatch patterns for `glob`: `{a,b}` alternation plus
    zero-directory `**/` segments. Raises ``GlobTooBroad`` past the cap."""
    return _cap([v for alt in _expand_braces(glob) for v in _zero_dir_variants(alt)])


@dataclass
class AgenticToolExecutor:
    """Per-review tool dispatcher with caching and an output-size budget.

    Construct one per chunk review. The `repo_tree` is captured once
    (already fetched for JIT) so `grep_repo` doesn't keep re-listing the
    repo. The `source_fetcher` already has its own per-path cache, so
    repeated `read_file` calls don't re-hit GitHub.
    """

    source_fetcher: SourceFetcher
    repo_tree: list[str]
    bytes_used: int = 0
    _content_cache: dict[str, str | None] = field(default_factory=dict)
    call_log: list[dict] = field(default_factory=list)

    async def execute(self, name: str, args: dict) -> str:
        """Dispatch a tool call. Returns the tool result as a string.

        Always returns *something* — errors are reported back to the LLM
        rather than raised, so the model can recover (e.g. by trying a
        different path). The string is what gets fed back into the next
        LLM hop as a `tool` message.
        """
        arg = (args or {}).get("path") or (args or {}).get("pattern") or ""
        self.call_log.append({"tool": name, "arg": arg})

        if self.bytes_used >= _MAX_TOTAL_OUTPUT_BYTES:
            return "[tool budget exhausted — no more tool calls accepted; submit your review now]"

        try:
            if name == "read_file":
                path = (args or {}).get("path") or ""
                if not path:
                    return "[error: missing `path` argument]"
                result = await self._read_file(path)
            elif name == "grep_repo":
                pattern = (args or {}).get("pattern") or ""
                if not pattern:
                    return "[error: missing `pattern` argument]"
                path_glob = (args or {}).get("path_glob") or None
                path_only = bool((args or {}).get("path_only") or False)
                result = await self._grep_repo(pattern, path_glob, path_only)
            else:
                return f"[error: unknown tool `{name}`]"
        except Exception as exc:
            logger.debug("Tool %s failed: %s", name, exc)
            return f"[error executing `{name}`: {exc}]"

        self.bytes_used += len(result)
        return result

    async def _read_file(self, path: str) -> str:
        # Don't waste tokens hunting for files we know don't exist.
        if self.repo_tree and path not in self.repo_tree:
            close = [p for p in self.repo_tree if path.lower() in p.lower()][:5]
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            return f"[file not found at PR head: `{path}`.{hint}]"

        if path in self._content_cache:
            content = self._content_cache[path]
        else:
            content = await self.source_fetcher.fetch(path)
            self._content_cache[path] = content

        if content is None:
            return f"[failed to read `{path}`]"

        truncated = False
        if len(content) > _MAX_FILE_BYTES:
            content = content[:_MAX_FILE_BYTES]
            truncated = True

        # Number lines so the LLM can cite them by number when filing comments.
        numbered = "\n".join(f"{i + 1:>5}  {line}" for i, line in enumerate(content.split("\n")))
        suffix = (
            f"\n... [truncated; file is longer than {_MAX_FILE_BYTES} bytes]" if truncated else ""
        )
        return f"`{path}`:\n```\n{numbered}{suffix}\n```"

    async def _grep_repo(self, pattern: str, path_glob: str | None, path_only: bool) -> str:
        if not self.repo_tree:
            return "[grep unavailable: repo tree not loaded]"

        candidates: list[str]
        if path_glob:
            try:
                patterns = _glob_patterns(path_glob)
            except GlobTooBroad:
                return (
                    f"[invalid path_glob `{path_glob}`: it expands to more than "
                    f"{_MAX_GLOB_ALTERNATIVES} patterns; use fewer `{{a,b}}` alternatives]"
                )
            candidates = [p for p in self.repo_tree if any(fnmatch.fnmatch(p, g) for g in patterns)]
        else:
            candidates = list(self.repo_tree)

        if path_only:
            try:
                rx = re.compile(pattern)
            except re.error:
                # Treat as substring on regex error.
                path_hits = [p for p in candidates if pattern in p][:_MAX_GREP_HITS]
            else:
                path_hits = [p for p in candidates if rx.search(p)][:_MAX_GREP_HITS]
            if not path_hits:
                return f"[no path matches for `{pattern}`]"
            return "Path matches:\n" + "\n".join(f"- `{p}`" for p in path_hits)

        try:
            rx = re.compile(pattern)
        except re.error as exc:
            return f"[invalid regex `{pattern}`: {exc}]"

        # Skip giant binaries / lockfiles / vendored dirs we'd never want to read.
        skip_exts = (
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".pdf",
            ".zip",
            ".gz",
            ".tar",
            ".woff",
            ".woff2",
            ".ttf",
            ".eot",
            ".ico",
            ".svg",
            ".map",
            ".lock",
        )
        skip_dirs = ("node_modules/", "vendor/", "dist/", "build/", ".git/")

        eligible = [
            c
            for c in candidates
            if not any(c.endswith(ext) for ext in skip_exts) and not any(d in c for d in skip_dirs)
        ]
        hits: list[str] = []
        files_scanned = 0

        for cand in eligible:
            if len(hits) >= _MAX_GREP_HITS or files_scanned >= _MAX_GREP_FILES:
                break
            files_scanned += 1
            content = self._content_cache.get(cand)
            if content is None and cand not in self._content_cache:
                content = await self.source_fetcher.fetch(cand)
                self._content_cache[cand] = content
            if not content:
                continue

            for lineno, line in enumerate(content.split("\n"), start=1):
                if rx.search(line):
                    snippet = line.strip()
                    if len(snippet) > _MAX_GREP_BYTES_PER_HIT:
                        snippet = snippet[:_MAX_GREP_BYTES_PER_HIT] + "…"
                    hits.append(f"{cand}:{lineno}: {snippet}")
                    if len(hits) >= _MAX_GREP_HITS:
                        break

        # Stopped by the file cap, not by running out of files: say so, so the
        # model never reads an unscanned remainder as evidence of absence.
        incomplete = len(hits) < _MAX_GREP_HITS and files_scanned < len(eligible)
        if not hits:
            if incomplete:
                return (
                    f"[no content matches for `{pattern}` in the {files_scanned} files scanned, "
                    f"out of {len(eligible)} candidates. The search is INCOMPLETE and is not "
                    "evidence of absence: narrow `path_glob` or read_file the likely file]"
                )
            return f"[no content matches for `{pattern}` (scanned {files_scanned} files)]"
        prefix = f"Content matches (scanned {files_scanned} files):"
        if incomplete:
            prefix = (
                f"Content matches (scanned {files_scanned} of {len(eligible)} candidate files; "
                "search incomplete):"
            )
        if len(hits) == _MAX_GREP_HITS:
            prefix += f" showing first {_MAX_GREP_HITS}"
        return prefix + "\n" + "\n".join(hits)


AGENTIC_TOOLS = [READ_FILE_TOOL, GREP_REPO_TOOL]
