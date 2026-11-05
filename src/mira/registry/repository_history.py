"""Read-only git history import: Markdown, code comments/docstrings, commit messages.

HAI-12825. From a local checkout of a manifest repository, at a frozen commit:

- every Markdown file's revisions along its history (bounded by
  ``max_commits``), so a passage that was later rewritten or deleted is kept;
- comments and docstrings in supported languages at the frozen commit,
  anchored to the enclosing symbol from the deterministic extractor;
- commit messages, whose trailers can state an agent co-author, model, or
  invoker.

Git author and committer labels are stored as source accounts. Neither is
assumed to be the decision author or the invoking person.

``GitReader`` runs a fixed allowlist of read-only git subcommands. Nothing
here writes to the repository.

Run: ``python -m mira.registry.repository_history import --repo owner/name
--checkout PATH [--commit SHA] [--apply]`` with ``REGISTRY_IMPORT_OPERATOR``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from mira.registry import attribution as attr
from mira.registry.access import AccessContext, AccessDenied, require_import
from mira.registry.imports import ImportRuns, ReadOnlyViolation, extract_decisions, materialize
from mira.registry.provenance import ProvenanceStore, SourceRevision

logger = logging.getLogger(__name__)

SOURCE_SYSTEM = "git"
IMPORTER = "kory-source.repository-history"
IMPORTER_VERSION = "1"
_READ_ONLY = {"log", "show", "ls-tree", "rev-parse", "cat-file", "diff"}
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
}
_BLOCK = re.compile(r"/\*.*?\*/", re.S)
_DOCSTRING = re.compile(r"(?P<q>\"\"\"|''')(?P<body>.*?)(?P=q)", re.S)
_SEP = "\x1e"


def language_of(path: str) -> str:
    return _EXT.get(Path(path).suffix.lower(), "")


class GitReader:
    """Allowlisted read-only git commands against one checkout."""

    def __init__(self, checkout: str | Path) -> None:
        self.checkout = Path(checkout)

    def git(self, *args: str) -> str:
        if not args or args[0] not in _READ_ONLY:
            raise ReadOnlyViolation(f"git {args[0] if args else ''} is not a read-only command")
        proc = subprocess.run(
            ["git", "-C", str(self.checkout), *args],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"git {args[0]} failed: {proc.stderr.strip()[:300]}")
        return proc.stdout

    def rev_parse(self, rev: str) -> str:
        return self.git("rev-parse", "--verify", f"{rev}^{{commit}}").strip()

    def files(self, commit: str) -> list[str]:
        return [p for p in self.git("ls-tree", "-r", "--name-only", commit).splitlines() if p]

    def show(self, commit: str, path: str) -> str | None:
        try:
            return self.git("show", f"{commit}:{path}")
        except RuntimeError:
            return None

    def commits(
        self, commit: str, paths: list[str] | None = None, max_count: int = 200
    ) -> list[dict]:
        fmt = _SEP.join(["%H", "%an", "%ae", "%at", "%cn", "%ce", "%ct", "%B"]) + "\x1d"
        args = [
            "log",
            f"--max-count={max_count}",
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
        return out

    def path_at(self, commit: str, path: str) -> list[tuple[str, str]]:
        """``(commit, path)`` pairs for a file's history, following renames."""
        out = self.git("log", "--follow", "--name-only", "--format=%x1e%H", commit, "--", path)
        pairs: list[tuple[str, str]] = []
        for block in out.split("\x1e"):
            lines = [line for line in block.strip().splitlines() if line]
            if len(lines) >= 2:
                pairs.append((lines[0], lines[-1]))
        return pairs


def comment_spans(text: str, language: str) -> list[tuple[int, int, int]]:
    """``(start, end, line)`` of each comment or docstring body in ``text``."""
    spans: list[tuple[int, int, int]] = []
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
            spans.append((m.start() + 2, m.end() - 2, text.count("\n", 0, m.start()) + 1))
    if language == "python":
        for m in _DOCSTRING.finditer(text):
            spans.append((m.start("body"), m.end("body"), text.count("\n", 0, m.start()) + 1))
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
        max_commits: int = 200,
    ) -> None:
        self.registry = registry
        self.reader = reader
        self.repo = repo
        self.access = access
        self.commit = commit
        self.max_commits = max_commits
        self.store = ProvenanceStore(registry)
        self.runs = ImportRuns(registry)

    def run(self, *, apply: bool = False) -> dict:
        policy = require_import(self.access)
        if not policy.admits_repository(self.repo):
            raise AccessDenied("the repository is not in the approved corpus manifest")
        frozen = self.reader.rev_parse(self.commit)
        files = self.reader.files(frozen)
        markdown = [p for p in files if p.lower().endswith(_MARKDOWN)]
        code = [p for p in files if language_of(p)]
        unsupported = len(files) - len(markdown) - len(code)
        if not apply:
            return {
                "apply": False,
                "commit": frozen,
                "markdown_files": len(markdown),
                "code_files": len(code),
                "unsupported_files": unsupported,
            }
        run = self.runs.start(
            self.access,
            source_system=SOURCE_SYSTEM,
            stream=f"repository:{self.repo}",
            importer=IMPORTER,
            importer_version=IMPORTER_VERSION,
            scope={"repo": self.repo, "commit": frozen, "max_commits": self.max_commits},
        )
        done = set((run["checkpoint"] or {}).get("done") or [])
        errors: list[str] = []
        units = [("md", p) for p in markdown] + [("code", p) for p in code] + [("log", "")]
        for kind, path in units:
            key = f"{kind}:{path}"
            if key in done:
                continue
            done.add(key)
            run["next_checkpoint"] = {"done": sorted(done)}
            try:
                with self.runs.page(run, self.access) as counts:
                    if kind == "md":
                        self._markdown(run, frozen, path, counts)
                    elif kind == "code":
                        self._code(run, frozen, path, counts)
                    else:
                        self._log(run, frozen, counts)
            except RuntimeError as exc:
                done.discard(key)
                errors.append(f"{key}: {exc}")
        with self.runs.page(run, self.access) as counts:
            counts["excluded_scope"] += unsupported
        final = self.runs.finish(
            run,
            complete=self.max_commits >= 10_000,
            errors=errors,
            unknown_totals=["history_before_max_commits"],
        )
        return {
            "run_id": final["run_id"],
            "status": final["status"],
            "commit": frozen,
            "coverage": final["coverage"],
        }

    def _git_attrs(self, eid: str, commit: dict) -> list[attr.Attribution]:
        found = attr.markers(commit.get("message") or "")
        return [*attr.git_author(eid, commit), *attr.from_markers(eid, found)]

    def _markdown(self, run: dict, frozen: str, path: str, counts: Counter[str]) -> None:
        history = self.reader.path_at(frozen, path)[: self.max_commits]
        meta = {c["sha"]: c for c in self.reader.commits(frozen, [path], self.max_commits)}
        for sha, old_path in history:
            text = self.reader.show(sha, old_path)
            if text is None:
                counts["unavailable"] += 1
                continue
            commit = meta.get(sha, {"sha": sha})
            rev = SourceRevision(
                SOURCE_SYSTEM,
                "git_markdown",
                f"{self.repo}:{path}",
                text,
                source_url=f"https://github.com/{self.repo}/blob/{sha}/{old_path}",
                source_labels={"path": old_path, "current_path": path},
                source_version=sha,
                source_created_at=float(commit.get("author_date") or 0),
                repository=self.repo,
                commit_sha=sha,
            )
            self._record(
                run,
                rev,
                self._git_attrs(rev.evidence_id, commit),
                counts,
                anchor_path=old_path,
                commit=sha,
                identity=f"{self.repo}:{path}",
            )

    def _code(self, run: dict, frozen: str, path: str, counts: Counter[str]) -> None:
        from mira.index.extract import extract_symbols

        text = self.reader.show(frozen, path)
        if text is None:
            counts["unavailable"] += 1
            return
        language = language_of(path)
        spans = comment_spans(text, language)
        if not spans:
            return
        candidates = []
        for start, end, _line in spans:
            for c in extract_decisions(text[start:end]):
                candidates.append(
                    c.__class__(
                        c.start + start,
                        c.end + start,
                        c.quote,
                        c.section,
                        f"{c.reason} in a code comment",
                    )
                )
        if not candidates:
            return
        commits = self.reader.commits(frozen, [path], 1)
        commit = commits[0] if commits else {"sha": frozen}
        rev = SourceRevision(
            SOURCE_SYSTEM,
            "git_source_file",
            f"{self.repo}:{path}",
            text,
            source_url=f"https://github.com/{self.repo}/blob/{frozen}/{path}",
            source_labels={"path": path, "language": language},
            source_version=frozen,
            source_created_at=float(commit.get("author_date") or 0),
            repository=self.repo,
            commit_sha=frozen,
        )
        symbols = extract_symbols(text, language)
        eid, created = self.store.record_revision(
            rev, corpus=(run["corpus_id"], run["policy_version"]), run_id=run["run_id"]
        )
        counts["fetched_versions"] += 1
        counts["stored_evidence" if created else "duplicate_evidence"] += 1
        attrs = self._git_attrs(eid, commit)
        self.store.record_attributions(attrs, corpus_id=run["corpus_id"])
        if not created:
            return
        for cand in candidates:
            line = text.count("\n", 0, cand.start) + 1
            enclosing = [s for s in symbols if s.start_line <= line <= s.end_line]
            sym = min(enclosing, key=lambda s: s.end_line - s.start_line) if enclosing else None
            anchor = {
                "repository": self.repo,
                "observed_commit": frozen,
                "path": path,
                "qualified_name": (sym.qualified_name or sym.name) if sym else "",
                "symbol_kind": sym.kind if sym else "",
                "line_start": line,
                "line_end": text.count("\n", 0, cand.end) + 1,
                "resolution_method": "deterministic_extractor" if sym else "path_only",
                "resolution_status": "exact" if sym else "path_only",
                "resolution_confidence": "high" if sym else "medium",
                "resolved_commit": frozen,
                "resolved_path": path,
                "resolved_qualified_name": (sym.qualified_name or sym.name) if sym else "",
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
            )

    def _log(self, run: dict, frozen: str, counts: Counter[str]) -> None:
        for commit in self.reader.commits(frozen, None, self.max_commits):
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
            self._record(run, rev, self._git_attrs(rev.evidence_id, commit), counts)

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
    ) -> None:
        eid, created = self.store.record_revision(
            rev, corpus=(run["corpus_id"], run["policy_version"]), run_id=run["run_id"]
        )
        counts["discovered"] += 1
        counts["fetched_versions"] += 1
        counts["stored_evidence" if created else "duplicate_evidence"] += 1
        self.store.record_attributions(attrs, corpus_id=run["corpus_id"])
        if not created:
            return
        candidates = extract_decisions(rev.original_text)
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
                identity=identity,
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mira.registry.repository_history")
    parser.add_argument("command", choices=("import",))
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--max-commits", type=int, default=200)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
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
    ).run(apply=args.apply)
    json.dump(out, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
