"""Workstation indexer.

Walks a checkout, extracts symbols, and resolves anchors with ``git diff -M``.
The Cloud Run job uses ``index_repo`` instead, because that image has no git
checkout. Both paths share the same fingerprint and anchor code.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from mira.config import load_config
from mira.index.indexer import (
    _annotate_and_store,
    _content_hash,
    _finish_index,
    _language_from_path,
    _should_index,
)
from mira.index.store import FileSummary, IndexStore

logger = logging.getLogger(__name__)

# Repositories the Cloud Run job indexes when no --repo flag is passed.
# A workstation run passes owner/repo=/checkout so rename detection can run.
DEFAULT_INDEX_REPOS = (
    "hickoryai/hickory-ai",
    "hickoryai/goose",
    "zaneclaes/router-acp",
    "hickoryai/kory-source",
)


@dataclass
class IndexStats:
    owner: str
    repo: str
    files: int
    symbols: int
    seconds: float
    commit: str


# Dependency and build directories are not this repo's source. A checkout
# can also hold files the process is not allowed to stat, so the walk prunes
# these names anywhere in the tree and skips an unreadable path.
_SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".next",
        ".nuxt",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "target",
        "vendor",
        "venv",
    }
)


def _iter_files(checkout: Path):
    """Yield regular files, without entering dependency or build trees."""

    def _on_error(exc: OSError) -> None:
        logger.warning("skip %s: %s", exc.filename, exc)

    for dirpath, dirnames, filenames in os.walk(checkout, followlinks=False, onerror=_on_error):
        dirnames[:] = [name for name in dirnames if name not in _SKIP_DIR_NAMES]
        for name in filenames:
            path = Path(dirpath) / name
            try:
                if not path.is_file():
                    continue
            except OSError as exc:
                logger.warning("skip %s: %s", path, exc)
                continue
            yield path


def _tracked_paths(checkout: Path) -> set[str] | None:
    """Return tracked paths, or None when this directory is not a git checkout.

    The GitHub indexer reads the git tree. A workstation checkout should index
    that same set. Caches, virtualenvs, and toolchain checkouts on disk are
    not part of the repository.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(checkout), "ls-files", "-z", "--"],
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("git ls-files failed: %s", exc)
        return None
    if proc.returncode != 0:
        return None
    return {item.decode("utf-8", "surrogateescape") for item in proc.stdout.split(b"\0") if item}


def _iter_tracked(checkout: Path, tracked: set[str]):
    """Yield tracked files that can be read."""
    for rel in sorted(tracked):
        path = checkout / rel
        try:
            if not path.is_file():
                continue
        except OSError as exc:
            logger.warning("skip %s: %s", rel, exc)
            continue
        yield path


def head_commit(checkout: Path) -> str:
    """Return HEAD, or an empty string when git cannot answer."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("git rev-parse failed: %s", exc)
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def index_checkout(
    owner: str,
    repo: str,
    checkout: Path,
    store: IndexStore,
    *,
    commit: str = "",
) -> IndexStats:
    """Index one local checkout. Symbols come from the extractor, not an LLM.

    A git checkout contributes its tracked files only.
    """
    started = time.perf_counter()
    checkout = checkout.resolve()
    if not commit:
        commit = head_commit(checkout)
    config = load_config()
    max_file_size = config.index.max_file_size
    seen: set[str] = set()
    indexed = 0
    tracked = _tracked_paths(checkout)
    files = _iter_tracked(checkout, tracked) if tracked is not None else _iter_files(checkout)
    for path in files:
        rel = path.relative_to(checkout).as_posix()
        if not _should_index(rel, config.filter.exclude_patterns):
            continue
        try:
            raw = path.read_bytes()
        except OSError as exc:
            logger.warning("skip %s: %s", rel, exc)
            continue
        if b"\x00" in raw[:1024]:
            continue
        if max_file_size and len(raw) > max_file_size:
            continue
        content = raw.decode("utf-8", errors="replace")
        seen.add(rel)
        summary = FileSummary(
            path=rel,
            language=_language_from_path(rel),
            summary="",
            content_hash=_content_hash(content),
            loc=content.count("\n") + (0 if content.endswith("\n") or not content else 1),
        )
        _annotate_and_store(store, summary, content, commit, preserve_existing=True)
        indexed += 1

    removed = store.all_paths() - seen
    if removed:
        store.remove_paths(sorted(removed))
    _finish_index(store, owner, repo, commit, str(checkout), indexed)
    elapsed = time.perf_counter() - started
    return IndexStats(
        owner=owner,
        repo=repo,
        files=len(store.all_paths()),
        symbols=store.count_symbols(),
        seconds=elapsed,
        commit=commit,
    )
