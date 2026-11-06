"""Resolve historical decision anchors to a later commit.

An adapter over existing read APIs; it does not change the indexer or its
anchor algorithm. Order, mirroring ``mira.index.anchors``:

1. a stable index ``anchor_id`` the index still resolves;
2. the same path and symbol at the target commit (``exact``);
3. a git rename (``git diff -M``) to a path that still holds the symbol
   (``renamed``);
4. the file exists but the symbol is gone (``pending``: needs a person, or a
   similarity relink the index proposes; never auto-confirmed);
5. nothing matches (``dangling``).

A path-only anchor (Markdown section, unsupported language) resolves by path
and rename only. Historical locations are never overwritten: the result is a
new view of the stored evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from mira.registry.repository_history import GitReader, language_of

_INDEX_STATUS = {"active": "exact", "renamed": "renamed", "relinked": "fingerprint"}


def _has_symbol(text: str, path: str, qualified_name: str) -> bool:
    from mira.index.extract import extract_symbols

    leaf = qualified_name.rsplit(".", 1)[-1]
    for sym in extract_symbols(text, language_of(path)):
        if qualified_name in (sym.qualified_name, sym.name) or leaf == sym.name:
            return True
    return False


def resolve(
    anchor: dict,
    *,
    target_commit: str,
    reader: GitReader | None = None,
    anchor_lookup: Callable[[str, str], dict | None] | None = None,
    renames: dict[str, str] | None = None,
) -> dict:
    """Where an anchor observed at ``anchor['observed_commit']`` is at ``target_commit``."""
    repo = anchor.get("repository") or anchor.get("repo") or ""
    path = anchor.get("path") or ""
    qname = anchor.get("qualified_name") or ""
    base = {
        "observed_commit": anchor.get("observed_commit") or "",
        "observed_path": path,
        "observed_qualified_name": qname,
        "target_commit": target_commit,
    }
    if anchor.get("anchor_id") and anchor_lookup is not None:
        try:
            found = anchor_lookup(repo, anchor["anchor_id"])
        except Exception:
            found = None
        if found and found.get("resolution_status") in _INDEX_STATUS:
            return {
                **base,
                "status": _INDEX_STATUS[found["resolution_status"]],
                "method": "index_anchor",
                "path": found.get("path") or path,
                "qualified_name": found.get("qualified_name") or qname,
                "confidence": "high",
            }
        if found and found.get("resolution_status") == "pending":
            return {
                **base,
                "status": "pending",
                "method": "index_anchor",
                "path": path,
                "qualified_name": qname,
                "confidence": "low",
            }
    if reader is None:
        return {
            **base,
            "status": "pending",
            "method": "no_checkout",
            "path": path,
            "qualified_name": qname,
            "confidence": "unknown",
        }
    if renames is None:
        from pathlib import Path

        from mira.index.anchors import git_rename_map

        renames = git_rename_map(Path(reader.checkout), base["observed_commit"], target_commit)
    candidates = [(path, "exact")]
    if renames.get(path):
        candidates.append((renames[path], "renamed"))
    file_seen = False
    for cand, how in candidates:
        text = reader.show(target_commit, cand)
        if text is None:
            continue
        file_seen = True
        if not qname:
            return {
                **base,
                "status": how if how == "renamed" else "exact",
                "method": "path" if how == "exact" else "git_rename",
                "path": cand,
                "qualified_name": "",
                "confidence": "medium",
            }
        if not language_of(cand):
            return {
                **base,
                "status": "unsupported",
                "method": "path",
                "path": cand,
                "qualified_name": qname,
                "confidence": "low",
            }
        if _has_symbol(text, cand, qname):
            return {
                **base,
                "status": how,
                "method": "symbol" if how == "exact" else "git_rename",
                "path": cand,
                "qualified_name": qname,
                "confidence": "high",
            }
    if file_seen:
        return {
            **base,
            "status": "pending",
            "method": "symbol_missing",
            "path": path,
            "qualified_name": qname,
            "confidence": "low",
        }
    return {
        **base,
        "status": "dangling",
        "method": "none",
        "path": path,
        "qualified_name": qname,
        "confidence": "low",
    }


def touched(anchors: Iterable[dict], changed: Iterable[dict]) -> list[tuple[dict, dict]]:
    """Anchors whose old or new path a PR changes: ``(anchor, changed_file)`` pairs.

    ``changed`` items follow GitHub's PR files API: ``filename``, ``status``,
    ``previous_filename``. Deleted and renamed files match on their old path.
    """
    by_path: dict[str, dict] = {}
    for f in changed:
        by_path[f.get("filename") or ""] = f
        if f.get("previous_filename"):
            by_path[f["previous_filename"]] = f
    out = []
    for a in anchors:
        for key in (a.get("resolved_path"), a.get("path")):
            if key and key in by_path:
                out.append((a, by_path[key]))
                break
    return out


def summarize(results: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return counts
