"""One-way import of Mira's approved learned rules as registry ``rule`` objects.

Reads through ``IndexStore.list_active_learned_rules()`` (a SELECT of
``active = 1 AND status = 'approved'``). Nothing here writes to Mira's tables;
``rules.py``, ``analysis/feedback.py``, and the learnings UI are untouched.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

from mira.registry.model import Actor
from mira.registry.service import Registry

logger = logging.getLogger(__name__)

SOURCE_SYSTEM = "mira_learned_rule"


def source_url(platform: str, owner: str, repo: str, rule_id: int) -> str:
    return f"mira://{platform}/{owner}/{repo}/learned-rules/{rule_id}"


def import_learned_rules(
    registry: Registry,
    repos: Iterable[tuple[str, str, str]],
    open_store: Callable[[str, str, str], Any] | None = None,
) -> dict[str, int]:
    """Mirror approved learned rules for each ``(platform, owner, repo)``.

    A rule that is no longer approved and active in Mira is retired in the
    registry, with ``sync_status = source_removed``.
    """
    if open_store is None:
        from mira.index.store import IndexStore

        def open_store(platform: str, owner: str, repo: str) -> Any:
            return IndexStore.open(owner, repo, platform=platform)

    totals = {"created": 0, "updated": 0, "unchanged": 0, "retired": 0}
    for platform, owner, repo in repos:
        try:
            store = open_store(platform, owner, repo)
        except Exception as exc:
            logger.debug("learned-rule import skipped %s/%s: %s", owner, repo, exc)
            continue
        try:
            rules = store.list_active_learned_rules()
        finally:
            store.close()
        seen: set[str] = set()
        for rule in rules:
            sid = f"{platform}/{owner}/{repo}/{rule.id}"
            seen.add(sid)
            actor = (
                Actor("human", rule.created_by)
                if rule.created_by
                else Actor("agent", "mira-learning")
            )
            item = {
                "source_id": sid,
                # Version on updated_at so an edited rule re-imports and an unedited one no-ops.
                "source_version": int(rule.updated_at * 1000) or 1,
                "kind": "rule",
                "title": rule.rule_text[:120],
                "body": rule.rule_text,
                "summary": " ".join(p for p in (rule.category, rule.path_pattern) if p),
                "repo": f"{owner}/{repo}",
                "scope": "repo",
                "source_url": source_url(platform, owner, repo, rule.id),
            }
            [res] = registry.import_items(
                SOURCE_SYSTEM, [item], actor, status="approved", basis="review_learning"
            )
            totals[res["outcome"]] = totals.get(res["outcome"], 0) + 1
            obj = registry.get(res["object_id"]) if "object_id" in res else None
            if obj and obj["status"] == "retired" and obj["status_reason"] == "source_removed":
                obj.update(status="approved", status_reason="", sync_status="synced")
                registry._write(obj)
        prefix = f"{platform}/{owner}/{repo}/"
        for obj in registry._select(
            "source_system = ? AND source_id LIKE ? AND status = 'approved'",
            (SOURCE_SYSTEM, prefix + "%"),
        ):
            if obj["source_id"] not in seen:
                obj.update(
                    status="retired", status_reason="source_removed", sync_status="source_removed"
                )
                registry._write(obj)
                totals["retired"] += 1
    return totals
