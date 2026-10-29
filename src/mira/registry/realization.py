"""Derive realization from the GitHub events Mira already receives.

``pull_requests`` rows carry no head branch, so this reads the webhook payload
directly. GitLab and Forgejo are not wired yet.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_PR_ACTIONS = ("opened", "reopened", "ready_for_review", "synchronize", "closed")


def observe_github(event: str, payload: dict[str, Any], registry: Any = None) -> list[str]:
    """Apply one GitHub webhook to the registry. Never raises."""
    try:
        if registry is None:
            from mira.registry import get_registry

            registry = get_registry(create=False)
            if registry is None:
                return []
        repo = (payload.get("repository") or {}).get("full_name") or ""
        if event == "push":
            ref = str(payload.get("ref") or "")
            if payload.get("deleted") or not ref.startswith("refs/heads/") or not repo:
                return []
            return registry.on_push(repo, ref.removeprefix("refs/heads/"))
        if event == "pull_request" and payload.get("action") in _PR_ACTIONS:
            pr = payload.get("pull_request") or {}
            head = pr.get("head") or {}
            head_repo = (head.get("repo") or {}).get("full_name") or repo
            branch = head.get("ref") or ""
            if not (head_repo and branch):
                return []
            return registry.on_pull_request(
                head_repo,
                branch,
                action=payload["action"],
                merged=bool(pr.get("merged")),
                pr_url=pr.get("html_url") or "",
                merge_commit=pr.get("merge_commit_sha") or "",
            )
    except Exception as exc:
        logger.warning("Registry realization skipped %s event: %s", event, exc)
    return []
