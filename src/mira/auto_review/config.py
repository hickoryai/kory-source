"""Fixed trial settings for the secondary reviewer.

The noise filter does not read the dashboard override blob
(``global_review_overrides``). The path stays off until the service env names
the same single repository as the GitHub App installation. Nothing here turns
the flag on.
"""

from __future__ import annotations

import os
import re
from typing import Any

from mira.config import FilterConfig, LLMConfig, MiraConfig, OverlapConfig, ReviewConfig

ENV_FLAG = "KORY_SOURCE_AUTO_REVIEW"
ENV_REPOS = "KORY_SOURCE_AUTO_REVIEW_REPOS"
ENV_TOKEN = "KORY_SOURCE_AUTO_REVIEW_TOKEN"
ENV_ACCOUNT = "KORY_SOURCE_AUTO_REVIEW_ACCOUNT"
ENV_SOURCE_URL = "KORY_SOURCE_URL"
PAUSE_SETTING = "auto_review_paused"

# One owner/repo. The organization name is not compiled in: the service env
# names the repository at enablement, and the GitHub App installation must
# list that same single repository or nothing is queued.
_ONE_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")

# The seat worker runs here: the workstation that holds the logged-in router
# account pool. It is not started by this process.
WORKER_HOST = "dev-kory-ai"

# The model stays this name. The account id is local to the worker's router
# and is supplied at enablement. Nothing here fills in another account or model.
MODEL_NAME = "grok-4.7"
_ACCOUNT = re.compile(r"grok@[0-9a-f]{32}\Z")
CALL_TIMEOUT_SECONDS = 1800
REVIEW_EVENT = "COMMENT"
SECONDARY_LABEL = "Kory Source (secondary reviewer)"

# Operator steps. Pause and rollback are flag and process changes; they do
# not require reconstructing this code. None of these steps run at import.
ENABLEMENT_STEPS: tuple[str, ...] = (
    "Build and publish a new image from this change. Image 0.11.0 does not "
    "contain this path and must not be enabled. Publish only after the "
    "public-image audit passes for that exact artifact and a maintainer with "
    "publishing authority has approved it.",
    "Deploy that image. Leave KORY_SOURCE_AUTO_REVIEW unset until the remaining steps are done.",
    "Install the GitHub App on the one repository chosen for the comparison. "
    "Confirm the installation lists that one repository and the dashboard has "
    "it registered to that installation.",
    "On the Kory Source service set KORY_SOURCE_AUTO_REVIEW=1, "
    "KORY_SOURCE_AUTO_REVIEW_REPOS to that one owner/repo, "
    "KORY_SOURCE_AUTO_REVIEW_TOKEN to a shared secret, and "
    "KORY_SOURCE_AUTO_REVIEW_ACCOUNT to the routeable grok@ account on the "
    "worker host. The model is grok-4.7. An unset account does not select "
    "another model. Restart the service.",
    "On dev-kory-ai (the workstation with the logged-in router account pool) "
    "start the seat worker with KORY_SOURCE_AUTO_REVIEW=1, the same token, "
    "KORY_SOURCE_URL pointing at the service, and the local relay credentials. "
    "Run `python -m mira.auto_review.worker`. A worker outage stops Kory only.",
    "Pause without a deploy: as an admin, POST /api/auto-review/switch with "
    '{"paused": true}. New heads are not queued and a review that has not '
    "posted yet does not post. Stop the worker and unset "
    "KORY_SOURCE_AUTO_REVIEW for a full stop.",
    "Rollback the same way: pause, stop the worker on dev-kory-ai, and unset "
    "the flag. No code reconstruction and no branch-protection change. Posted "
    "comments stay on the frozen commit they reviewed. Greptile is untouched.",
)


def flag_on() -> bool:
    """True only when the process env explicitly enables the path."""
    return os.environ.get(ENV_FLAG) == "1"


def pinned_model() -> str | None:
    """The one review model, or None when the account is not configured.

    The value is ``grok@<32 hex>/<MODEL_NAME>``. ``auto``, a different model,
    and a missing account all return None. This does not substitute another id.
    """
    account = os.environ.get(ENV_ACCOUNT, "").strip()
    if _ACCOUNT.fullmatch(account) is None:
        return None
    return f"{account}/{MODEL_NAME}"


def env_repos() -> set[str]:
    raw = os.environ.get(ENV_REPOS, "")
    return {part.strip() for part in raw.split(",") if part.strip()}


def selected_repo() -> str | None:
    """The one repository named by the service env, or None when it is not exactly one."""
    repos = env_repos()
    if len(repos) != 1:
        return None
    name = next(iter(repos))
    if _ONE_REPO.fullmatch(name) is None:
        return None
    return name


def paused(db: Any) -> bool:
    try:
        return db.get_setting(PAUSE_SETTING) == "1"
    except Exception:
        return False


def set_paused(db: Any, value: bool) -> None:
    db.set_setting(PAUSE_SETTING, "1" if value else "0")


def trial_filter() -> FilterConfig:
    """Round-1 trial filter. Blockers and warnings still bypass the cap of 5."""
    return FilterConfig(
        confidence_threshold=0.7,
        max_comments=5,
        min_severity="suggestion",
    )


def trial_mira_config() -> MiraConfig:
    """Engine settings for one frozen head.

    Built here, not via ``load_config()``, so a dashboard override cannot
    lower the floor, raise the cap, or turn nitpicks back on. Refuses to build
    a config when the account env does not name a grok account.
    """
    model = pinned_model()
    if model is None:
        raise RuntimeError(
            f"{ENV_ACCOUNT} must be a grok@ account id. The model stays {MODEL_NAME}."
        )
    llm = LLMConfig(
        provider="router-acp",
        model=model,
        review_model=model,
        indexing_model=model,
        security_model=model,
        router_acp_conversations=True,
        router_acp_timeout_seconds=CALL_TIMEOUT_SECONDS,
        router_acp_priority="review",
    )
    review = ReviewConfig(
        auto_resolve_conversations=False,
        pr_summary="disable",
        overlap=OverlapConfig(enabled=False),
    )
    return MiraConfig(llm=llm, filter=trial_filter(), review=review)


def installation_repo_names(db: Any, installation_id: int) -> set[str]:
    if not installation_id:
        return set()
    names: set[str] = set()
    for record in db.list_repos():
        if record.platform == "github" and int(record.installation_id or 0) == installation_id:
            names.add(f"{record.owner}/{record.repo}")
    return names


def trial_repo_selected(owner: str, repo: str) -> bool:
    """The service env names this one repository and no other."""
    return selected_repo() == f"{owner}/{repo}"


def ownership(db: Any, owner: str, repo: str, installation_id: int) -> str | None:
    """How a pull_request delivery for this repo is handled.

    ``None`` leaves today's webhook reviewer in place. ``enqueue`` queues one
    secondary review and must not call that reviewer. ``ignore`` means the
    flag is on and this delivery is the one selected repository, but the
    pause bit or the installation does not match, so neither reviewer runs.
    """
    if not flag_on():
        return None
    chosen = selected_repo()
    if chosen is None or f"{owner}/{repo}" != chosen:
        return None
    if paused(db) or pinned_model() is None or not installation_is_only_trial(db, installation_id):
        return "ignore"
    return "enqueue"


def installation_is_only_trial(db: Any, installation_id: int) -> bool:
    chosen = selected_repo()
    if chosen is None:
        return False
    return installation_repo_names(db, installation_id) == {chosen}


def review_preamble(head_sha: str) -> str:
    return f"**{SECONDARY_LABEL}**\n\nReviewed head `{head_sha}`"


def switch_view(db: Any) -> dict[str, Any]:
    """Admin view of the env and pause gate.

    ``effective`` is that gate. Each delivery also requires the GitHub App
    installation to list only the trial repository.
    """
    enabled = flag_on()
    is_paused = paused(db)
    repos = sorted(env_repos())
    chosen = selected_repo()
    model = pinned_model()
    return {
        "env_enabled": enabled,
        "paused": is_paused,
        "effective": enabled and not is_paused and chosen is not None and model is not None,
        "allowlist": [chosen] if chosen is not None else [],
        "env_repos": repos,
        "worker_host": WORKER_HOST,
        "model": model,
        "timeout_seconds": CALL_TIMEOUT_SECONDS,
        "review_event": REVIEW_EVENT,
        "conversations": True,
        "registry_context": False,
    }
