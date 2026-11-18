"""Fixed trial settings for the secondary reviewer.

The noise filter does not read the dashboard override blob
(``global_review_overrides``). The path stays off until
``KORY_SOURCE_AUTO_REVIEW=1``. While the flag is on, every pull request
delivered for a repository the GitHub App is installed on is queued.
Nothing here turns the flag on.
"""

from __future__ import annotations

import os
from typing import Any

from mira.config import FilterConfig, LLMConfig, MiraConfig, OverlapConfig, ReviewConfig

ENV_FLAG = "KORY_SOURCE_AUTO_REVIEW"
ENV_TOKEN = "KORY_SOURCE_AUTO_REVIEW_TOKEN"
ENV_SOURCE_URL = "KORY_SOURCE_URL"
ENV_WORKER_HOST = "KORY_SOURCE_WORKER_HOST"
PAUSE_SETTING = "auto_review_paused"

# No organization default. Deployments name the seat-worker machine at runtime.
GENERIC_WORKER_HOST = "the workstation with the logged-in router account pool"

# The request stays unpinned. The router selects the seat from the skill line.
DEFAULT_ROLE = "opinion"
ROLE_SKILL = {"opinion": "review-opinion", "gate": "review-gate"}
# Later-version extra opinion. Candidates are Grok only, so a full gate is
# not duplicated when Grok has no capacity.
GROK_ONLY_SKILL = "review-grok"
ALLOWED_SKILLS = frozenset({*ROLE_SKILL.values(), GROK_ONLY_SKILL})
CALL_TIMEOUT_SECONDS = 1800
REVIEW_EVENT = "COMMENT"
SECONDARY_LABEL = "Kory Source (secondary reviewer)"


def worker_host() -> str:
    """Seat-worker host from the process environment.

    Unset or blank stays generic. This process does not start the worker.
    """
    return os.environ.get(ENV_WORKER_HOST, "").strip() or GENERIC_WORKER_HOST


def _worker_place() -> str:
    host = worker_host()
    if host == GENERIC_WORKER_HOST:
        return host
    return f"{host} (the workstation with the logged-in router account pool)"


def enablement_steps() -> tuple[str, ...]:
    """Operator steps. Pause and rollback are flag and process changes.

    They do not require reconstructing this code. None of these steps run at
    import. The seat-worker place is read from the environment on each call.
    """
    place = _worker_place()
    return (
        "Build and publish a new image from this change. Image 0.11.0 does not "
        "contain this path and must not be enabled. Publish only after the "
        "public-image audit passes for that exact artifact and a maintainer with "
        "publishing authority has approved it.",
        "Deploy that image. Leave KORY_SOURCE_AUTO_REVIEW unset until the remaining steps are done.",
        "Install the GitHub App on the repositories Kory should review. "
        "The installation is the scope: Kory reviews every repository the App "
        "is installed on.",
        "On the Kory Source service set KORY_SOURCE_AUTO_REVIEW=1 and "
        "KORY_SOURCE_AUTO_REVIEW_TOKEN to a shared secret. The review request "
        "stays auto. The router picks the seat. Restart the service.",
        f"On {place} start the seat worker with KORY_SOURCE_AUTO_REVIEW=1, the same "
        "token, KORY_SOURCE_URL pointing at the service, and the local relay "
        "credentials. Run `python -m mira.auto_review.worker`. A worker outage "
        "stops Kory only.",
        "Pause without a deploy: as an admin, POST /api/auto-review/switch with "
        '{"paused": true}. New heads are not queued and a review that has not '
        "posted yet does not post. Stop the worker and unset "
        "KORY_SOURCE_AUTO_REVIEW for a full stop.",
        f"Rollback the same way: pause, stop the worker on {place}, and unset "
        "the flag. No code reconstruction and no branch-protection change. Posted "
        "comments stay on the frozen commit they reviewed. Greptile is untouched.",
    )


def concurrency_limit(env: str, default: int, maximum: int) -> int:
    """Bound runtime concurrency without changing the review configuration."""
    value = int(os.environ.get(env, str(default)))
    if not 1 <= value <= maximum:
        raise ValueError(f"{env} must be between 1 and {maximum}")
    return value


def flag_on() -> bool:
    """True only when the process env explicitly enables the path."""
    return os.environ.get(ENV_FLAG) == "1"


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
    lower the floor, raise the cap, or turn nitpicks back on. The model stays
    ``auto`` so the router can fall through the skill route.
    """
    llm = LLMConfig(
        provider="router-acp",
        model="auto",
        review_model="auto",
        indexing_model="auto",
        security_model="auto",
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


def ownership(db: Any, owner: str, repo: str, installation_id: int) -> str | None:
    """How a pull_request delivery for an installed repository is handled.

    ``None`` means the flag is off, so today's webhook reviewer runs.
    ``enqueue`` queues one secondary review and must not call that reviewer.
    ``ignore`` means the flag is on and the pause bit is set, so neither
    reviewer runs. The GitHub App installation is the scope: ``owner``,
    ``repo``, and ``installation_id`` are not a filter.
    """
    del owner, repo, installation_id
    if not flag_on():
        return None
    if paused(db):
        return "ignore"
    return "enqueue"


def jobs_for(has_posted: bool) -> list[tuple[str, str]]:
    """Role and router skill for each review of one head.

    The first reviewed head gets a full opinion that prefers Grok and a full
    gate that prefers Opus or Sol. A later head reviews the new changes: the
    gate prefers Opus or Sol, and the extra opinion runs only on Grok.
    """
    if has_posted:
        return [("gate", "review-gate"), ("opinion", GROK_ONLY_SKILL)]
    return [("opinion", "review-opinion"), ("gate", "review-gate")]


def skill_line(role: str, skill: str = "") -> str:
    """One stripped router directive. The model does not see it.

    An allowed skill wins. Otherwise the role's default skill is used.
    """
    chosen = skill if skill in ALLOWED_SKILLS else ROLE_SKILL.get(role)
    if chosen not in ALLOWED_SKILLS:
        raise ValueError(f"unknown review role {role!r}")
    return f"[router: skill={chosen}]"


def review_preamble(head_sha: str, role: str = DEFAULT_ROLE) -> str:
    if role not in ROLE_SKILL:
        raise ValueError(f"unknown review role {role!r}")
    return f"**{SECONDARY_LABEL}**\n\nReviewed head `{head_sha}`\nRole: {role}"


def switch_view(db: Any) -> dict[str, Any]:
    """Admin view of the env and pause gate.

    ``effective`` is that gate. While the flag is on, each delivery from an
    installed repository is queued unless this view says it is paused.
    """
    enabled = flag_on()
    is_paused = paused(db)
    return {
        "env_enabled": enabled,
        "paused": is_paused,
        "effective": enabled and not is_paused,
        "worker_host": worker_host(),
        "model": "auto",
        "timeout_seconds": CALL_TIMEOUT_SECONDS,
        "review_event": REVIEW_EVENT,
        "conversations": True,
        "registry_context": False,
    }
