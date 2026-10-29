"""Deployment settings for the review API, read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


@dataclass(frozen=True)
class ReviewApiSettings:
    enabled: bool
    max_running: int
    max_queued: int
    deadline_seconds: int
    lease_seconds: int
    poll_seconds: float
    public_url: str


def load_settings() -> ReviewApiSettings:
    """Executor is off unless ``MIRA_REVIEW_API_ENABLED=1``.

    Turn it on only where CPU stays allocated between requests and at least
    one instance stays warm. On Cloud Run that is ``min_instance_count >= 1``
    (the kory_source module then sets ``cpu_idle = false``).
    """
    return ReviewApiSettings(
        enabled=os.environ.get("MIRA_REVIEW_API_ENABLED", "").strip() == "1",
        max_running=_int("MIRA_REVIEW_API_MAX_RUNNING", 2, 1),
        max_queued=_int("MIRA_REVIEW_API_MAX_QUEUED", 20, 0),
        deadline_seconds=_int("MIRA_REVIEW_API_DEADLINE_SECONDS", 3600, 60),
        lease_seconds=_int("MIRA_REVIEW_API_LEASE_SECONDS", 120, 10),
        poll_seconds=float(_int("MIRA_REVIEW_API_POLL_SECONDS", 2, 1)),
        public_url=os.environ.get("MIRA_REVIEW_API_PUBLIC_URL", "").strip().rstrip("/"),
    )
