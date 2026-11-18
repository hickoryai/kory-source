"""Shared classification of inline review-comment authorship."""

from __future__ import annotations

import re
from collections.abc import Iterable

DEFAULT_AI_REVIEWERS = (
    "greptile-apps",
    "copilot-pull-request-reviewer",
    "claude",
    "coderabbitai",
    "kory-source",
)

_BOT_SUFFIX = re.compile(r"\[bot\]\s*$", re.IGNORECASE)
_BOT_BODY_PREFIX = re.compile(r"^(?:\s|>\s*)*🤖")


def normalize_login(login: str) -> str:
    """Normalize a login for identity comparisons, including app bot suffixes."""
    return _BOT_SUFFIX.sub("", login.strip()).casefold()


def is_own_review(author: str, bot_login: str) -> bool:
    """Whether ``author`` is Mira itself or the configured posting identity."""
    normalized = normalize_login(author)
    return bool(normalized) and normalized in {
        normalize_login("kory-source"),
        normalize_login(bot_login),
    }


def classify_review_comment(
    author: str,
    body: str,
    *,
    account_type: str = "",
    bot_login: str = "",
    known_ai_reviewers: Iterable[str] | None = None,
) -> str:
    """Return ``human_review`` only when the supplied evidence is human."""
    if known_ai_reviewers is None:
        from mira.config import load_config

        known_ai_reviewers = load_config().feedback.known_ai_reviewers
    known = known_ai_reviewers
    known_logins = {normalize_login(login) for login in known}
    login = normalize_login(author)
    is_human = (
        account_type.casefold() != "bot"
        and not author.strip().casefold().endswith("[bot]")
        and login not in known_logins
        and not is_own_review(author, bot_login)
        and not _BOT_BODY_PREFIX.match(body or "")
    )
    return "human_review" if is_human else "bot_review"
