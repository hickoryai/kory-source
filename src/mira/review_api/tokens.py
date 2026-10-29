"""Bearer tokens for the review API.

Tokens are configured by the platform administrator, never issued by this
service. The deployment holds only SHA-256 digests, in
``MIRA_REVIEW_API_TOKENS`` (JSON) or the file named by
``MIRA_REVIEW_API_TOKENS_FILE``::

    [{"principal": "zane-laptop", "kind": "engineer",
      "sha256": "<hex digest of the raw token>",
      "repos": ["hickoryai/kory-source", "hickoryai/*"]}]

``kind`` is ``engineer`` or ``relay``. A bearer token reaches only
``/api/reviews`` routes; it never grants dashboard or admin access.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_KINDS = frozenset({"engineer", "relay"})
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ReviewPrincipal:
    name: str
    kind: str
    repos: tuple[str, ...]

    def allows(self, owner: str, repo: str) -> bool:
        full = f"{owner}/{repo}".lower()
        for pattern in self.repos:
            pattern = pattern.lower()
            if pattern == full or (pattern.endswith("/*") and full.split("/")[0] == pattern[:-2]):
                return True
        return False


@dataclass(frozen=True)
class _Entry:
    digest: str
    principal: ReviewPrincipal


def _raw_config() -> str:
    path = os.environ.get("MIRA_REVIEW_API_TOKENS_FILE", "").strip()
    if path:
        try:
            with open(path, encoding="utf-8") as f:
                return f.read()
        except OSError as exc:
            logger.warning("review API token file unreadable: %s", exc)
            return ""
    return os.environ.get("MIRA_REVIEW_API_TOKENS", "")


def load_entries() -> list[_Entry]:
    """Parse the configured digests. Malformed entries are skipped and logged."""
    raw = _raw_config().strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("review API token config is not valid JSON")
        return []
    if not isinstance(data, list):
        logger.warning("review API token config must be a JSON list")
        return []
    entries: list[_Entry] = []
    names: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("principal") or "").strip()
        kind = str(item.get("kind") or "").strip()
        digest = str(item.get("sha256") or "").strip().lower()
        repos = item.get("repos")
        if (
            not name
            or name in names
            or kind not in _KINDS
            or not _DIGEST.match(digest)
            or not isinstance(repos, list)
            or not all(isinstance(r, str) and "/" in r for r in repos)
        ):
            logger.warning("review API token entry %r skipped: invalid fields", name or "?")
            continue
        names.add(name)
        entries.append(_Entry(digest, ReviewPrincipal(name, kind, tuple(repos))))
    return entries


def authenticate(token: str) -> ReviewPrincipal | None:
    """Return the principal for a raw bearer token, or ``None``."""
    if not token:
        return None
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    match: ReviewPrincipal | None = None
    for entry in load_entries():
        # Compare every entry so timing does not reveal which one matched.
        if hmac.compare_digest(entry.digest, digest):
            match = entry.principal
    return match


def transport_allowed(request: Any) -> bool:
    """Bearer tokens travel over HTTPS, or plain HTTP to a loopback host.

    Behind a TLS-terminating proxy (Cloud Run) the app sees ``http``; the
    proxy's ``X-Forwarded-Proto: https`` is accepted.
    """
    if request.url.scheme == "https":
        return True
    if request.headers.get("x-forwarded-proto", "").split(",")[0].strip() == "https":
        return True
    host = request.url.hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
