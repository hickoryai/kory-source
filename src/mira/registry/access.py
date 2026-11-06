"""The complete-corpus access boundary (implementation contract, plan section 5a).

Imported history from GitHub and every Linear team, private teams included, is
readable by ONE explicitly approved audience and nobody else. Engineering role,
dashboard admin, a ``kory.source`` capability, ``kind=human`` in a token, or a
model header grant nothing here.

The policy is deployment-owned: ``REGISTRY_CORPUS_ACCESS_POLICY`` holds a JSON
manifest (see ``docs/decision-provenance.md``). A request can never supply its
own allowlist, policy version, or authority. A missing, invalid, disabled, or
expired policy denies every protected import and read. The policy is re-read
on every request, so removing a member takes effect without waiting for their
token to expire.

Membership keys are namespaced immutable principal ids, never display names
or usernames:

- ``registry-jwt``: the verified ``sub`` of a registry JWT. The Kory MCP gateway
  signs the server-resolved owner UUID there; a direct caller's ``sub`` is its
  own issued identity.
- ``dashboard-account``: the dashboard's numeric account id.
- ``operator``: an import job's configured identity
  (``REGISTRY_IMPORT_OPERATOR``, set by the deployment, never by a request).

The SQL reader (``pg_read_all_data``) is a separate approval for the entire
database; nothing in this module grants or implies it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mira.registry.model import RegistryError

logger = logging.getLogger(__name__)

POLICY_ENV = "REGISTRY_CORPUS_ACCESS_POLICY"
OPERATOR_ENV = "REGISTRY_IMPORT_OPERATOR"
POLICY_SCHEMA = "kory-source-corpus-policy/1"
NAMESPACES = ("registry-jwt", "dashboard-account", "operator")
# Imported rows from these sources are not part of the decision corpus: they
# mirror Mira's own approved learned rules and keep their prior visibility.
LEGACY_VISIBLE_SOURCES = ("", "mira_learned_rule")
_WILDCARDS = ("*", "all", "any", "%")


class PolicyError(RegistryError):
    """The corpus policy manifest is malformed. Treated as deny-all."""


class AccessDenied(RegistryError):
    """The caller may not import or read protected corpus data.

    The message never names an object, a count, or whether something exists.
    """

    def __init__(self, message: str = "corpus access is not granted") -> None:
        super().__init__(message)


@dataclass(frozen=True)
class Member:
    namespace: str
    principal_id: str
    approval_reference: str = ""
    expires_at: float = 0.0  # 0: the policy's own expiry applies

    @property
    def key(self) -> str:
        return f"{self.namespace}:{self.principal_id}"


@dataclass(frozen=True)
class CorpusPolicy:
    corpus_id: str
    policy_version: str
    enabled: bool
    approval_reference: str
    approved_by: str
    approved_at: str
    expires_at: float
    repositories: tuple[tuple[str, str], ...]  # (stable id, owner/name)
    linear_teams: tuple[tuple[str, str], ...]  # (stable id, key)
    readers: tuple[Member, ...]
    import_operators: tuple[Member, ...]
    legacy_reviewer_sources: tuple[str, ...]
    manifest_sha256: str
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def active(self, now: float) -> bool:
        return self.enabled and now < self.expires_at

    def admits_repository(self, repo: str) -> bool:
        """True for a manifest repository, matched by stable id or owner/name."""
        repo = (repo or "").strip()
        return any(repo == rid or repo.lower() == name.lower() for rid, name in self.repositories)

    def repository_id(self, repo: str) -> str:
        for rid, name in self.repositories:
            if repo == rid or repo.lower() == name.lower():
                return rid
        return ""

    def admits_linear_team(self, *, team_id: str = "", key: str = "") -> bool:
        """A Linear team is admitted only by its manifest id; a key alone must match it too."""
        for tid, tkey in self.linear_teams:
            if team_id:
                if team_id == tid:
                    return not key or key == tkey
            elif key and key == tkey:
                return True
        return False

    @property
    def team_keys(self) -> list[str]:
        return [key for _, key in self.linear_teams]

    def member(self, members: tuple[Member, ...], namespace: str, pid: str, now: float):
        for m in members:
            if m.namespace == namespace and m.principal_id == pid:
                if m.expires_at and now >= m.expires_at:
                    return None
                return m
        return None


@dataclass(frozen=True)
class AccessContext:
    """Who is reading, and which corpus they may see. Derived server-side only."""

    principal: str = ""
    purpose: str = "none"  # none | read | import | reviewer
    corpus_ids: frozenset[str] = frozenset()
    policy_version: str = ""
    manifest_sha256: str = ""
    legacy_sources: frozenset[str] = frozenset()

    def can_read(self, corpus_id: str) -> bool:
        return bool(corpus_id) and corpus_id in self.corpus_ids

    @property
    def has_corpus(self) -> bool:
        return bool(self.corpus_ids)

    @property
    def cache_key(self) -> str:
        """Cache partition: principal, corpus, and policy version together."""
        corpora = ",".join(sorted(self.corpus_ids))
        return f"{self.principal}|{corpora}|{self.policy_version}|{self.manifest_sha256}"


DENIED = AccessContext()


def _epoch(value: Any, what: str) -> float:
    if not isinstance(value, str) or not value.strip():
        raise PolicyError(f"{what} must be an ISO-8601 timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        raise PolicyError(f"{what} is not an ISO-8601 timestamp") from None


def _text(obj: Mapping[str, Any], key: str, what: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PolicyError(f"{what}.{key} is required")
    value = value.strip()
    if value.lower() in _WILDCARDS:
        raise PolicyError(f"{what}.{key} may not be a wildcard; enumerate it")
    return value


def _members(raw: Any, what: str, namespaces: tuple[str, ...]) -> tuple[Member, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise PolicyError(f"{what} must be a list")
    out: list[Member] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise PolicyError(f"{what}[{i}] must be an object")
        ns = _text(entry, "namespace", f"{what}[{i}]")
        if ns not in namespaces:
            raise PolicyError(f"{what}[{i}].namespace must be one of {', '.join(namespaces)}")
        pid = _text(entry, "id", f"{what}[{i}]")
        expires = entry.get("expires_at")
        out.append(
            Member(
                ns,
                pid,
                str(entry.get("approval_reference") or ""),
                _epoch(expires, f"{what}[{i}].expires_at") if expires else 0.0,
            )
        )
    return tuple(out)


def canonical_sha256(manifest: Mapping[str, Any]) -> str:
    text = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def parse_policy(raw: str | Mapping[str, Any]) -> CorpusPolicy:
    """Validate a policy manifest. Raises ``PolicyError`` on anything doubtful."""
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except ValueError:
            raise PolicyError("corpus policy is not valid JSON") from None
    else:
        data = dict(raw)
    if not isinstance(data, Mapping):
        raise PolicyError("corpus policy must be a JSON object")
    if data.get("schema") != POLICY_SCHEMA:
        raise PolicyError(f"corpus policy schema must be {POLICY_SCHEMA!r}")
    approval = data.get("approval")
    if not isinstance(approval, Mapping):
        raise PolicyError("approval is required")
    repos_raw = data.get("repositories")
    teams_raw = data.get("linear_teams")
    if not isinstance(repos_raw, list) or not isinstance(teams_raw, list):
        raise PolicyError("repositories and linear_teams must be enumerated lists")
    repositories = tuple(
        _pair(r, f"repositories[{i}]", "full_name") for i, r in enumerate(repos_raw)
    )
    teams = tuple(_pair(t, f"linear_teams[{i}]", "key") for i, t in enumerate(teams_raw))
    for name in (n for _, n in repositories):
        if "/" not in name:
            raise PolicyError(f"repository {name!r} must be owner/name")
    legacy = data.get("legacy_reviewer_sources") or []
    if not isinstance(legacy, list) or not all(isinstance(s, str) and s for s in legacy):
        raise PolicyError("legacy_reviewer_sources must be a list of source system names")
    enabled = data.get("enabled")
    if not isinstance(enabled, bool):
        raise PolicyError("enabled must be true or false")
    return CorpusPolicy(
        corpus_id=_text(data, "corpus_id", "policy"),
        policy_version=_text(data, "policy_version", "policy"),
        enabled=enabled,
        approval_reference=_text(approval, "reference", "approval"),
        approved_by=_text(approval, "approved_by", "approval"),
        approved_at=_text(approval, "approved_at", "approval"),
        expires_at=_epoch(approval.get("expires_at"), "approval.expires_at"),
        repositories=repositories,
        linear_teams=teams,
        readers=_members(data.get("readers"), "readers", ("registry-jwt", "dashboard-account")),
        import_operators=_members(data.get("import_operators"), "import_operators", ("operator",)),
        legacy_reviewer_sources=tuple(legacy),
        manifest_sha256=canonical_sha256(data),
        raw=data,
    )


def _pair(entry: Any, what: str, label: str) -> tuple[str, str]:
    if not isinstance(entry, Mapping):
        raise PolicyError(f"{what} must be an object")
    return _text(entry, "id", what), _text(entry, label, what)


class PolicySource:
    """Reads the deployment policy on each call; parses each distinct value once."""

    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        self._env = env
        self._lock = threading.Lock()
        self._cached: tuple[str, CorpusPolicy | None] | None = None

    def raw(self) -> str:
        env = self._env if self._env is not None else os.environ
        return env.get(POLICY_ENV, "")

    def policy(self) -> CorpusPolicy | None:
        raw = self.raw()
        if not raw.strip():
            return None
        with self._lock:
            if self._cached is not None and self._cached[0] == raw:
                return self._cached[1]
            try:
                parsed: CorpusPolicy | None = parse_policy(raw)
            except PolicyError as exc:
                # No manifest content in the log: it names private teams.
                logger.warning("Corpus access policy rejected; denying all: %s", exc)
                parsed = None
            self._cached = (raw, parsed)
            return parsed


_source = PolicySource()


def set_policy_source(source: PolicySource) -> None:
    """Swap the policy source (tests)."""
    global _source
    _source = source


def current_policy(now: float | None = None) -> CorpusPolicy | None:
    """The policy if it is valid, enabled, and unexpired; otherwise None."""
    policy = _source.policy()
    now = time.time() if now is None else now
    if policy is None or not policy.active(now):
        return None
    return policy


def _context(policy: CorpusPolicy, principal: str, purpose: str) -> AccessContext:
    return AccessContext(
        principal=principal,
        purpose=purpose,
        corpus_ids=frozenset({policy.corpus_id}),
        policy_version=policy.policy_version,
        manifest_sha256=policy.manifest_sha256,
    )


def reader_context(namespace: str, principal_id: str, *, now: float | None = None) -> AccessContext:
    """A complete-corpus read context, or ``DENIED``."""
    now = time.time() if now is None else now
    policy = current_policy(now)
    principal_id = str(principal_id or "").strip()
    if policy is None or not principal_id or namespace not in ("registry-jwt", "dashboard-account"):
        return DENIED
    if policy.member(policy.readers, namespace, principal_id, now) is None:
        return DENIED
    return _context(policy, f"{namespace}:{principal_id}", "read")


def context_for_actor(actor: Any, *, now: float | None = None) -> AccessContext:
    """From a verified registry JWT. ``kind`` and ``model`` play no part."""
    return reader_context("registry-jwt", getattr(actor, "sub", ""), now=now)


def context_for_dashboard_user(user: Any, *, now: float | None = None) -> AccessContext:
    """From a logged-in dashboard account's immutable id. Admin grants nothing."""
    uid = getattr(user, "id", None)
    if uid is None or isinstance(uid, bool):
        return DENIED
    return reader_context("dashboard-account", str(uid), now=now)


def operator_context(identity: str | None = None, *, now: float | None = None) -> AccessContext:
    """An import job's context. Raises ``AccessDenied`` unless policy lists the operator."""
    now = time.time() if now is None else now
    identity = (identity if identity is not None else os.environ.get(OPERATOR_ENV, "")).strip()
    policy = current_policy(now)
    if policy is None:
        raise AccessDenied("no enabled, unexpired corpus policy; protected imports are disabled")
    if not identity or policy.member(policy.import_operators, "operator", identity, now) is None:
        raise AccessDenied("this import operator is not approved by the corpus policy")
    return _context(policy, f"operator:{identity}", "import")


def reviewer_context(*, now: float | None = None) -> AccessContext:
    """The automated reviewer: no corpus, only sources the policy explicitly lists."""
    policy = current_policy(now)
    if policy is None or not policy.legacy_reviewer_sources:
        return DENIED
    return AccessContext(
        principal="reviewer",
        purpose="reviewer",
        policy_version=policy.policy_version,
        manifest_sha256=policy.manifest_sha256,
        legacy_sources=frozenset(policy.legacy_reviewer_sources),
    )


def require_import(ctx: AccessContext) -> CorpusPolicy:
    """The live policy behind an import context, re-checked now."""
    if ctx.purpose != "import" or not ctx.has_corpus:
        raise AccessDenied("protected imports need an approved operator context")
    policy = current_policy()
    if (
        policy is None
        or policy.corpus_id not in ctx.corpus_ids
        or policy.manifest_sha256 != ctx.manifest_sha256
    ):
        raise AccessDenied("the corpus policy changed or expired; start a new import run")
    return policy


def visible_sql(ctx: AccessContext | None, alias: str = "") -> tuple[str, list[Any]]:
    """A WHERE fragment over ``registry_objects`` columns for ``ctx``.

    Not a database boundary: it keeps every service read path from returning
    protected rows to a caller without corpus access. Unclassified imported rows
    (an imported source with no ``corpus_id``) match nothing but a policy-listed
    reviewer source.
    """
    ctx = ctx or DENIED
    p = f"{alias}." if alias else ""
    legacy = ", ".join("?" for _ in LEGACY_VISIBLE_SOURCES)
    clauses = [f"({p}corpus_id = '' AND {p}source_system IN ({legacy}))"]
    params: list[Any] = list(LEGACY_VISIBLE_SOURCES)
    if ctx.corpus_ids:
        clauses.append(f"{p}corpus_id IN ({', '.join('?' for _ in ctx.corpus_ids)})")
        params += sorted(ctx.corpus_ids)
    if ctx.legacy_sources:
        clauses.append(f"{p}source_system IN ({', '.join('?' for _ in ctx.legacy_sources)})")
        params += sorted(ctx.legacy_sources)
    return "(" + " OR ".join(clauses) + ")", params


def visible(obj: Mapping[str, Any], ctx: AccessContext | None) -> bool:
    """The Python form of ``visible_sql`` for one row."""
    ctx = ctx or DENIED
    corpus = str(obj.get("corpus_id") or "")
    system = str(obj.get("source_system") or "")
    if not corpus and system in LEGACY_VISIBLE_SOURCES:
        return True
    if corpus and ctx.can_read(corpus):
        return True
    return bool(system) and system in ctx.legacy_sources
