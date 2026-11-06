"""Who said, ran, directed, reviewed, approved, imported, and extracted a decision.

HAI-12825 keeps these roles apart. One ``author`` string cannot tell a bot
account from the model behind it, the person who invoked it, or the importer
that later copied the text. Each attribution row records:

- original labels exactly as the source gave them (``original_*``);
- normalized values only for mappings this module supports, stamped with
  ``NORMALIZATION_VERSION``;
- a per-field state, ``stated`` / ``inferred`` / ``unknown``, with the evidence
  id and a pointer (quoted line or JSON pointer) that supports it.

Rules this module enforces:

- Unknown stays unknown: ``None`` with a reason, never an empty string.
- A bot display name never yields a model, a version, or a human invoker.
- An account of type ``User`` is a person's account; the text it posted may
  still be agent-authored, so authorship from the account alone is ``inferred``.
- Importer and extractor identities are their own roles. They never fill a
  source author's model fields.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

NORMALIZATION_VERSION = "attribution-normalize/1"
ROLES = (
    "decision_author",
    "source_account",
    "invoking_actor",
    "director",
    "reviewer",
    "approver",
    "importer",
    "extractor",
)
ACTOR_KINDS = ("human", "bot", "automation", "unknown")
STATES = ("stated", "inferred", "unknown")
ORIGINAL_FIELDS = (
    "original_identity",
    "original_provider",
    "original_bot",
    "original_model",
    "original_client",
    "original_environment",
)
NORMALIZED_FIELDS = (
    "normalized_identity",
    "provider_family",
    "model_family",
    "model_version",
    "client_interface",
    "execution_environment",
)
SESSION_FIELDS = ("source_session_id", "source_run_id", "occurred_at")
ALL_FIELDS = ("actor_kind", *ORIGINAL_FIELDS, *NORMALIZED_FIELDS, *SESSION_FIELDS)

# ── normalization ───────────────────────────────────────────────────────

_ANTHROPIC = re.compile(
    r"\bclaude[\s_-]*(?P<family>opus|sonnet|haiku|fable)[\s_-]*(?P<version>\d+(?:[.\-_]\d+)?)?",
    re.I,
)
_ANTHROPIC_VERSION_FIRST = re.compile(
    r"\bclaude[\s_-]*(?P<version>\d+(?:[.\-_]\d+)?)[\s_-]*(?P<family>opus|sonnet|haiku|fable)\b",
    re.I,
)
_OPENAI = re.compile(r"\b(?P<family>gpt|o)[\s_-]?(?P<version>\d+(?:\.\d+)?(?:[\w.-]*)?)", re.I)
_GEMINI = re.compile(r"\bgemini[\s_-]*(?P<version>\d+(?:\.\d+)?)?[\s_-]*(?P<tier>pro|flash)?", re.I)
_CLIENTS = {
    "cli": "cli",
    "terminal": "cli",
    "vscode": "ide",
    "vs code": "ide",
    "jetbrains": "ide",
    "ide": "ide",
    "web": "web",
    "desktop": "desktop",
    "api": "api",
    "github actions": "github_actions",
    "github-actions": "github_actions",
}
_ENVIRONMENTS = {
    "dev-workstation": "workstation",
    "workstation": "workstation",
    "laptop": "laptop",
    "ci": "ci",
    "github actions": "ci",
    "cloud": "cloud",
}


def _dotted(version: str | None) -> str | None:
    if not version:
        return None
    return re.sub(r"[\-_]", ".", version)


def normalize_model(label: str | None) -> dict[str, str | None]:
    """Provider family, model family and version for a supported label, else Nones."""
    out: dict[str, str | None] = {
        "provider_family": None,
        "model_family": None,
        "model_version": None,
    }
    text = (label or "").strip()
    if not text:
        return out
    m = _ANTHROPIC.search(text) or _ANTHROPIC_VERSION_FIRST.search(text)
    if m:
        out.update(
            provider_family="anthropic",
            model_family=m.group("family").lower(),
            model_version=_dotted(m.group("version")),
        )
        return out
    m = _GEMINI.search(text)
    if m:
        out.update(provider_family="google", model_family="gemini", model_version=m["version"])
        return out
    m = _OPENAI.search(text)
    if m and (m["family"].lower() == "gpt" or text.lower().startswith(("o", "openai"))):
        out.update(
            provider_family="openai",
            model_family=m["family"].lower(),
            model_version=m["version"].lower(),
        )
    return out


def normalize_client(label: str | None) -> str | None:
    return _CLIENTS.get((label or "").strip().lower())


def normalize_environment(label: str | None) -> str | None:
    return _ENVIRONMENTS.get((label or "").strip().lower())


# ── the record ──────────────────────────────────────────────────────────


@dataclass
class Attribution:
    """One role's attribution for one piece of evidence."""

    role: str
    evidence_id: str
    object_id: str = ""
    decision_revision: int = 0
    values: dict[str, Any] = field(default_factory=dict)
    states: dict[str, dict[str, Any]] = field(default_factory=dict)
    confidence: str = "unknown"
    reason: str = ""

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown attribution role {self.role!r}")
        self.values.setdefault("actor_kind", "unknown")

    def set(
        self,
        name: str,
        value: Any,
        state: str,
        *,
        pointer: str = "",
        reason: str = "",
        evidence_id: str | None = None,
    ) -> Attribution:
        if name not in ALL_FIELDS:
            raise ValueError(f"unknown attribution field {name!r}")
        if state not in STATES:
            raise ValueError(f"unknown field state {state!r}")
        if value in (None, "") and state != "unknown":
            raise ValueError(f"{name}: a blank value must be state unknown")
        if name == "actor_kind" and value not in ACTOR_KINDS:
            raise ValueError(f"actor_kind must be one of {', '.join(ACTOR_KINDS)}")
        self.values[name] = None if value == "" else value
        self.states[name] = {
            "state": state,
            "evidence_id": evidence_id or self.evidence_id,
            "pointer": pointer,
            "reason": reason,
        }
        return self

    def unknown(self, name: str, reason: str) -> Attribution:
        if name == "actor_kind":
            self.values["actor_kind"] = "unknown"
            self.states[name] = {
                "state": "unknown",
                "evidence_id": "",
                "pointer": "",
                "reason": reason,
            }
            return self
        return self.set(name, None, "unknown", reason=reason, evidence_id="")

    def field_states(self) -> dict[str, dict[str, Any]]:
        """Every field, with unknown ones explicit."""
        out = {}
        for name in ALL_FIELDS:
            out[name] = self.states.get(
                name,
                {"state": "unknown", "evidence_id": "", "pointer": "", "reason": "not in source"},
            )
        return out

    @property
    def attribution_id(self) -> str:
        """Stable per (evidence, object, role, values): a re-import is a no-op."""
        key = json.dumps(
            [self.evidence_id, self.object_id, self.decision_revision, self.role, self.values],
            sort_keys=True,
            default=str,
        )
        return "att_" + hashlib.sha256(key.encode()).hexdigest()[:24]

    def row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "attribution_id": self.attribution_id,
            "evidence_id": self.evidence_id,
            "object_id": self.object_id,
            "decision_revision": self.decision_revision,
            "role": self.role,
            "normalization_version": NORMALIZATION_VERSION,
            "field_states": json.dumps(self.field_states(), sort_keys=True),
            "confidence": self.confidence,
            "reason": self.reason,
        }
        for name in ALL_FIELDS:
            row[name] = self.values.get(name)
        row["actor_kind"] = row["actor_kind"] or "unknown"
        return row


def set_model(attr: Attribution, label: str, *, pointer: str, state: str = "stated") -> None:
    """Store a model label verbatim and normalize only what this module supports."""
    attr.set("original_model", label, state, pointer=pointer)
    norm = normalize_model(label)
    for key, value in norm.items():
        if value:
            attr.set(key, value, "inferred", pointer=pointer, reason=NORMALIZATION_VERSION)
        else:
            attr.unknown(key, f"label {label!r} has no supported {key} mapping")


# ── markers in text ─────────────────────────────────────────────────────

_TRAILER = re.compile(
    r"^\s*(?P<key>co-authored-by|model|provider|client|environment|session|run|"
    r"invoked-by|directed-by|generated-by)\s*:\s*(?P<value>.+?)\s*$",
    re.I | re.M,
)
_GENERATED_WITH = re.compile(
    r"generated with \[?(?P<client>[A-Za-z][\w .-]*?)\]?(?:\(|\s*$)", re.I | re.M
)
_AGENT_NAMES = re.compile(r"\b(claude|codex|gpt|copilot|gemini|cursor|devin|kory)\b", re.I)
_EMAIL = re.compile(r"<(?P<email>[^>]+)>")


@dataclass(frozen=True)
class Marker:
    key: str
    value: str
    quote: str
    start: int


def markers(text: str) -> list[Marker]:
    """Explicit attribution statements in ``text`` (trailers and tool footers)."""
    found: list[Marker] = []
    for m in _TRAILER.finditer(text or ""):
        found.append(Marker(m["key"].lower(), m["value"], m.group(0).strip(), m.start()))
    for m in _GENERATED_WITH.finditer(text or ""):
        found.append(Marker("generated-with", m["client"].strip(), m.group(0).strip(), m.start()))
    return found


def agent_coauthors(found: list[Marker]) -> list[Marker]:
    return [m for m in found if m.key == "co-authored-by" and _AGENT_NAMES.search(m.value)]


def from_markers(
    evidence_id: str, found: list[Marker], *, object_id: str = "", revision: int = 0
) -> list[Attribution]:
    """decision_author / invoking_actor / director rows the text states about itself.

    These are self-reported by the text: ``stated`` means "the source says so",
    not "verified". The verification level is kept in ``reason``.
    """
    out: list[Attribution] = []
    by_key: dict[str, list[Marker]] = {}
    for m in found:
        by_key.setdefault(m.key, []).append(m)
    agents = agent_coauthors(found)
    model_markers = by_key.get("model", [])
    client_markers = by_key.get("client", []) + by_key.get("generated-with", [])
    if agents or model_markers or client_markers or by_key.get("generated-by"):
        attr = Attribution(
            "decision_author",
            evidence_id,
            object_id,
            revision,
            reason="self-reported by markers in the source text; not independently verified",
        )
        attr.set(
            "actor_kind",
            "bot",
            "stated",
            pointer=(agents or model_markers or client_markers or by_key["generated-by"])[0].quote,
        )
        if agents:
            name = _EMAIL.sub("", agents[0].value).strip()
            email = _EMAIL.search(agents[0].value)
            attr.set("original_identity", agents[0].value, "stated", pointer=agents[0].quote)
            if email:
                attr.set(
                    "normalized_identity",
                    f"email:{email['email'].lower()}",
                    "inferred",
                    pointer=agents[0].quote,
                )
            if not model_markers:
                set_model(attr, name, pointer=agents[0].quote)
        if model_markers:
            set_model(attr, model_markers[0].value, pointer=model_markers[0].quote)
        if not (agents or model_markers):
            for name in ("original_model", "provider_family", "model_family", "model_version"):
                attr.unknown(name, "no model named in the source")
        for m in by_key.get("provider", [])[:1]:
            attr.set("original_provider", m.value, "stated", pointer=m.quote)
        if client_markers:
            m = client_markers[0]
            attr.set("original_client", m.value, "stated", pointer=m.quote)
            client = normalize_client(m.value)
            if client:
                attr.set(
                    "client_interface",
                    client,
                    "inferred",
                    pointer=m.quote,
                    reason=NORMALIZATION_VERSION,
                )
            else:
                attr.unknown("client_interface", f"client {m.value!r} names no single interface")
        for m in by_key.get("environment", [])[:1]:
            attr.set("original_environment", m.value, "stated", pointer=m.quote)
            env = normalize_environment(m.value)
            if env:
                attr.set(
                    "execution_environment",
                    env,
                    "inferred",
                    pointer=m.quote,
                    reason=NORMALIZATION_VERSION,
                )
        for key, name in (("session", "source_session_id"), ("run", "source_run_id")):
            for m in by_key.get(key, [])[:1]:
                attr.set(name, m.value, "stated", pointer=m.quote)
        out.append(attr)
    for key, role in (("invoked-by", "invoking_actor"), ("directed-by", "director")):
        for m in by_key.get(key, [])[:1]:
            attr = Attribution(
                role,
                evidence_id,
                object_id,
                revision,
                reason="self-reported by the source text; the named person did not sign it",
            )
            attr.set("original_identity", m.value, "stated", pointer=m.quote)
            attr.set("actor_kind", "human", "inferred", pointer=m.quote, reason="named as invoker")
            out.append(attr)
    return out


# ── source accounts ─────────────────────────────────────────────────────


def github_account(
    evidence_id: str, user: dict | None, *, occurred_at: float | None = None, pointer: str = "/user"
) -> Attribution:
    """The GitHub account that posted. Account facts are stated by GitHub."""
    attr = Attribution("source_account", evidence_id, reason="GitHub API user object")
    if not user:
        attr.unknown("actor_kind", "GitHub returned no user (deleted account or ghost)")
        attr.unknown("original_identity", "no user")
        return attr
    login = str(user.get("login") or "")
    account_type = str(user.get("type") or "")
    if login:
        attr.set("original_identity", login, "stated", pointer=f"{pointer}/login")
    node = user.get("node_id") or user.get("id")
    if node:
        attr.set("normalized_identity", f"github:{node}", "stated", pointer=f"{pointer}/node_id")
    if account_type == "Bot":
        attr.set("actor_kind", "bot", "stated", pointer=f"{pointer}/type")
        attr.set("original_bot", login, "stated", pointer=f"{pointer}/login")
    elif account_type == "User":
        attr.set(
            "actor_kind",
            "human",
            "stated",
            pointer=f"{pointer}/type",
            reason="a person's account; posted text may still be agent-authored",
        )
    else:
        attr.unknown("actor_kind", f"GitHub account type {account_type!r}")
    if occurred_at:
        attr.set("occurred_at", occurred_at, "stated", pointer="/created_at")
    return attr


def author_from_account(account: Attribution, found: list[Marker]) -> Attribution | None:
    """decision_author inferred from the posting account when the text names no agent.

    Returns None when markers already state the author.
    """
    if agent_coauthors(found) or any(
        m.key in ("model", "generated-with", "generated-by") for m in found
    ):
        return None
    attr = Attribution(
        "decision_author",
        account.evidence_id,
        reason="inferred from the posting account; no author marker in the text",
    )
    kind = account.values.get("actor_kind") or "unknown"
    if kind == "unknown":
        attr.unknown("actor_kind", "posting account unknown")
    else:
        attr.set(
            "actor_kind",
            kind,
            "inferred",
            pointer="source_account.actor_kind",
            reason="same as posting account",
        )
    if account.values.get("original_identity"):
        attr.set(
            "original_identity",
            account.values["original_identity"],
            "inferred",
            pointer="source_account.original_identity",
        )
    for name in ("original_model", "provider_family", "model_family", "model_version"):
        attr.unknown(name, "a display name or account never implies a model")
    return attr


def linear_actor(
    evidence_id: str, node: dict, *, user_key: str = "user", pointer: str = ""
) -> list[Attribution]:
    """source_account (and invoking_actor for integrations) from a Linear node."""
    out: list[Attribution] = []
    user = node.get(user_key) or {}
    bot = node.get("botActor") or {}
    attr = Attribution("source_account", evidence_id, reason="Linear API actor fields")
    base = pointer or f"/{user_key}"
    if user:
        attr.set(
            "original_identity",
            user.get("email") or user.get("displayName") or user.get("name") or "",
            "stated",
            pointer=base,
        )
        if user.get("id"):
            attr.set(
                "normalized_identity", f"linear-user:{user['id']}", "stated", pointer=f"{base}/id"
            )
        if user.get("app") is True:
            attr.set("actor_kind", "bot", "stated", pointer=f"{base}/app")
        elif user.get("app") is False:
            attr.set(
                "actor_kind",
                "human",
                "stated",
                pointer=f"{base}/app",
                reason="a person's account; text may still be agent-authored",
            )
        else:
            attr.unknown("actor_kind", "Linear did not say whether the user is an app")
    elif bot:
        attr.set("actor_kind", "bot", "stated", pointer="/botActor")
        if bot.get("name"):
            attr.set("original_bot", bot["name"], "stated", pointer="/botActor/name")
        provider = " ".join(p for p in (bot.get("type"), bot.get("subType")) if p)
        if provider:
            attr.set("original_provider", provider, "stated", pointer="/botActor/type")
        for name in ("original_model", "model_family", "model_version"):
            attr.unknown(name, "a bot name never implies a model")
    else:
        attr.unknown("actor_kind", "no user or bot actor on this record")
    out.append(attr)
    if bot.get("userDisplayName"):
        inv = Attribution(
            "invoking_actor",
            evidence_id,
            reason="Linear botActor.userDisplayName: the external user on whose behalf the bot acted",
        )
        inv.set(
            "original_identity",
            bot["userDisplayName"],
            "stated",
            pointer="/botActor/userDisplayName",
        )
        inv.set("actor_kind", "human", "inferred", pointer="/botActor/userDisplayName")
        inv.unknown("normalized_identity", "a display name is not a stable identity")
        out.append(inv)
    return out


def git_author(evidence_id: str, commit: dict) -> list[Attribution]:
    """Git author and committer labels. Neither is automatically the decision author."""
    out = []
    for who in ("author", "committer"):
        name, email = commit.get(f"{who}_name") or "", commit.get(f"{who}_email") or ""
        if not (name or email):
            continue
        attr = Attribution(
            "source_account",
            evidence_id,
            reason=f"git {who}; not necessarily the speaker or invoker",
        )
        attr.set("original_identity", f"{name} <{email}>".strip(), "stated", pointer=f"/{who}")
        if email:
            attr.set(
                "normalized_identity", f"email:{email.lower()}", "inferred", pointer=f"/{who}_email"
            )
        attr.unknown(
            "actor_kind", "git identities do not say whether a person or agent wrote the change"
        )
        if commit.get(f"{who}_date"):
            attr.set("occurred_at", float(commit[f"{who}_date"]), "stated", pointer=f"/{who}_date")
        out.append(attr)
    return out


def tool_attribution(
    role: str,
    evidence_id: str,
    *,
    identity: str,
    version: str,
    model: str = "",
    object_id: str = "",
    revision: int = 0,
) -> Attribution:
    """importer or extractor: who copied or interpreted the source. Never the author."""
    if role not in ("importer", "extractor"):
        raise ValueError("tool attribution is for importer or extractor")
    attr = Attribution(role, evidence_id, object_id, revision, reason=f"{role} run record")
    attr.set("actor_kind", "automation", "stated", pointer="import_run")
    attr.set("original_identity", identity, "stated", pointer="import_run.operator")
    attr.set("original_client", version, "stated", pointer="import_run.version")
    if model:
        set_model(attr, model, pointer="import_run.extractor_model")
    else:
        for name in ("original_model", "provider_family", "model_family", "model_version"):
            attr.unknown(name, f"{role} is deterministic; no model ran")
    return attr
