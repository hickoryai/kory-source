"""Registry vocabulary, the actor making a call, and the confidence function."""

from __future__ import annotations

from dataclasses import dataclass

KINDS = (
    "work",
    "decision",
    "rule",
    "constraint",
    "requirement",
    "concept",
    "verification_evidence",
)
# Kinds that make a claim about how code should behave. They carry session and
# branch, run the conflict check, follow the realization tracker, and expire.
CLAIM_KINDS = ("decision", "rule", "constraint", "requirement")

STATUSES = ("proposed", "approved", "rejected", "superseded", "retired")
REALIZATIONS = (
    "stated",
    "in_progress",
    "submitted",
    "merged",
    "merged_with_conflict",
    "abandoned",
)
DELIVERY_STATES = ("planned", "active", "shipped", "canceled", "archived")
BASES = ("statement", "repository_analysis", "review_learning", "migration", "merged_pr")
EVIDENCE_RESULTS = ("pass", "fail", "inconclusive")

RELATIONSHIP_TYPES = (
    "supersedes",
    "constrains",
    "derived_from",
    "conflicts_with",
    "belongs_to",
    "parent_of",
    "blocks",
    "relates",
    "addresses",
    "records",
    "verifies",
)
# Work links: (relationship, required source kind, allowed target kinds).
WORK_LINKS = {
    "addresses": ("work", ("requirement",)),
    "records": ("work", ("decision",)),
    "parent_of": ("work", ("work",)),
    "blocks": ("work", ("work",)),
    "relates": ("work", KINDS),
}

CONFLICT_RELATIONS = ("narrows", "widens", "reverses")
EXPIRY_DAYS = 14

_APPROVAL_WEIGHT = {
    "approved": 0.70,
    "proposed": 0.40,
    "rejected": 0.05,
    "superseded": 0.20,
    "retired": 0.05,
}
_REALIZATION_WEIGHT = {
    "merged": 0.12,
    "merged_with_conflict": 0.12,
    "submitted": 0.06,
    "in_progress": 0.03,
    "stated": 0.0,
    "abandoned": -0.10,
}
_DAY = 86400.0


class RegistryError(ValueError):
    """A request the registry refuses. The message is safe to show the caller."""


class Forbidden(RegistryError):
    """The actor may not perform this action."""


class NotFound(RegistryError):
    """No object with that id."""


@dataclass(frozen=True)
class Actor:
    """Who is calling. Comes from the JWT or the dashboard session, never the body."""

    kind: str  # 'human' | 'agent'
    sub: str
    model: str = ""

    @property
    def is_human(self) -> bool:
        return self.kind == "human"


def compute_confidence(
    *,
    status: str,
    author_kind: str,
    realization: str,
    created_at: float,
    restored_at: float,
    now: float,
) -> float:
    """``clamp(approval + author + realization + age, 0.01, 0.99)``.

    Any merged, approved object (>= 0.82) outranks any stated proposal (<= 0.55).
    """
    score = _APPROVAL_WEIGHT.get(status, 0.0)
    if author_kind == "human":
        score += 0.15
    score += _REALIZATION_WEIGHT.get(realization, 0.0)
    age_days = (now - max(created_at, restored_at)) / _DAY
    if status == "proposed" and 7 <= age_days <= EXPIRY_DAYS:
        score -= 0.05
    return round(min(0.99, max(0.01, score)), 4)
