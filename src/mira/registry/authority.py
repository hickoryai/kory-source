"""Authority tiers and relation-specific cycle rules (HAI-12825, plan section 3).

Ratified by a human on 2026-11-05 (HAI-12825 revision 2 approval receipt
https://linear.app/hickoryai/issue/HAI-12825#comment-f35a8621).

Tiers, strongest first, each within the same scope and applicable revision:

1. ``explicit_human``: a person's attributable statement, or a scoped human
   approval naming the decision. Both are cited.
2. ``human_directed_agent``: an agent wrote it under a source-backed, bounded
   human direction. The agent is the author, the person the director.
3. ``bot_proposal``: autonomous suggestions, automated reviews, summaries.
4. ``inference``: code-derived rationale or similarity. Never approval or
   supersession.

A merge is realization, never approval. Newer, more confident, repeated, or
merged statements never silently override an approved decision; only an
approved ``supersedes`` / ``corrects`` edge does, and it leaves the older
decision and its evidence visible.

Cycles: ``supersedes`` and ``corrects`` are directed ancestry edges (newer ->
older). Their UNION is a DAG. Everything else (``compatible_with``,
``conflicts_with``, ``constrains``, ``derived_from``, ``relates``) is a
cross-link and may form cycles, which traversal handles with visited sets.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from mira.registry.model import ANCESTRY_TYPES, INACTIVE_REL_STATUSES, SYMMETRIC_TYPES

AUTHORITY_POLICY_VERSION = "authority/1"
TIERS = ("explicit_human", "human_directed_agent", "bot_proposal", "inference")
TIER_RANK = {t: i for i, t in enumerate(TIERS)}
CONFIDENCE_LABELS = ("high", "medium", "low", "unknown")


def _kind(attr: dict | None) -> tuple[str, str]:
    """(actor_kind, its state) from an attribution row or dict."""
    if not attr:
        return "unknown", "unknown"
    kind = attr.get("actor_kind") or "unknown"
    states = attr.get("field_states") or {}
    if isinstance(states, str):
        import json

        try:
            states = json.loads(states)
        except ValueError:
            states = {}
    state = (states.get("actor_kind") or {}).get("state") or (
        "unknown" if kind == "unknown" else "stated"
    )
    return kind, state


def classify(
    *,
    author: dict | None,
    director: dict | None = None,
    approvals: Iterable[dict] = (),
    stated_or_inferred: str = "stated",
) -> tuple[str, list[str]]:
    """The authority tier for one decision revision, and why.

    ``approvals`` are approver attributions tied to this decision's own scope.
    A PR approval or merge is not passed here: it does not name a decision.
    """
    reasons: list[str] = []
    if stated_or_inferred != "stated":
        return "inference", ["extracted by inference, not stated in the source"]
    human_approvals = [a for a in approvals if _kind(a)[0] == "human"]
    a_kind, a_state = _kind(author)
    d_kind, _ = _kind(director)
    if a_kind == "human":
        reasons.append(f"author is a person ({a_state})")
        if a_state != "stated":
            reasons.append("authorship is inferred from the posting account; unverified")
        return "explicit_human", reasons
    if human_approvals:
        reasons.append("a person approved this decision's scope")
        return "explicit_human", reasons
    if a_kind in ("bot", "automation") and d_kind == "human":
        reasons.append("agent author with a source-backed human director")
        return "human_directed_agent", reasons
    if a_kind in ("bot", "automation"):
        reasons.append("agent or automation author with no human direction or approval")
        return "bot_proposal", reasons
    reasons.append("author unknown; treated as a proposal")
    return "bot_proposal", reasons


def merge_is_approval() -> bool:
    """Policy constant, asserted by tests: a merge records realization only."""
    return False


def confidence_label(
    *, anchor_status: str = "", author_state: str = "unknown", history_complete: bool = False
) -> tuple[str, list[str]]:
    """Evidence certainty, not authority. Age never lowers it."""
    reasons = []
    if author_state == "unknown":
        reasons.append("author unknown")
    if anchor_status in ("dangling", "pending", "ambiguous"):
        reasons.append(f"anchor {anchor_status}")
    if not history_complete:
        reasons.append("source history incomplete")
    if not reasons:
        return "high", ["stated author, resolved anchor, complete history"]
    if len(reasons) == 1 and "source history incomplete" in reasons:
        return "medium", reasons
    return ("low" if len(reasons) < 3 else "unknown"), reasons


# ── cycles ──────────────────────────────────────────────────────────────


def operative(status: str) -> bool:
    return status not in INACTIVE_REL_STATUSES


def pair_key(a: str, b: str, type: str) -> str:
    """The canonical unordered pair for a symmetric relationship."""
    if type not in SYMMETRIC_TYPES:
        return ""
    lo, hi = sorted((a, b))
    return f"{type}:{lo}:{hi}"


def ancestry_reaches(start: str, goal: str, edges_out: Callable[[str], Iterable[str]]) -> bool:
    """True when ``goal`` is reachable from ``start`` over the ancestry union."""
    seen: set[str] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node == goal:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(edges_out(node))
    return False


class CycleError(ValueError):
    pass


def check_ancestry_edge(
    source: str, target: str, type: str, edges_out: Callable[[str], Iterable[str]]
) -> None:
    """Refuse an ancestry edge that would put a node in its own ancestry.

    ``edges_out(node)`` yields targets of operative ``supersedes`` OR ``corrects``
    edges out of ``node``; proposed edges count, rejected ones do not.
    """
    if type not in ANCESTRY_TYPES:
        return
    if source == target:
        raise CycleError("a decision cannot supersede or correct itself")
    if ancestry_reaches(target, source, edges_out):
        raise CycleError(f"that {type} edge would make a supersedes/corrects cycle")


def check_symmetric(source: str, target: str, type: str) -> None:
    if type in SYMMETRIC_TYPES and source == target:
        raise ValueError(f"{type} needs two different decisions")


def topological_history(nodes: Iterable[str], ancestry: Iterable[tuple[str, str]]) -> list[str]:
    """Oldest first over ancestry edges only (newer -> older); ties keep input order.

    Cross-links are not ordered here; a mixed graph may contain cycles.
    """
    order = list(dict.fromkeys(nodes))
    older_of: dict[str, set[str]] = {n: set() for n in order}
    for newer, older in ancestry:
        if newer in older_of and older in older_of:
            older_of[newer].add(older)
    out: list[str] = []
    done: set[str] = set()
    remaining = list(order)
    while remaining:
        progressed = False
        for n in list(remaining):
            if older_of[n] <= done:
                out.append(n)
                done.add(n)
                remaining.remove(n)
                progressed = True
        if not progressed:  # cannot happen when the ancestry union is a DAG
            out.extend(remaining)
            break
    return out
