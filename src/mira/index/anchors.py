"""Resolve symbol anchors after each index run.

Order is fixed: exact path and name, then a git rename, then a body-fingerprint
match above the threshold, then a pending relink proposal. An anchor that
matches nothing is marked dangling. The row is kept either way.
"""

from __future__ import annotations

import logging
import subprocess
import uuid
from pathlib import Path
from typing import Protocol

from mira.index.anchor_model import IndexedSymbol, RelinkProposal, SymbolAnchor
from mira.index.fingerprint import (
    FINGERPRINT_THRESHOLD,
    PROPOSAL_FLOOR,
    fingerprint_score,
)

logger = logging.getLogger(__name__)


class RelinkProposer(Protocol):
    """Turn an unresolved anchor into a proposal a human can confirm.

    The seat-backed model provider is the intended implementation. Until that
    provider is wired, ``UnwiredRelinkProposer`` records a similarity
    candidate and leaves it pending.
    """

    def propose(
        self,
        anchor: SymbolAnchor,
        candidates: list[tuple[float, IndexedSymbol]],
    ) -> RelinkProposal | None:
        """Return a proposal, or None when the anchor should stay dangling."""


class UnwiredRelinkProposer:
    """Record the best body-similarity candidate without calling a model."""

    def propose(
        self,
        anchor: SymbolAnchor,
        candidates: list[tuple[float, IndexedSymbol]],
    ) -> RelinkProposal | None:
        viable = [(score, sym) for score, sym in candidates if score >= PROPOSAL_FLOOR]
        if not viable:
            return None
        score, sym = viable[0]
        return RelinkProposal(
            path=sym.path,
            qualified_name=sym.qualified_name or sym.name,
            fingerprint=sym.body_fingerprint,
            score=score,
            reason="model unwired; pending body-similarity candidate",
        )


def new_anchor_id() -> str:
    return uuid.uuid4().hex


def anchor_from_symbol(
    symbol: IndexedSymbol, *, commit: str, anchor_id: str | None = None
) -> SymbolAnchor:
    """Bind a new resolved anchor to a symbol that exists in the current index."""
    return SymbolAnchor(
        id=anchor_id or new_anchor_id(),
        symbol_kind=symbol.kind,
        qualified_name=symbol.qualified_name or symbol.name,
        path=symbol.path,
        body_fingerprint=symbol.body_fingerprint,
        body_simhash=symbol.body_simhash,
        last_resolved_commit=commit or symbol.last_seen_commit,
        resolution_status="resolved",
    )


def names_match(anchor_name: str, symbol: IndexedSymbol) -> bool:
    """True when the anchor's stored name still identifies this symbol.

    A bare name still matches after the extractor starts qualifying methods,
    so an extractor upgrade does not by itself dangle the anchor.
    """
    if not anchor_name:
        return False
    if anchor_name in {symbol.qualified_name, symbol.name}:
        return True
    return symbol.name == anchor_name and symbol.qualified_name.endswith("." + anchor_name)


def git_rename_map(checkout: Path | None, old_commit: str, new_commit: str) -> dict[str, str]:
    """Map old paths to new paths using ``git diff -M`` between two commits."""
    if checkout is None or not old_commit or not new_commit or old_commit == new_commit:
        return {}
    try:
        proc = subprocess.run(
            ["git", "-C", str(checkout), "diff", "-M", "--name-status", old_commit, new_commit],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("git rename detection failed: %s", exc)
        return {}
    if proc.returncode != 0:
        logger.warning("git diff -M exited %s: %s", proc.returncode, proc.stderr.strip()[:300])
        return {}
    renamed: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0].startswith("R"):
            renamed[parts[1]] = parts[2]
    return renamed


def confirm_anchor(anchor: SymbolAnchor) -> SymbolAnchor:
    """Apply a pending proposal. The anchor stays in the table."""
    if anchor.resolution_status != "pending":
        raise ValueError("only a pending anchor can be confirmed")
    if not anchor.proposal_path or not anchor.proposal_qualified_name:
        raise ValueError("pending anchor has no proposal to confirm")
    anchor.path = anchor.proposal_path
    anchor.qualified_name = anchor.proposal_qualified_name
    if anchor.proposal_fingerprint:
        anchor.body_fingerprint = anchor.proposal_fingerprint
    anchor.resolution_status = "resolved"
    _clear_proposal(anchor)
    return anchor


def retire_anchor(anchor: SymbolAnchor) -> SymbolAnchor:
    """Stop resolving an anchor. The row stays so the link is not lost."""
    anchor.resolution_status = "retired"
    return anchor


def resolve_anchors(
    anchors: list[SymbolAnchor],
    symbols: list[IndexedSymbol],
    *,
    commit: str,
    rename_maps: dict[str, dict[str, str]] | None = None,
    proposer: RelinkProposer | None = None,
) -> list[SymbolAnchor]:
    """Re-resolve every non-retired anchor. None are removed."""
    maps = rename_maps or {}
    if proposer is None:
        proposer = UnwiredRelinkProposer()
    by_path: dict[str, list[IndexedSymbol]] = {}
    for symbol in symbols:
        by_path.setdefault(symbol.path, []).append(symbol)

    for anchor in anchors:
        if anchor.resolution_status == "retired":
            continue
        exact = _first_named(anchor, by_path.get(anchor.path, []))
        if exact is not None:
            _adopt(anchor, exact, commit)
            continue
        renamed = maps.get(anchor.last_resolved_commit, {}).get(anchor.path, "")
        if renamed and renamed != anchor.path:
            moved = _first_named(anchor, by_path.get(renamed, []))
            if moved is not None:
                _adopt(anchor, moved, commit)
                continue
        scored = _scored_candidates(anchor, symbols)
        chosen = _unique_above_threshold(anchor, scored)
        if chosen is not None:
            _adopt(anchor, chosen, commit)
            continue
        proposal = proposer.propose(anchor, scored)
        if proposal is None:
            anchor.resolution_status = "dangling"
            _clear_proposal(anchor)
            continue
        anchor.resolution_status = "pending"
        anchor.proposal_path = proposal.path
        anchor.proposal_qualified_name = proposal.qualified_name
        anchor.proposal_fingerprint = proposal.fingerprint
        anchor.proposal_score = proposal.score
        anchor.proposal_reason = proposal.reason
    return anchors


def _first_named(anchor: SymbolAnchor, symbols: list[IndexedSymbol]) -> IndexedSymbol | None:
    for symbol in symbols:
        if names_match(anchor.qualified_name, symbol):
            return symbol
    return None


def _scored_candidates(
    anchor: SymbolAnchor, symbols: list[IndexedSymbol]
) -> list[tuple[float, IndexedSymbol]]:
    scored: list[tuple[float, IndexedSymbol]] = []
    for symbol in symbols:
        if not symbol.body_fingerprint and not symbol.body_simhash:
            continue
        score = fingerprint_score(
            anchor.body_fingerprint,
            anchor.body_simhash,
            symbol.body_fingerprint,
            symbol.body_simhash,
        )
        if score > 0:
            scored.append((score, symbol))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def _unique_above_threshold(
    anchor: SymbolAnchor, scored: list[tuple[float, IndexedSymbol]]
) -> IndexedSymbol | None:
    above = [(score, symbol) for score, symbol in scored if score >= FINGERPRINT_THRESHOLD]
    if not above:
        return None
    best = above[0][0]
    top = [symbol for score, symbol in above if score == best]
    if len(top) == 1:
        return top[0]
    bare = anchor.qualified_name.rsplit(".", 1)[-1]
    named = [
        symbol
        for symbol in top
        if symbol.name == bare or symbol.qualified_name == anchor.qualified_name
    ]
    if len(named) == 1:
        return named[0]
    return None


def _adopt(anchor: SymbolAnchor, symbol: IndexedSymbol, commit: str) -> None:
    anchor.path = symbol.path
    anchor.qualified_name = symbol.qualified_name or symbol.name
    if symbol.kind:
        anchor.symbol_kind = symbol.kind
    anchor.body_fingerprint = symbol.body_fingerprint
    anchor.body_simhash = symbol.body_simhash
    anchor.last_resolved_commit = commit or symbol.last_seen_commit or anchor.last_resolved_commit
    anchor.resolution_status = "resolved"
    _clear_proposal(anchor)


def _clear_proposal(anchor: SymbolAnchor) -> None:
    anchor.proposal_path = ""
    anchor.proposal_qualified_name = ""
    anchor.proposal_fingerprint = ""
    anchor.proposal_score = 0.0
    anchor.proposal_reason = ""
