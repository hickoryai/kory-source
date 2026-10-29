"""Anchor and indexed-symbol records shared by the stores and the resolver."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IndexedSymbol:
    """One symbol row used while resolving anchors."""

    path: str
    name: str
    qualified_name: str
    kind: str
    body_fingerprint: str
    body_simhash: str
    last_seen_commit: str = ""


@dataclass
class SymbolAnchor:
    """A stable link to a symbol. Rows are updated in place and never deleted."""

    id: str
    symbol_kind: str
    qualified_name: str
    path: str
    body_fingerprint: str = ""
    body_simhash: str = ""
    last_resolved_commit: str = ""
    # resolved | pending | dangling | retired
    resolution_status: str = "resolved"
    proposal_path: str = ""
    proposal_qualified_name: str = ""
    proposal_fingerprint: str = ""
    proposal_score: float = 0.0
    proposal_reason: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class IndexRun:
    """One completed index of a repository."""

    indexed_at: float
    commit: str
    file_count: int
    symbol_count: int


@dataclass
class RelinkProposal:
    """A candidate relink that a human still has to confirm."""

    path: str
    qualified_name: str
    fingerprint: str
    score: float
    reason: str
