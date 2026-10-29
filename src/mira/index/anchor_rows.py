"""Row mapping shared by the SQLite and Postgres index stores."""

from __future__ import annotations

from mira.index.anchor_model import IndexedSymbol, IndexRun, SymbolAnchor


def anchor_from_row(row: tuple) -> SymbolAnchor:
    return SymbolAnchor(
        id=row[0],
        symbol_kind=row[1] or "",
        qualified_name=row[2] or "",
        path=row[3] or "",
        body_fingerprint=row[4] or "",
        body_simhash=row[5] or "",
        last_resolved_commit=row[6] or "",
        resolution_status=row[7] or "resolved",
        proposal_path=row[8] or "",
        proposal_qualified_name=row[9] or "",
        proposal_fingerprint=row[10] or "",
        proposal_score=float(row[11] or 0),
        proposal_reason=row[12] or "",
        created_at=float(row[13] or 0),
        updated_at=float(row[14] or 0),
    )


def anchor_tuple(anchor: SymbolAnchor, now: float) -> tuple:
    created = anchor.created_at or now
    return (
        anchor.id,
        anchor.symbol_kind,
        anchor.qualified_name,
        anchor.path,
        anchor.body_fingerprint,
        anchor.body_simhash,
        anchor.last_resolved_commit,
        anchor.resolution_status,
        anchor.proposal_path,
        anchor.proposal_qualified_name,
        anchor.proposal_fingerprint,
        anchor.proposal_score,
        anchor.proposal_reason,
        created,
        now,
    )


def indexed_from_row(row: tuple) -> IndexedSymbol:
    return IndexedSymbol(
        path=row[0],
        name=row[1],
        qualified_name=row[3] or row[1],
        kind=row[2] or "",
        body_fingerprint=row[4] or "",
        body_simhash=row[5] or "",
        last_seen_commit=row[6] or "",
    )


def index_run_from_row(row: tuple) -> IndexRun:
    return IndexRun(
        indexed_at=float(row[0] or 0),
        commit=row[1] or "",
        file_count=int(row[2] or 0),
        symbol_count=int(row[3] or 0),
    )
