"""Attach deterministic symbol identities to a file summary.

The LLM summarizer still supplies descriptions. Bodies, fingerprints, and
qualified names come from the extractor so they survive a rename.
"""

from __future__ import annotations

from collections import Counter

from mira.index.anchor_model import IndexedSymbol
from mira.index.extract import extract_symbols
from mira.index.fingerprint import body_fingerprint
from mira.index.relationships import github_repo_mentions
from mira.index.store import ExternalRef, FileSummary, SymbolInfo


def indexed_from_info(path: str, symbol: SymbolInfo) -> IndexedSymbol:
    return IndexedSymbol(
        path=path,
        name=symbol.name,
        qualified_name=symbol.qualified_name or symbol.name,
        kind=symbol.kind,
        body_fingerprint=symbol.body_fingerprint,
        body_simhash=symbol.body_simhash,
        last_seen_commit=symbol.last_seen_commit,
    )


def annotate_summary(summary: FileSummary, content: str, commit: str) -> FileSummary:
    """Fill fingerprints from the extractor and record GitHub repo mentions.

    When the extractor finds nothing, existing symbols are kept and stamped
    with the commit. Descriptions already on the summary are copied onto the
    extracted symbols by bare name.
    """
    language = summary.language or ""
    spans = extract_symbols(content, language) if content else []
    descriptions = {symbol.name: symbol.description for symbol in summary.symbols}
    qualified_descriptions = {
        symbol.qualified_name: symbol.description
        for symbol in summary.symbols
        if symbol.qualified_name
    }
    if spans:
        extracted: list[SymbolInfo] = []
        for span in spans:
            qualified = span.qualified_name or span.name
            digest, simhash = body_fingerprint(span.source, span.name)
            first_line = span.source.splitlines()[0] if span.source else ""
            extracted.append(
                SymbolInfo(
                    name=span.name,
                    kind=span.kind,
                    signature=first_line[:180],
                    description=descriptions.get(span.name)
                    or qualified_descriptions.get(qualified)
                    or "",
                    qualified_name=qualified,
                    body_fingerprint=digest,
                    body_simhash=simhash,
                    last_seen_commit=commit,
                )
            )
        _disambiguate_names(extracted)
        summary.symbols = extracted
    else:
        for symbol in summary.symbols:
            symbol.last_seen_commit = commit or symbol.last_seen_commit
            if not symbol.qualified_name:
                symbol.qualified_name = symbol.name

    seen = {(ref.kind, ref.target) for ref in summary.external_refs}
    for repo in github_repo_mentions(content):
        target = f"https://github.com/{repo}"
        key = ("git_url", target)
        if key in seen:
            continue
        seen.add(key)
        summary.external_refs.append(
            ExternalRef(file_path=summary.path, kind="git_url", target=target, description="")
        )
    return summary


def _disambiguate_names(symbols: list[SymbolInfo]) -> None:
    """Keep the primary key unique when one file has two methods of one name."""
    counts = Counter(symbol.name for symbol in symbols)
    for symbol in symbols:
        if counts[symbol.name] > 1 and symbol.qualified_name:
            symbol.name = symbol.qualified_name
