"""Normalized body fingerprints for rename-safe symbol identity.

The sha256 is an exact identity of the normalized body. The simhash supports
a similarity threshold when a body was edited as well as moved.
"""

from __future__ import annotations

import hashlib
import re

# Unique candidates at or above this similarity are resolved automatically.
FINGERPRINT_THRESHOLD = 0.85
# Weaker matches are recorded as a pending relink proposal, not applied.
PROPOSAL_FLOOR = 0.55

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//.*?$", re.MULTILINE)
_HASH_COMMENT = re.compile(r"#.*?$", re.MULTILINE)
_WHITESPACE = re.compile(r"\s+")


def normalize_body(source: str, name: str) -> str:
    """Collapse a symbol body so a rename of the symbol still matches.

    Comments and insignificant whitespace are removed. The symbol's own name
    is replaced, so the fingerprint follows the body rather than the spelling
    of the declaration.
    """
    text = _BLOCK_COMMENT.sub(" ", source)
    text = _LINE_COMMENT.sub(" ", text)
    text = _HASH_COMMENT.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip().lower()
    if name:
        text = re.sub(rf"\b{re.escape(name.lower())}\b", "#", text)
    return text


def simhash_hex(text: str) -> str:
    """64-bit simhash of whitespace-separated tokens, as 16 hex chars."""
    vector = [0] * 64
    tokens = text.split() or [""]
    for token in tokens:
        digest = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:16], 16)
        for bit in range(64):
            vector[bit] += 1 if (digest >> bit) & 1 else -1
    value = 0
    for bit, weight in enumerate(vector):
        if weight > 0:
            value |= 1 << bit
    return f"{value:016x}"


def body_fingerprint(source: str, name: str) -> tuple[str, str]:
    """Return ``(sha256, simhash)`` for a symbol body."""
    normal = normalize_body(source, name)
    digest = hashlib.sha256(normal.encode("utf-8")).hexdigest()
    return digest, simhash_hex(normal)


def simhash_similarity(left: str, right: str) -> float:
    """Return 1.0 for identical simhashes and 0.0 for missing values."""
    if not left or not right or len(left) != 16 or len(right) != 16:
        return 0.0
    try:
        left_bits = int(left, 16)
        right_bits = int(right, 16)
    except ValueError:
        return 0.0
    distance = (left_bits ^ right_bits).bit_count()
    return 1.0 - (distance / 64.0)


def fingerprint_score(
    anchor_fingerprint: str,
    anchor_simhash: str,
    symbol_fingerprint: str,
    symbol_simhash: str,
) -> float:
    """Exact normalized-body match scores 1. Otherwise return simhash similarity."""
    if anchor_fingerprint and anchor_fingerprint == symbol_fingerprint:
        return 1.0
    return simhash_similarity(anchor_simhash, symbol_simhash)
