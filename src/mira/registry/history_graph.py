"""Bounded traversal of a decision's whole relationship history (implementation contract).

``Registry.chain`` follows the first ``supersedes`` edge each way and is kept
for the legacy view. This walks every decision relationship in both
directions:

- each node is expanded once and each edge (or symmetric pair) is emitted once,
  so a mixed graph with cycles terminates and is not called invalid;
- ``supersedes`` / ``corrects`` give the ancestry order (oldest first), a
  topological order over ancestry edges only; every other type is a cross-link;
- forks, multiple predecessors and unresolved conflicts are all returned, with
  no first-edge winner;
- limits are reported (``truncated``) instead of silently shortening the walk;
- invisible endpoints (another corpus, no access) are omitted and counted only
  as ``hidden_links`` so the caller knows the history is incomplete without
  learning what was hidden.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from mira.registry import access as _access
from mira.registry.access import AccessContext
from mira.registry.authority import pair_key, topological_history
from mira.registry.model import ANCESTRY_TYPES, SYMMETRIC_TYPES, NotFound

DECISION_RELATIONS = (
    "supersedes",
    "corrects",
    "conflicts_with",
    "compatible_with",
    "constrains",
    "derived_from",
    "relates",
)
NODE_FIELDS = (
    "id",
    "kind",
    "title",
    "status",
    "realization",
    "repo",
    "source_url",
    "source_system",
    "authority_tier",
    "author",
    "author_kind",
    "decided_by",
    "created_at",
)


def walk(
    registry: Any,
    start: list[str],
    access: AccessContext | None,
    *,
    max_nodes: int = 200,
    max_depth: int = 8,
    types: tuple[str, ...] = DECISION_RELATIONS,
) -> dict:
    nodes: dict[str, dict] = {}
    edges: dict[str, dict] = {}
    depth: dict[str, int] = {}
    hidden = 0
    truncated: list[str] = []
    queue: deque[str] = deque()
    for oid in start:
        try:
            obj = registry.get_visible(oid, access)
        except NotFound:
            continue
        nodes[oid] = {k: obj.get(k) for k in NODE_FIELDS}
        depth[oid] = 0
        queue.append(oid)
    expanded: set[str] = set()
    while queue:
        node = queue.popleft()
        if node in expanded:
            continue
        expanded.add(node)
        if depth[node] >= max_depth:
            truncated.append(f"depth limit {max_depth} at {node}")
            continue
        for rel in registry.relationships(node):
            if rel["type"] not in types:
                continue
            other = rel["target_id"] if rel["source_id"] == node else rel["source_id"]
            try:
                other_obj = registry.get(other)
            except NotFound:
                hidden += 1
                continue
            if not _access.visible(other_obj, access):
                hidden += 1
                continue
            key = pair_key(rel["source_id"], rel["target_id"], rel["type"]) or rel["id"]
            if key not in edges:
                edges[key] = {
                    "id": rel["id"],
                    "type": rel["type"],
                    "source_id": rel["source_id"],
                    "target_id": rel["target_id"],
                    "status": rel["status"],
                    "relation": rel.get("relation") or "",
                    "new_clause": rel.get("new_clause") or "",
                    "existing_clause": rel.get("existing_clause") or "",
                    "ancestry": rel["type"] in ANCESTRY_TYPES,
                    "symmetric": rel["type"] in SYMMETRIC_TYPES,
                }
            if other not in nodes:
                if len(nodes) >= max_nodes:
                    truncated.append(f"node limit {max_nodes}")
                    continue
                nodes[other] = {k: other_obj.get(k) for k in NODE_FIELDS}
                depth[other] = depth[node] + 1
                queue.append(other)
    ancestry = [
        (e["source_id"], e["target_id"])
        for e in edges.values()
        if e["ancestry"] and e["status"] not in ("rejected", "withdrawn", "kept_existing")
    ]
    order = topological_history(sorted(nodes, key=lambda n: nodes[n]["created_at"] or 0), ancestry)
    replaced = {older for _, older in ancestry}
    return {
        "start": [s for s in start if s in nodes],
        "nodes": nodes,
        "edges": list(edges.values()),
        "history_order": order,
        # Current heads: no operative supersedes/corrects edge points at them.
        "heads": [n for n in order if n not in replaced],
        "hidden_links": hidden,
        "truncated": sorted(set(truncated)),
        "complete": not truncated and hidden == 0,
    }
