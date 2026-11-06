"""Cited, read-only answers to source-history questions (implementation contract).

The central question: "Does PR #X override any prior decisions?"

1. Freeze the PR: base, head, merge base, title and body, changed files with
   renames and deletions, through the GET-only ``SnapshotReader``. A head that
   moved since the caller looked (``expected_head_sha``) is reported, not
   silently re-read.
2. Collect decisions anchored on any old or new path the PR touches (live
   anchors and historical anchor evidence), decisions stated in the PR's own
   imported revisions, and decisions recorded on tickets the PR links
   (unanchored: they are labeled so, never as a code match).
3. Walk every decision relationship with ``history_graph.walk``.
4. Judge each prior decision separately:
   ``confirmed_supersession`` (an approved, active supersedes/corrects edge
   whose newer side a person approved), ``proposed_supersession``,
   ``conflict``, ``compatible``, ``no_supported_override_found``, or
   ``insufficient_evidence``. "The PR changes code contrary to D" is a
   conflict; only a human-approved edge is an override.
5. Cite the PR revision and each decision's original URL, exact quote,
   evidence revision, and every attribution role with original and normalized
   labels and unknown flags. Importer and extractor are a separate section.
6. The narrative is deterministic over the structured findings. Every
   sentence lists evidence ids, and the renderer refuses an id that is not in
   the citations.

Nothing here imports, approves, supersedes, re-anchors, reviews, or writes to
GitHub or Linear. Callers need complete-corpus access; without it the answer
is the same uniform denial for every PR.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from mira.registry import access as _access
from mira.registry.access import AccessContext, AccessDenied
from mira.registry.history_graph import walk
from mira.registry.imports import ImportRuns
from mira.registry.model import CLAIM_KINDS, NotFound, RegistryError
from mira.registry.provenance import ProvenanceStore

VERDICTS = (
    "confirmed_supersession",
    "proposed_supersession",
    "conflict",
    "compatible",
    "no_supported_override_found",
    "insufficient_evidence",
)
_RANK = {v: i for i, v in enumerate(VERDICTS)}
_UNCERTAIN_ANCHORS = ("pending", "ambiguous", "dangling", "unsupported")
# Relationship states that are audit history only: never a current verdict.
_INACTIVE_LINKS = ("rejected", "withdrawn", "kept_existing", "superseded_existing")
_RELATIONSHIP_VERDICTS = (
    "confirmed_supersession",
    "proposed_supersession",
    "conflict",
    "compatible",
)
_PR_REF = re.compile(
    r"(?:(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+))?\s*(?:#|pr\s*#?|pull\s*(?:request)?\s*#?)"
    r"\s*(?P<num>\d+)",
    re.I,
)
_OVERRIDE_Q = re.compile(r"\b(overrid|supersed|replac|revers|contradict|conflict|break)", re.I)
_HISTORY_Q = re.compile(r"\b(history|why|decid|decision|explain|reason|rationale)", re.I)
_TICKET = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")

SnapshotSource = Callable[[str, int], Awaitable[dict]]


class CitationError(RegistryError):
    """A narrative sentence cited evidence that is not in the answer."""


def changed_files(diff_text: str, changed_paths: Iterable[str]) -> list[dict]:
    """GitHub-style ``filename`` / ``status`` / ``previous_filename`` from a frozen diff."""
    out: dict[str, dict] = {p: {"filename": p, "status": "modified"} for p in changed_paths}
    current: dict | None = None
    for line in diff_text.splitlines():
        if line.startswith("diff --git a/"):
            rest = line[len("diff --git a/") :]
            old, _, new = rest.partition(" b/")
            current = out.setdefault(new, {"filename": new, "status": "modified"})
            if old != new:
                current.update(status="renamed", previous_filename=old)
        elif current is not None and line.startswith("new file mode"):
            current["status"] = "added"
        elif current is not None and line.startswith("deleted file mode"):
            current["status"] = "removed"
    return list(out.values())


async def default_snapshot(repo: str, number: int) -> dict:
    """Freeze a PR through the review API's GET-only reader."""
    from mira.review_api.service import current

    service = current()
    if service is None or service.reader_factory is None:
        raise RegistryError("no GitHub reader is configured for PR snapshots")
    owner, name = repo.split("/", 1)
    reader = await service.reader_factory(owner, name)
    meta = await reader.get_pull(number)
    snap = await reader.freeze(meta)
    return {
        "repo": repo,
        "number": meta.number,
        "url": meta.url,
        "title": meta.title,
        "body": meta.description,
        "state": meta.state,
        "author": meta.author,
        "base_sha": meta.base_sha,
        "head_sha": meta.head_sha,
        "merge_base_sha": snap.merge_base_sha,
        "files": changed_files(snap.diff_text, snap.changed_paths),
        "unavailable_paths": list(snap.unavailable_paths),
    }


class QuestionService:
    def __init__(self, registry: Any, *, snapshot: SnapshotSource | None = None) -> None:
        self.registry = registry
        self.store = ProvenanceStore(registry)
        self.snapshot = snapshot or default_snapshot

    # ── entry points ──────────────────────────────────────────────────

    async def ask(
        self,
        question: str,
        access: AccessContext | None,
        *,
        repo: str | None = None,
        pr_number: int | None = None,
        paths: list[str] | None = None,
        symbols: list[str] | None = None,
        as_of_commit: str | None = None,
    ) -> dict:
        self._require(access)
        question = (question or "").strip()
        ref = _PR_REF.search(question)
        if pr_number is None and ref:
            pr_number = int(ref["num"])
            repo = repo or ref["repo"]
        if pr_number is not None and (_OVERRIDE_Q.search(question) or not (paths or symbols)):
            if not repo:
                return self._needs("repo (owner/name) for that pull request")
            out = await self.explain_pr_decisions(repo, int(pr_number), access)
            out["question"] = question
            return out
        if (paths or symbols) and repo:
            return self.decision_history(
                repo,
                paths or [],
                symbols or [],
                access,
                question=question,
                as_of_commit=as_of_commit,
            )
        if _HISTORY_Q.search(question) and not repo:
            return self._needs("repo plus a path or symbol, or a pull request number")
        return self._needs("a pull request number, or repo plus paths or symbols")

    async def explain_pr_decisions(
        self,
        repo: str,
        pr_number: int,
        access: AccessContext | None,
        *,
        expected_head_sha: str | None = None,
    ) -> dict:
        self._require(access)
        policy = _access.current_policy()
        if policy is None or not policy.admits_repository(repo):
            raise AccessDenied()
        try:
            snap = await self.snapshot(repo, pr_number)
        except Exception as exc:  # reported, never guessed around
            return self._insufficient(repo, pr_number, f"PR snapshot unavailable: {exc}")
        limits: list[str] = []
        # Revision limits mean the evidence is not about the frozen head; no
        # affirmative verdict can stand on it.
        revision_limits: list[str] = []
        if expected_head_sha and snap["head_sha"] != expected_head_sha:
            revision_limits.append(
                f"head moved: expected {expected_head_sha[:12]}, frozen {snap['head_sha'][:12]}"
            )
        if snap.get("unavailable_paths"):
            limits.append(f"{len(snap['unavailable_paths'])} changed files have no patch")
        files = snap["files"]
        paths = sorted(
            {f["filename"] for f in files}
            | {f["previous_filename"] for f in files if f.get("previous_filename")}
        )
        prior, anchored = self._anchored(repo, paths, access)
        stated, stale_stated, import_limit = self._stated_in_pr(repo, pr_number, snap, access)
        if import_limit:
            revision_limits.append(import_limit)
        tickets = self._ticket_decisions(f"{snap.get('title', '')}\n{snap.get('body', '')}", access)
        for oid in tickets:
            prior.setdefault(oid, {"why": "linked ticket (unanchored)", "anchors": []})
        for oid in stated | stale_stated:
            prior.pop(oid, None)
        graph = walk(self.registry, sorted({*prior, *stated}), access)
        if graph["truncated"]:
            limits += graph["truncated"]
        if graph["hidden_links"]:
            limits.append("some related decisions are outside your readable corpus")
        coverage = self._coverage(repo, access)
        if not coverage["github_complete"]:
            limits.append("GitHub history import for this repository is not complete")
        receipts: dict[str, dict] = {}
        findings = [
            self._judge(oid, info, stated, graph, anchored, access, receipts)
            for oid, info in prior.items()
        ]
        findings.sort(key=lambda f: (_RANK[f["verdict"]], f["decision"]["id"]))
        verdict = self._overall(findings, limits + revision_limits, bool(revision_limits))
        limits = revision_limits + limits
        citations = self._citations(
            [f["decision"]["id"] for f in findings] + sorted(stated), access
        )
        answer = {
            "kind": "pr_decisions",
            "repo": repo,
            "pr": {
                "number": pr_number,
                "url": snap.get("url", ""),
                "title": snap.get("title", ""),
                "base_sha": snap["base_sha"],
                "head_sha": snap["head_sha"],
                "merge_base_sha": snap.get("merge_base_sha", ""),
                "changed_files": files,
            },
            "verdict": verdict,
            "findings": findings,
            "pr_decisions": sorted(stated),
            "stale_pr_decisions": sorted(stale_stated),
            "graph": graph,
            "coverage": coverage,
            "limits": limits,
            "revision_bound": not revision_limits,
            "citations": citations,
            "receipts": receipts,
            "read_only": True,
        }
        answer["narrative"] = render_narrative(answer)
        return answer

    def decision_history(
        self,
        repo: str,
        paths: list[str],
        symbols: list[str],
        access: AccessContext | None,
        *,
        question: str = "",
        as_of_commit: str | None = None,
    ) -> dict:
        self._require(access)
        prior, _ = self._anchored(repo, paths, access, symbols)
        graph = walk(self.registry, sorted(prior), access)
        limits = list(graph["truncated"])
        if graph["hidden_links"]:
            limits.append("some related decisions are outside your readable corpus")
        ids = list(graph["history_order"])
        verdict = "insufficient_evidence" if not ids else "history"
        if as_of_commit:
            # Not supported yet: the registry answers about its current state, so
            # presenting that as the history at a past commit would be false proof.
            verdict = "insufficient_evidence"
            limits.insert(
                0,
                f"as_of_commit {as_of_commit[:12]} is not supported: these are current "
                "registry records, not proof of what held at that commit",
            )
        answer = {
            "kind": "decision_history",
            "question": question,
            "repo": repo,
            "paths": paths,
            "symbols": symbols,
            "as_of_commit": as_of_commit or "",
            "as_of_supported": False,
            "verdict": verdict,
            "graph": graph,
            "limits": limits,
            "citations": self._citations(ids, access),
            "read_only": True,
        }
        answer["narrative"] = render_narrative(answer)
        return answer

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _require(access: AccessContext | None) -> None:
        if access is None or not access.has_corpus:
            raise AccessDenied()

    @staticmethod
    def _needs(what: str) -> dict:
        return {
            "kind": "needs_context",
            "verdict": "needs_context",
            "needs": what,
            "read_only": True,
        }

    def _insufficient(self, repo: str, number: int, why: str) -> dict:
        answer = {
            "kind": "pr_decisions",
            "repo": repo,
            "pr": {"number": number},
            "verdict": "insufficient_evidence",
            "findings": [],
            "limits": [why],
            "citations": {},
            "read_only": True,
        }
        answer["narrative"] = render_narrative(answer)
        return answer

    def _anchored(
        self,
        repo: str,
        paths: list[str],
        access: AccessContext | None,
        symbols: Iterable[str] = (),
    ) -> tuple[dict[str, dict], dict[str, list[dict]]]:
        found: dict[str, dict] = {}
        anchors: dict[str, list[dict]] = {}
        for oid in self.registry.objects_on_paths(repo, paths, symbols):
            found.setdefault(oid, {"why": "anchored on a changed path", "anchors": []})
        if paths:
            marks = ", ".join("?" for _ in paths)
            rows = self.registry.db.execute(
                "SELECT object_id, path, resolved_path, qualified_name, observed_commit, "
                "resolution_status, corpus_id FROM registry_anchor_evidence "
                f"WHERE LOWER(repository) = ? AND (path IN ({marks}) OR resolved_path IN ({marks}))",
                (repo.lower(), *paths, *paths),
            )
            for oid, path, rpath, qname, commit, status, corpus in rows:
                if not (access and access.can_read(corpus)):
                    continue
                found.setdefault(oid, {"why": "historical anchor on a changed path", "anchors": []})
                anchors.setdefault(oid, []).append(
                    {
                        "path": path,
                        "resolved_path": rpath,
                        "qualified_name": qname,
                        "observed_commit": commit,
                        "resolution_status": status,
                    }
                )
        out: dict[str, dict] = {}
        for oid, info in found.items():
            try:
                obj = self.registry.get_visible(oid, access)
            except NotFound:
                continue
            if obj["kind"] in CLAIM_KINDS and obj["status"] not in ("rejected",):
                info["anchors"] = anchors.get(oid, [])
                out[oid] = info
        return out, anchors

    def _stated_in_pr(
        self, repo: str, number: int, snap: dict, access
    ) -> tuple[set[str], set[str], str]:
        """Decisions this PR states at its frozen head: ``(current, stale, limit)``.

        A description revision counts only when it was observed at the frozen
        head. Conversation comments have no head and count. A review or inline
        comment on another commit is stale. If the description was never
        imported at this head, ``limit`` says so and no verdict may be
        affirmative.
        """
        prefix = f"{repo}#{number}"
        head = snap.get("head_sha") or ""
        rows = self.registry.db.execute(
            "SELECT e.object_id, r.source_type, r.head_sha, r.commit_sha FROM "
            "registry_decision_evidence e JOIN registry_source_revisions r "
            "ON r.evidence_id = e.evidence_id WHERE r.source_system = 'github' "
            "AND (r.source_id = ? OR r.parent_source_id = ?)",
            (prefix, prefix),
        )
        current: set[str] = set()
        stale: set[str] = set()
        for oid, source_type, rev_head, commit in rows:
            if source_type == "github_pr_description":
                at_head = rev_head == head
            elif source_type in ("github_review", "github_review_comment"):
                at_head = not commit or commit == head
            else:
                at_head = True
            (current if at_head else stale).add(oid)
        stale -= current
        heads = self.registry.db.execute(
            "SELECT head_sha FROM registry_source_revisions WHERE source_system = 'github' "
            "AND source_type = 'github_pr_description' AND source_id = ?",
            (prefix,),
        )
        imported = {h[0] for h in heads}
        limit = ""
        if not imported:
            limit = "this PR's description has not been imported; its own decisions are unknown"
        elif head not in imported:
            limit = (
                f"stale import: this PR was imported at {sorted(imported)[-1][:12]}, "
                f"not the frozen head {head[:12]}"
            )
        url = snap.get("url") or ""
        if url:  # registry decisions recorded on this PR's branch
            current |= {o["id"] for o in self.registry._select_visible(access, "pr = ?", (url,))}
        current = {i for i in current if self._visible(i, access)}
        stale = {i for i in stale if self._visible(i, access)} - current
        return current, stale, limit

    def _ticket_decisions(self, text: str, access) -> set[str]:
        ids: set[str] = set()
        for ticket in dict.fromkeys(_TICKET.findall(text)):
            try:
                work = self.registry.work_by_ticket(ticket, access)
            except NotFound:
                continue
            ids |= {d["id"] for d in self.registry._linked(work["id"], "records", access=access)}
        return ids

    def _visible(self, oid: str, access) -> bool:
        try:
            self.registry.get_visible(oid, access)
            return True
        except NotFound:
            return False

    def _coverage(self, repo: str, access) -> dict:
        runs = [
            r
            for r in ImportRuns(self.registry).coverage(access)
            if r["stream"] in (f"pulls:{repo}", f"repository:{repo}")
        ]
        return {
            "github_complete": any(
                r["stream"] == f"pulls:{repo}" and r["status"] == "complete" for r in runs
            ),
            "repository_runs": [
                {"run_id": r["run_id"], "stream": r["stream"], "status": r["status"]} for r in runs
            ],
        }

    def _relationship_receipt(self, edge: dict, access) -> dict:
        """The stored relationship and every evidence record for it."""
        rows = self.registry.db.execute(
            "SELECT created_by, created_at, resolved_by, resolved_at, origin FROM "
            "registry_relationships WHERE id = ?",
            (edge["id"],),
        )
        created_by, created_at, resolved_by, resolved_at, origin = rows[0] if rows else ("",) * 5
        return {
            "kind": "relationship",
            "id": edge["id"],
            "type": edge["type"],
            "status": edge["status"],
            "source_id": edge["source_id"],
            "target_id": edge["target_id"],
            "relation": edge.get("relation") or "",
            "created_by": created_by,
            "created_at": created_at,
            "resolved_by": resolved_by,
            "resolved_at": resolved_at,
            "origin": origin,
            "evidence": self.store.relationship_evidence(edge["id"], access),
        }

    def _approval_receipt(self, object_id: str, edge_receipt: dict, access) -> dict | None:
        """Proof a person approved ``object_id``, tied to the edge's activation.

        Needs a recorded approval lifecycle event with a human approver
        attribution, cited by a relationship-evidence record of this edge. A
        legacy ``decided_by`` alone is not proof.
        """
        approval = self.registry._latest_approval(object_id)
        if not approval:
            return None
        cited = any(r.get("approving_evidence_id") == approval for r in edge_receipt["evidence"])
        if not cited:
            return None
        approvers = [
            a
            for a in self.store.attributions(object_id, [approval], access)
            if a["role"] == "approver"
            and a["evidence_id"] == approval
            and a["actor_kind"] == "human"
        ]
        try:
            event = self.store.source_revision(approval, access)
        except NotFound:
            return None
        if not approvers:
            return None
        return {
            "kind": "approval",
            "id": approval,
            "object_id": object_id,
            "event": {k: event.get(k) for k in ("source_type", "source_labels", "observed_at")},
            "approvers": [_attr_view(a) for a in approvers],
        }

    def _judge(
        self,
        oid: str,
        info: dict,
        stated: set[str],
        graph: dict,
        anchored: dict[str, list[dict]],
        access,
        receipts: dict[str, dict],
    ) -> dict:
        obj = self.registry.get(oid)
        verdict = "no_supported_override_found"
        via: list[dict] = []
        inactive: list[dict] = []
        reasons: list[str] = []
        cited: list[str] = []
        unproven = False  # an edge exists but its proof does not
        for e in graph["edges"]:
            ends = {e["source_id"], e["target_id"]}
            if oid not in ends:
                continue
            other_id = next(iter((ends - {oid}) or {oid}))
            if other_id not in stated:
                continue
            if e["type"] not in ("supersedes", "corrects", "conflicts_with", "compatible_with"):
                continue
            if e["status"] in _INACTIVE_LINKS:
                # Audit history: shown, never a current verdict.
                inactive.append(e)
                continue
            receipt = self._relationship_receipt(e, access)
            if e["type"] in ("supersedes", "corrects") and e["target_id"] == oid:
                newer = self.registry.get(e["source_id"])
                approval = (
                    self._approval_receipt(newer["id"], receipt, access)
                    if e["status"] == "active" and newer["status"] == "approved"
                    else None
                )
                if approval is not None:
                    v = "confirmed_supersession"
                    who = ", ".join(a["original_identity"] or "?" for a in approval["approvers"])
                    reasons.append(f"{newer['id']} {e['type']} it; approval recorded for {who}")
                    receipts[approval["id"]] = approval
                    cited.append(approval["id"])
                elif e["status"] == "active" and newer["status"] == "approved":
                    v = "insufficient_evidence"
                    unproven = True
                    reasons.append(
                        f"{newer['id']} is marked approved but no recorded approval event backs "
                        "this edge (a legacy decided_by is not proof)"
                    )
                else:
                    v = "proposed_supersession"
                    reasons.append(f"{newer['id']} would {e['type'].rstrip('s')} it; not approved")
            elif e["type"] in ("supersedes", "corrects"):
                continue  # the PR's decision is the older side: not an override of this one
            elif e["type"] == "conflicts_with":
                v = "conflict"
                reasons.append(
                    f"{other_id} conflicts ({e['relation'] or 'unspecified'}); "
                    "a conflict is not an authorized override"
                )
            else:
                v = "compatible"
                reasons.append(f"{other_id} is recorded as compatible")
            receipts[receipt["id"]] = receipt
            cited.append(receipt["id"])
            via.append(e)
            if _RANK[v] < _RANK[verdict]:
                verdict = v
        uncertain = [
            a for a in anchored.get(oid, []) if a["resolution_status"] in _UNCERTAIN_ANCHORS
        ]
        if verdict == "no_supported_override_found" and unproven:
            verdict = "insufficient_evidence"
        elif verdict == "no_supported_override_found":
            if info["why"].startswith("linked ticket"):
                verdict = "insufficient_evidence"
                reasons.append("linked through a ticket, not anchored to the changed code")
            elif uncertain:
                verdict = "insufficient_evidence"
                reasons.append(
                    "its anchor is uncertain ("
                    + ", ".join(sorted({a["resolution_status"] for a in uncertain}))
                    + ")"
                )
            else:
                reasons.append(
                    "the PR touches its anchored code; no current supersession, "
                    "correction, conflict or compatibility with this PR's decisions"
                )
        if inactive:
            reasons.append(
                f"{len(inactive)} inactive link(s) kept as history: "
                + ", ".join(f"{e['type']} {e['status']}" for e in inactive)
            )
        return {
            "decision": {
                k: obj.get(k)
                for k in (
                    "id",
                    "title",
                    "status",
                    "realization",
                    "authority_tier",
                    "source_url",
                    "repo",
                )
            },
            "verdict": verdict,
            "why_considered": info["why"],
            "anchors": info["anchors"],
            "via": via,
            "inactive_links": inactive,
            "receipts": cited,
            "reasons": reasons,
        }

    @staticmethod
    def _overall(findings: list[dict], limits: list[str], revision_unbound: bool = False) -> str:
        """Strongest finding, but never affirmative on evidence not bound to the head."""
        if revision_unbound:
            return "insufficient_evidence"
        if not findings:
            return "insufficient_evidence" if limits else "no_supported_override_found"
        best = min(findings, key=lambda f: _RANK[f["verdict"]])["verdict"]
        if best == "no_supported_override_found" and (
            limits or any(f["verdict"] == "insufficient_evidence" for f in findings)
        ):
            return "insufficient_evidence"
        return best

    def _citations(self, ids: Iterable[str], access) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for oid in dict.fromkeys(ids):
            try:
                ev = self.store.evidence_for(oid, access)
            except NotFound:
                continue
            source_roles = [
                a for a in ev["attributions"] if a["role"] not in ("importer", "extractor")
            ]
            out[oid] = {
                "quotes": [
                    {
                        "evidence_id": q["evidence_id"],
                        "quote": q["quote"],
                        "role": q["evidence_role"],
                        "start": q["quote_start"],
                        "end": q["quote_end"],
                    }
                    for q in ev["quotes"]
                ],
                "revisions": [
                    {
                        k: r.get(k)
                        for k in (
                            "evidence_id",
                            "source_type",
                            "source_url",
                            "source_version",
                            "source_created_at",
                            "commit_sha",
                            "base_sha",
                            "head_sha",
                            "content_sha256",
                            "import_run_id",
                        )
                    }
                    for r in ev["revisions"]
                ],
                "attribution": [_attr_view(a) for a in source_roles],
                "provenance": [
                    _attr_view(a)
                    for a in ev["attributions"]
                    if a["role"] in ("importer", "extractor")
                ],
                "legacy": ev["legacy"],
                "legacy_projection": ev["legacy_projection"],
                "anchors": ev["anchors"],
            }
        return out


def _attr_view(a: dict) -> dict:
    keep = (
        "role",
        "actor_kind",
        "original_identity",
        "original_provider",
        "original_bot",
        "original_model",
        "original_client",
        "original_environment",
        "normalized_identity",
        "provider_family",
        "model_family",
        "model_version",
        "client_interface",
        "execution_environment",
        "source_session_id",
        "source_run_id",
        "occurred_at",
        "evidence_id",
        "reason",
    )
    view = {k: a.get(k) for k in keep}
    states = a.get("field_states") or {}
    view["unknown_fields"] = sorted(k for k, s in states.items() if s.get("state") == "unknown")
    view["inferred_fields"] = sorted(k for k, s in states.items() if s.get("state") == "inferred")
    return view


_SENTENCES = {
    "confirmed_supersession": "{title} ({id}) was replaced; the edge and a person's recorded "
    "approval are cited.",
    "proposed_supersession": "{title} ({id}) would be replaced, but no person has approved that.",
    "conflict": "This PR's recorded decisions conflict with {title} ({id}); a conflict is "
    "not an authorized override.",
    "compatible": "{title} ({id}) is recorded as compatible with this PR.",
    "no_supported_override_found": "This PR touches code anchored to {title} ({id}); nothing "
    "recorded says it overrides that decision.",
    "insufficient_evidence": "The record is not enough to judge {title} ({id}).",
}


def _quote_ids(cited: dict, oid: str) -> list[str]:
    return [q["evidence_id"] for q in (cited.get(oid) or {}).get("quotes", [])]


def render_narrative(answer: dict) -> list[dict]:
    """Deterministic sentences with the quotes and receipts they rest on. Validated."""
    cited = answer.get("citations") or {}
    receipts = answer.get("receipts") or {}
    out: list[dict] = []
    if answer.get("kind") == "pr_decisions":
        pr = answer.get("pr") or {}
        head = (pr.get("head_sha") or "")[:12]
        out.append(
            {
                "text": f"Verdict for PR #{pr.get('number')} at head {head or 'unknown'}: "
                f"{answer['verdict'].replace('_', ' ')}.",
                "evidence": [],
                "receipts": [],
            }
        )
        for f in answer.get("findings") or []:
            d = f["decision"]
            ev = _quote_ids(cited, d["id"])
            for e in f.get("via") or []:
                other = e["source_id"] if e["target_id"] == d["id"] else e["target_id"]
                ev += [x for x in _quote_ids(cited, other) if x not in ev]
            out.append(
                {
                    "text": _SENTENCES[f["verdict"]].format(title=d["title"], id=d["id"]),
                    "evidence": ev,
                    "receipts": list(f.get("receipts") or []),
                    "verdict": f["verdict"],
                }
            )
    else:
        for oid in (answer.get("graph") or {}).get("history_order", []):
            node = answer["graph"]["nodes"][oid]
            out.append(
                {
                    "text": f"{node['title']} ({oid}), {node['status']}.",
                    "evidence": _quote_ids(cited, oid),
                    "receipts": [],
                }
            )
    for limit in answer.get("limits") or []:
        out.append({"text": f"Limit: {limit}.", "evidence": [], "receipts": []})
    validate_citations(out, cited, receipts)
    return out


def validate_citations(
    sentences: list[dict], citations: dict[str, dict], receipts: dict[str, dict] | None = None
) -> None:
    """Refuse a sentence whose cited proof is missing or does not support its claim.

    Every quote and receipt id must be in the answer. A relationship verdict
    must cite a relationship receipt; a confirmed supersession must also cite
    a human approval receipt.
    """
    receipts = receipts or {}
    known = {q["evidence_id"] for c in citations.values() for q in c.get("quotes", [])}
    for sentence in sentences:
        missing = [e for e in sentence.get("evidence", []) if e not in known]
        missing += [r for r in sentence.get("receipts", []) if r not in receipts]
        if missing:
            raise CitationError(f"narrative cites evidence not in the answer: {missing}")
        kinds = {receipts[r]["kind"] for r in sentence.get("receipts", [])}
        verdict = sentence.get("verdict")
        if verdict in _RELATIONSHIP_VERDICTS and "relationship" not in kinds:
            raise CitationError(f"a {verdict} sentence must cite the relationship it asserts")
        if verdict == "confirmed_supersession" and "approval" not in kinds:
            raise CitationError("a confirmed supersession must cite a recorded human approval")
