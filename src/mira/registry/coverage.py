"""Operator-only reconciliation of live source IDs, evidence and import runs.

``report`` is read-only. Without a local Git checkout, repository counts come
from evidence and runs and are explicitly unverified. Supply --repo, --checkout
and optionally --commit to reconcile Git against a frozen commit as well.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from typing import Any

from mira.registry import linear
from mira.registry.access import AccessContext, AccessDenied, require_import
from mira.registry.imports import ImportRuns, read_cursors
from mira.registry.repository_history import GitReader, language_of


def inventory_ids(ids) -> dict:
    values = sorted(set(ids))
    return {
        "count": len(values),
        "ids": values,
        "sha256": hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest(),
    }


def reconcile(expected, stored, *, stale=()) -> dict:
    expected, stored = set(expected), set(stored)
    return {
        "source": inventory_ids(expected),
        "stored": inventory_ids(stored),
        "missing": inventory_ids(expected - stored),
        "extra": inventory_ids(stored - expected),
        "stale": inventory_ids(stale),
    }


class CoverageReport:
    def __init__(self, registry: Any, *, access: AccessContext):
        self.registry = registry
        self.access = access
        self.policy = require_import(access)
        self.runs = ImportRuns(registry).coverage(access)

    def _revisions(self, system, kind=None, repo=None):
        query = (
            "SELECT source_id, source_type, source_version, source_updated_at, "
            "import_run_id, repository, evidence_id FROM registry_source_revisions "
            "WHERE corpus_id = ? AND source_system = ?"
        )
        args = [self.policy.corpus_id, system]
        if kind:
            query += " AND source_type = ?"
            args.append(kind)
        if repo:
            query += " AND repository = ?"
            args.append(repo)
        return list(self.registry.db.execute(query, tuple(args)))

    def _runs(self, system):
        return [r for r in self.runs if r["source_system"] == system]

    def _provenance(self, rows, runs):
        known = {r["run_id"] for r in runs}
        unlinked = [row[6] for row in rows if row[4] not in known]
        source_runs = defaultdict(set)
        for row in rows:
            if row[4] in known:
                source_runs[row[0]].add(row[4])
        return {
            "run_ids": sorted(known),
            "evidence_without_run": inventory_ids(unlinked),
            "source_run_ids": {sid: sorted(ids) for sid, ids in sorted(source_runs.items())},
        }

    def linear(self, client: linear.LinearClient) -> dict:
        require_import(self.access)
        teams = self.policy.team_keys
        specs = {
            "issues": (linear.ISSUES_QUERY, linear._team_filter(teams), "linear_issue_description"),
            "comments": (linear.COMMENTS_QUERY, linear._on_issue(teams), "linear_comment"),
            "documents": (linear.DOCUMENTS_QUERY, {}, "linear_document"),
        }
        runs = self._runs("linear")
        result = {}
        by_team = {team: {} for team in teams}
        by_team["workspace_documents"] = {}
        partial_document_teams = set()
        for stream, (query, filt, kind) in specs.items():
            nodes = {}
            groups = defaultdict(set)
            for key in teams:
                groups[key] = set()
            if stream == "documents":
                groups["workspace_documents"] = set()
            for page, _ in client.pages(query, stream, filt):
                for node in page:
                    team = (
                        node.get("team")
                        if stream == "issues"
                        else (node.get("issue") or {}).get("team")
                    )
                    if stream != "documents" and (
                        not linear._admitted(self.policy, team)
                        or (team or {}).get("key") not in teams
                    ):
                        continue
                    nodes[node["id"]] = node
                    if stream == "documents":
                        # Documents are workspace-wide; do not invent a team for
                        # initiative, cycle, release or unattached documents.
                        groups["workspace_documents"].add(node["id"])
                        if (
                            ((node.get("project") or {}).get("teams") or {}).get("pageInfo") or {}
                        ).get("hasNextPage"):
                            partial_document_teams.add(node["id"])
                        document_teams = [team] if team else []
                        document_teams += ((node.get("project") or {}).get("teams") or {}).get(
                            "nodes"
                        ) or []
                        for document_team in document_teams:
                            if (
                                linear._admitted(self.policy, document_team)
                                and document_team.get("key") in teams
                            ):
                                groups[document_team["key"]].add(node["id"])
                    else:
                        groups[team["key"]].add(node["id"])
            rows = self._revisions("linear", kind)
            latest = defaultdict(int)
            for row in rows:
                latest[row[0]] = max(latest[row[0]], int(row[2] or 0))
            stale = {
                sid
                for sid, node in nodes.items()
                if sid in latest and linear.version(node.get("updatedAt")) > latest[sid]
            }
            result[stream] = reconcile(nodes, latest, stale=stale)
            result[stream].update(
                self._provenance(rows, [r for r in runs if r["stream"] == stream])
            )
            for team, ids in groups.items():
                by_team[team][stream] = {
                    "source": len(ids),
                    "stored": len(ids & latest.keys()),
                    "missing": len(ids - latest.keys()),
                    "stale": len(ids & stale),
                }
        return {
            "streams": result,
            "teams": by_team,
            "partial_document_team_labels": inventory_ids(partial_document_teams),
            "cursors": read_cursors(self.registry, "linear"),
            "runs": runs,
        }

    def git(self, *, repo=None, checkout=None, commit=None) -> dict:
        require_import(self.access)
        if repo and not self.policy.admits_repository(repo):
            raise AccessDenied("the repository is not in the approved corpus manifest")
        if checkout and not repo:
            raise ValueError("--checkout requires --repo")
        result = {}
        runs = self._runs("git")
        for _rid, name in self.policy.repositories:
            if repo and repo.lower() != name.lower() and repo != _rid:
                continue
            rows = self._revisions("git", repo=name)
            repo_runs = [r for r in runs if r["stream"] == f"repository:{name}"]
            row = {
                "status": "unverified_no_checkout",
                "evidence_revisions": len(rows),
                "stored_source_ids": inventory_ids(r[0] for r in rows),
                "runs": repo_runs,
                **self._provenance(rows, repo_runs),
            }
            if checkout:
                reader = GitReader(checkout)
                frozen = reader.rev_parse(commit or "HEAD")
                files = reader.files(frozen)
                units = (
                    {"log:"}
                    | {f"md:{p}" for p in files if p.lower().endswith((".md", ".mdx", ".markdown"))}
                    | {f"code:{p}" for p in files if language_of(p)}
                )
                matching = [
                    r
                    for r in repo_runs
                    if r["status"] == "complete"
                    and json.loads(r["scope_manifest"]).get("commit") == frozen
                    and json.loads(r["scope_manifest"]).get("max_commits") == 0
                ]
                matching.sort(key=lambda r: (r["finished_at"], r["run_id"]), reverse=True)
                complete = matching[0] if matching else None
                done = (complete["checkpoint"] or {}).get("done", []) if complete else []
                commits = {f"{name}@{c['sha']}" for c in reader.commits(frozen, max_count=0)}
                row.update(
                    status="reconciled" if complete else "no_complete_run",
                    commit=frozen,
                    complete_run_id=complete["run_id"] if complete else None,
                    files=inventory_ids(files),
                    units=reconcile(units, done),
                    commits=reconcile(
                        commits, [r[0] for r in rows if r[1] == "git_commit_message"]
                    ),
                )
            result[name] = row
        return {
            "repositories": result,
            "cursors": {
                stream: cursor
                for stream, cursor in read_cursors(self.registry, "git").items()
                if stream in {f"repository:{name}" for _, name in self.policy.repositories}
            },
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mira.registry.coverage")
    parser.add_argument("command", choices=("report",))
    parser.add_argument("--source", choices=("linear", "git"))
    parser.add_argument("--repo")
    parser.add_argument("--checkout")
    parser.add_argument("--commit")
    args = parser.parse_args(argv)
    if (args.checkout and not args.repo) or (args.commit and not args.checkout):
        parser.error("--checkout requires --repo; --commit requires --checkout")
    if args.source == "linear" and (args.repo or args.checkout or args.commit):
        parser.error("Git arguments require --source git or no --source")
    from mira.registry import get_registry
    from mira.registry.access import operator_context

    access = operator_context()
    require_import(access)
    registry = get_registry()
    assert registry is not None
    report = CoverageReport(registry, access=access)
    out = {"corpus_id": report.policy.corpus_id, "policy_version": report.policy.policy_version}
    if args.source != "git":
        out["linear"] = report.linear(linear.LinearClient.from_env())
    if args.source != "linear":
        out["git"] = report.git(repo=args.repo, checkout=args.checkout, commit=args.commit)
    json.dump(out, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
