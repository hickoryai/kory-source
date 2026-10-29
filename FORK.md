# Kory Source fork baseline

`hickoryai/kory-source` is a deliberately narrow fork of
`miracodeai/mira`. Keep upstream behavior intact unless a change falls inside
the policy below.

## Fork-delta policy

Fork-owned product changes are limited to:

- one Hickory LLM provider under `src/mira/llm/`;
- one package named `registry`;
- dashboard pages;
- configuration;
- database migrations.

Everything else should be contributed upstream or kept out of the fork. Do
not edit upstream source merely to make a sync easier.

## Divergent files

The fork was cut at `567ed7e98b0155448e3449477a58f58b453fcc2e`
(`origin/main`). That commit is also the common ancestor with upstream, so the
fork-authored baseline at the fork point is **0 lines**.

This drift snapshot was refreshed after `git fetch upstream`:

- fork baseline: `567ed7e98b0155448e3449477a58f58b453fcc2e`;
- upstream baseline: `dddb6d333cb9f3337a2ff24549df60afecbb12d2`;
- task head measured: `6bfd34b958a3892c1b04fba90ecb2619c2f50f46`.

At the task head, the only fork-authored files are:

- `FORK.md`;
- `tests/benchmarks/fixtures/hickory/README.md`;
- `tests/benchmarks/fixtures/hickory/baseline-codex-cli.json`;
- `tests/benchmarks/fixtures/hickory/baseline-run-1.json`;
- `tests/benchmarks/fixtures/hickory/baseline-run-2.json`;
- `tests/benchmarks/fixtures/hickory/baseline-run-3.json`;
- `tests/benchmarks/fixtures/hickory/manifest.json`;
- `tests/benchmarks/fixtures/hickory/run_baseline.py`.

There are no fork-owned divergent files under `src/`, `tests/` outside that
fixture directory, or `ui/`. The seven fixture files account for 8,730
insertions under the drift command's path scope; `FORK.md` is outside that
scope.

Measure source drift with:

```bash
git diff --stat upstream/main -- src/ tests/ ui/
```

At the revisions above, the command reports **39 paths, 8,858 insertions, and
2,938 deletions**. That total is the sum of the seven fork fixture files and
these 32 upstream-only paths whose newer upstream changes have not yet been
synchronized into the fork:

- `src/mira/core/engine.py`;
- `src/mira/core/passes.py`;
- `src/mira/dashboard/routers/core.py`;
- `src/mira/dashboard/routers/vulnerabilities.py`;
- `src/mira/index/pg_store.py`;
- `src/mira/index/store.py`;
- `src/mira/llm/base.py`;
- `src/mira/llm/bedrock.py`;
- `src/mira/llm/codex_cli.py`;
- `src/mira/llm/provider.py`;
- `src/mira/llm/provider_profiles.py`;
- `src/mira/llm/providers.json`;
- `src/mira/llm/response_parser.py`;
- `src/mira/llm/responses.py`;
- `src/mira/platforms/fetch.py`;
- `src/mira/security/pr_scan.py`;
- `tests/test_activity.py`;
- `tests/test_activity_search_postgres.py`;
- `tests/test_agentic_tools.py`;
- `tests/test_bedrock_provider.py`;
- `tests/test_codex_cli_provider.py`;
- `tests/test_engine.py`;
- `tests/test_index_store.py`;
- `tests/test_llm_provider.py`;
- `tests/test_org_wide_store.py`;
- `tests/test_pr_scan.py`;
- `tests/test_provider_profiles.py`;
- `tests/test_repo_fetcher.py`;
- `tests/test_responses_provider.py`;
- `tests/test_review_history_postgres.py`;
- `tests/test_vulnerability_routes.py`;
- `tests/test_walkthrough.py`.

The upstream-only portion is 32 paths, 128 insertions, and 2,938 deletions
when comparing `upstream/main` to the fork baseline. These are absent upstream
commits, not fork-authored edits. After an upstream-sync PR lands, any
remaining output is the intentional fork delta and must comply with the policy
above.

## Local Python boot (SQLite)

Docker is not required. From the repository root:

```bash
uv sync --extra serve
npm --prefix ui/mira ci
npm --prefix ui/mira run build
unset DATABASE_URL
export MIRA_INDEX_DIR=/tmp/kory-source-mira-indexes
uv run uvicorn 'mira.platforms.server:create_app' --factory --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765/`. With `DATABASE_URL` unset, Mira uses SQLite.
The first boot creates a local admin password file inside `MIRA_INDEX_DIR`;
keep that generated credential out of Git.

The baseline smoke check used `GET http://127.0.0.1:8765/health` and received
`{"status":"ok"}`, then fetched the dashboard HTML from `/`.

## Monthly upstream sync

1. Fetch both remotes and start a sync branch from the current fork default:

   ```bash
   git fetch --prune origin
   git fetch --prune upstream
   git switch main
   git pull --ff-only origin main
   git switch -c upstream-sync/$(date +%Y-%m)
   git merge --no-edit upstream/main
   ```

2. Resolve conflicts without broadening the fork-delta policy. Re-run the
   drift command and review every remaining fork-owned source line.
3. Rebuild and verify:

   ```bash
   uv sync --all-extras
   npm --prefix ui/mira ci
   npm --prefix ui/mira run build
   uv run pytest
   git diff --stat upstream/main -- src/ tests/ ui/
   ```

4. Push the sync branch and open a draft PR into `main`. Record the upstream
   SHA, test results, drift output, and any intentional divergent files in the
   PR body. Never sync by force-pushing `main`.
5. After review and merge, update the fixture/baseline only when the monthly
   benchmark policy calls for it; do not expose golden comments to reviewer
   prompts or configuration.

## Symbol anchors and the indexer

The fork keeps Mira's index and adds rename-safe anchors on top of it. These
files are the intentional delta. Extractor changes are general (qualified
Python methods, TypeScript methods, interfaces, type aliases, enums, and
`export default function`) and should be offered upstream.

- `src/mira/index/fingerprint.py`
- `src/mira/index/anchor_model.py`
- `src/mira/index/anchor_rows.py`
- `src/mira/index/anchors.py`
- `src/mira/index/symbols.py`
- `src/mira/index/local.py`
- `src/mira/index/extract.py` (qualified names and TypeScript declarations)
- `src/mira/index/store.py` and `src/mira/index/pg_store.py` (fingerprint columns, `symbol_anchors`, `index_runs`)
- `src/mira/index/indexer.py` (resolve anchors at the end of every index run)
- `src/mira/index/relationships.py` (`github_repo_mentions`)
- `src/mira/platforms/fetch.py` (`head_commit`)
- `src/mira/cli.py` (`mira index-repos`)
- `src/mira/dashboard/api.py` and `src/mira/dashboard/routers/anchors.py`
- `ui/mira/src/pages/anchors.tsx` and the anchors API client
- `tests/test_symbol_anchors.py`

`mira index-repos` is the Cloud Run job command. It reads `DATABASE_URL`.
The database login is IAM: the Cloud SQL Auth Proxy runs with
`--auto-iam-authn`, and the URL has no password. The service account is
`kory-source-{env}`. A workstation uses `kory-source-indexer-{env}` and
passes `owner/repo=/checkout` so `git diff -M` can see renames. The workstation indexes tracked files only, so caches and virtualenvs on disk stay out of the graph. The image
does not contain git, so the Cloud Run job resolves anchors by path and by
fingerprint only.

The relink step is a `RelinkProposer`. The default records a pending
proposal from body similarity and does not call a model. A later provider
can implement the same interface. Pending and dangling rows stay in
`symbol_anchors` until a person confirms or retires them.
