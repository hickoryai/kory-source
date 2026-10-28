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

At this baseline, the only fork-owned files are:

- `FORK.md`;
- `tests/benchmarks/fixtures/hickory/` (the frozen Hickory benchmark manifest
  and baseline artifacts).

There are no fork-owned divergent files under `src/`, `tests/` outside that
fixture directory, or `ui/`. The fork-authored source baseline is **0 lines**.

Measure source drift with:

```bash
git diff --stat upstream/main -- src/ tests/ ui/
```

The command shows both upstream commits not yet synchronized into the fork and
fork-owned source changes. After an upstream-sync PR lands, any remaining
output is the fork delta and must comply with the policy above.

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
