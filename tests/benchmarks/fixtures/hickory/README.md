# Hickory benchmark fixtures

`manifest.json` freezes 30 merged `hickoryai/hickory-ai` pull requests from
the 90-day window ending 2026-10-28. Every entry contains the repository, PR
URL, base and head SHAs, and the Greptile comment export. The ten newest PRs
are `held-out`; the preceding twenty are `development`.

Greptile inline comments are golden judge truth. GitHub thread resolution and
outdated state are retained as export metadata, but do not rewrite that truth.
Summary comments are provenance only and are not findings.

Golden text must never enter reviewer context. `run_baseline.py` constructs a
minimal runner fixture, passes only the PR URL to `ReviewEngine`, and makes
findings available only to Mira's judge phase.

Run the three-trial baseline with the box's existing Codex login and GitHub CLI
session; do not write either credential into a file:

```bash
GITHUB_TOKEN="$(gh auth token)" CODEX_HOME=/home/devagent/.codex \
  uv run python tests/benchmarks/fixtures/hickory/run_baseline.py
```

The command creates three raw run artifacts and `baseline-codex-cli.json` in
this directory. The summary records population σ for F1, precision, recall,
all per-PR latency observations, and each PR across the three runs.

## Router-acp harness

`run_router_acp.py` must run on the box that owns the relay, with
`MIRA_ROUTER_ACP_URL` (or its loopback relay entry) pointing at that box's
loopback relay. It constructs `RouterAcpProvider` directly with that URL.
Kory Source itself stores no relay URL for reviews: the review API hands
each model call to the client that submitted the job (HAI-12330), and
`create_llm("router-acp")` fails with `seat_provider_unconfigured`.
