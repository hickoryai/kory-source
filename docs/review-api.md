# Review API (review without posting)

The review API runs Kory Source's normal review engine against one exact PR
snapshot and returns the findings to the caller. It never writes to GitHub:
no review, comment, label, description edit, or thread resolution. It also
leaves webhook review history, PR progress, and the last-reviewed SHA alone.

Callers:

- the goose CLI (HAI-11602), run by an engineer on a laptop or a dev box;
- the Kory Code relay (HAI-11603).

Both clients live in their own repositories. This service only provides the
API.

## Flow

1. `POST /api/reviews` with the repository, PR number, and the head SHA the
   caller expects.
2. The service checks the token and the repository allowlist before it reads
   anything from GitHub. Then it reads the PR. If the PR head is not the
   requested SHA, it returns `409 head_sha_mismatch` and starts nothing.
3. The job is stored and the call returns `202` with the job ID and URLs.
4. A worker claims the job and freezes the snapshot: PR metadata, the base
   SHA, and the `base...head` diff, all read by SHA. It checks the head
   again first; if the PR moved, the job fails with `head_sha_mismatch`
   before any model call. Pushes after capture do not change the review.
5. The engine reviews the frozen diff. Every file read, tree listing, and
   history lookup is pinned to the frozen head SHA.
6. The result is written to the database, then the state becomes
   `succeeded`.
7. The caller polls `GET /api/reviews/{id}` until the state is terminal.

## Authentication

Send `Authorization: Bearer <token>`. Tokens are only accepted over HTTPS,
or over plain HTTP to a loopback host (`localhost`, `127.0.0.1`, `::1`).
Behind a TLS-terminating proxy such as Cloud Run, `X-Forwarded-Proto: https`
counts as HTTPS.

A review token reaches `/api/reviews` routes only. It is not a dashboard
login and never grants admin access; using one anywhere else returns `401`.
Logged-in dashboard users can read any job and open its report with their
session cookie. They cannot submit, because a submission must come from a
token with a repository allowlist.

### Token configuration (server)

The deployment stores SHA-256 digests only. Configure them in
`MIRA_REVIEW_API_TOKENS` (JSON) or in a file named by
`MIRA_REVIEW_API_TOKENS_FILE`:

```json
[
  {
    "principal": "zane-laptop",
    "kind": "engineer",
    "sha256": "<hex SHA-256 of the raw token>",
    "repos": ["hickoryai/kory-source"]
  },
  {
    "principal": "kory-code-relay",
    "kind": "relay",
    "sha256": "<hex SHA-256 of the raw token>",
    "repos": ["hickoryai/*"]
  }
]
```

- `kind` is `engineer` or `relay`. Give the relay and each engineer separate
  tokens.
- `repos` lists `owner/repo` names, or `owner/*` for every repository of one
  owner. Matching ignores case.
- A principal sees only its own jobs. Another principal's job returns `404`.
- Entries with a missing field, an unknown kind, a malformed digest, or a
  duplicate principal are skipped and logged.

### Issuing a token

There is no issuance UI or API. The platform administrator the coordinator
designates:

1. generates a random secret, for example
   `python3 -c 'import secrets; print("kst_" + secrets.token_urlsafe(32))'`;
2. stores `sha256(secret)` in the deployment's token config;
3. delivers the raw secret privately to its owner, who keeps it in approved
   secret storage.

To revoke a token, delete its entry. To rotate one, add the new entry, move
the client over, then delete the old entry.

### Client contract (goose CLI)

The goose CLI reads:

- `KORY_SOURCE_URL`, the service base URL (for example
  `https://kory-source.example.com`);
- `KORY_SOURCE_REVIEW_TOKEN`, the raw bearer token.

A laptop shell and a dev-box shell each load these from approved secret
storage on their own. Do not assume one forwards them to the other. Prefer a
separate token per environment. The client implementation belongs to
HAI-11602.

## `POST /api/reviews`

The body must be exactly these three fields. Extra fields and wrong types are
rejected.

```json
{
  "repo": "hickoryai/kory-source",
  "pr_number": 42,
  "head_sha": "0123456789abcdef0123456789abcdef01234567"
}
```

- `repo`: `owner/repo`.
- `pr_number`: a JSON integer greater than zero. Strings and booleans are
  rejected.
- `head_sha`: the full 40-character lowercase hex SHA.

`202 Accepted` returns the job view (below) in state `queued`. Submitting the
same principal, repository, PR, and SHA while that job is still queued or
running returns the existing job with `"deduplicated": true`.

Errors use one shape:

```json
{"error": {"code": "head_sha_mismatch", "message": "...", "current_head_sha": "..."}}
```

| Status | Code | Meaning |
| --- | --- | --- |
| 401 | (plain `error` string) | No token, an unknown token, or a token used outside `/api/reviews`. |
| 403 | (plain `error` string) | Token sent over plain HTTP to a non-loopback host. |
| 403 | `token_required` | A dashboard session tried to submit. |
| 403 | `repo_not_allowed` | The token's allowlist does not include the repository. No GitHub read happened. |
| 422 | `invalid_request` | The body is not exactly `{repo, pr_number, head_sha}`. `fields` lists the problems. |
| 503 | `review_execution_unavailable` | The executor is off on this deployment. No GitHub read happened. |
| 404 | `repo_not_installed` | The GitHub App is not installed on the repository. |
| 404 | `pr_not_found` | The PR does not exist. |
| 409 | `pr_not_open` | The PR is closed or merged. |
| 409 | `head_sha_mismatch` | The PR head is not `head_sha`. `current_head_sha` gives the real head. |
| 429 | `review_queue_full` | Too many jobs are queued. Retry later. |
| 502 | `github_unavailable` | GitHub could not be reached. |

## `GET /api/reviews/{id}`

```json
{
  "id": "rev_5f0c...",
  "state": "succeeded",
  "stage": "done",
  "repo": "hickoryai/kory-source",
  "pr_number": 42,
  "requested_head_sha": "0123...",
  "reviewed_head_sha": "0123...",
  "base_sha": "89ab...",
  "merge_base_sha": "4567...",
  "principal": "zane-laptop",
  "created_at": 1793300000.1,
  "started_at": 1793300001.2,
  "updated_at": 1793300090.3,
  "finished_at": 1793300090.3,
  "deadline_at": 1793303601.2,
  "findings": [
    {
      "id": "f1",
      "path": "src/app.py",
      "line": 12,
      "end_line": null,
      "severity": "blocker",
      "category": "bug",
      "title": "Division by zero",
      "body": "...",
      "confidence": 0.95,
      "suggestion": null,
      "source_pass": "main",
      "registry_ids": []
    }
  ],
  "key_issues": [{"issue": "...", "path": "src/app.py", "line": 12}],
  "summary": "...",
  "walkthrough": "...",
  "coverage": {
    "changed_files": 3,
    "reviewed_files": 1,
    "reviewed_paths": ["src/app.py"],
    "skipped": [
      {"path": "assets/logo.png", "reason": "patch_unavailable"},
      {"path": "dist/app.min.js", "reason": "excluded_by_filter"}
    ],
    "note": ""
  },
  "models": {"requested": ["auto"], "actual": ["grok/grok-4.6"]},
  "model_calls": [
    {"state": "ok", "requested_model": "auto", "actual_model": "grok/grok-4.6",
     "attempts": 1, "priority": "review", "attribution": ""}
  ],
  "error": null,
  "status_url": "https://.../api/reviews/rev_5f0c...",
  "report_url": "https://.../api/reviews/rev_5f0c.../report"
}
```

- `findings` is `null` until the job succeeds, and stays `null` if it fails.
  `[]` means the review succeeded and found nothing.
- `registry_ids` is always `[]` for now. Validated registry citations arrive
  with HAI-11605.
- `coverage.skipped[].reason` is one of `patch_unavailable` (GitHub returned
  no patch: binary or too large), `excluded_by_filter` (matched the review
  file filter), or `size_or_priority_cap` (cut by the diff size budget).
- `models.requested` and `models.actual` are the distinct router model IDs
  this job asked for and got. `model_calls` lists each logical completion
  with its outcome and attempt count. They cover this job only, even when
  other jobs run at the same time.

### States and stages

| State | Terminal | Meaning |
| --- | --- | --- |
| `queued` | no | Stored, waiting for a worker. |
| `running` | no | A worker holds the job. `stage` is `capturing`, `reviewing`, or `persisting`. |
| `succeeded` | yes | Result stored. `findings` is a list. |
| `failed` | yes | `error.code` says why. `findings` is `null`. |
| `interrupted` | yes | The worker stopped before finishing. The job is never re-run, because its model calls may already have been paid for. Submit again. |

Failure codes (`error.code`):

| Code | Meaning |
| --- | --- |
| `head_sha_mismatch` | The PR moved between submission and capture. Nothing was reviewed. |
| `pr_not_found`, `repo_not_installed`, `github_unavailable` | GitHub lookup failed during capture. |
| `diff_unavailable` | GitHub could not compare the base and head commits. |
| `diff_incomplete` | GitHub returned fewer files than the PR changes (over the 300-file compare limit). The service will not report partial coverage as a full review. |
| `seat_exhausted` | No router-acp seat was free after the configured retries. Retry later. |
| `workstation_outage` | The workstation behind the relay is unreachable. Retry after it recovers. |
| `model_timeout` | A seat did not answer in time. |
| `model_mismatch` | A pinned model was not the model that ran. |
| `model_call_failed` | Any other failed or unparseable model reply. |
| `seat_provider_unconfigured` | The deployment is not configured for router-acp. There is no fallback. |
| `deadline_exceeded` | The job ran past its deadline. |
| `write_blocked` | The engine tried to write to GitHub and was refused. This is a bug; report it. |
| `review_failed` | The engine failed for another reason, such as an unparseable diff. |
| `internal_error` | The worker crashed. |
| `executor_interrupted` | The worker's lease lapsed (crash or restart). |
| `executor_shutdown` | The server stopped while the job ran. |

A failed model call fails the whole job, even in a pass the webhook review
would skip on error (walkthrough, security pass, summary). A partial review
is never reported as a clean one.

### Polling

Poll `status_url` every 5–10 seconds until the state is terminal. A review
usually takes one to several minutes. Stop after the job's `deadline_at`.

## `GET /api/reviews/{id}/report`

A read-only HTML page for one job: state, SHAs, models, findings, coverage.
Open it with the same bearer token, or in a browser with a dashboard login.

## Models

Every model role (review, indexing, security) uses the router-acp seat
provider. The job fails with `seat_provider_unconfigured` if
`llm.provider` is not `router-acp`; there is no cloud-key or local fallback.
Ensemble sampling, tool calling, and cross-PR overlap detection are off for
API reviews. Seat retries follow `llm.router_acp_max_retries`.

## Execution

The executor runs inside the web server process and is **off by default**.
Until it is on, `POST /api/reviews` returns `503
review_execution_unavailable`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MIRA_REVIEW_API_ENABLED` | unset | `1` starts the executor. |
| `MIRA_REVIEW_API_MAX_RUNNING` | `2` | Jobs running at once, across all instances sharing the database. |
| `MIRA_REVIEW_API_MAX_QUEUED` | `20` | Queued jobs before `429 review_queue_full`. |
| `MIRA_REVIEW_API_DEADLINE_SECONDS` | `3600` | Per-job deadline (minimum 60). |
| `MIRA_REVIEW_API_LEASE_SECONDS` | `120` | Lease length; the worker renews it every third of that. |
| `MIRA_REVIEW_API_POLL_SECONDS` | `2` | How often a worker looks for queued jobs. |
| `MIRA_REVIEW_API_PUBLIC_URL` | request host | Base URL for `status_url` and `report_url`. |
| `MIRA_REVIEW_API_TOKENS` / `MIRA_REVIEW_API_TOKENS_FILE` | unset | Token digests (above). |

A job is claimed with one conditional update that also enforces the running
cap. A worker renews its lease while the job runs. Running jobs whose lease
lapsed are marked `interrupted` at startup and on every poll. On shutdown a
worker cancels its jobs and marks them `interrupted`.

Jobs live in the `review_api_jobs` table of the application database
(SQLite or PostgreSQL). The table is created at startup if missing; no
existing table changes.

### Cloud Run

Jobs run after the `202` response returns, so the instance needs CPU between
requests and must not scale to zero mid-review.

- The `kory_source` Terraform module sets `cpu_idle = (min_instance_count == 0)`.
  CPU stays allocated whenever at least one instance is kept warm.
- **prod** sets `min_instance_count = 1`, so it already meets this. Set
  `MIRA_REVIEW_API_ENABLED=1` there when ready.
- **dev** sets `min_instance_count = 0`. Raise it to at least `1` before
  enabling the executor in dev.

Set the token config and `MIRA_REVIEW_API_PUBLIC_URL` from the service's
secret and environment configuration.

## Registry rules hook

`mira.review_api.runner.custom_rules_providers` is a list of async callables.
Each takes the frozen snapshot and returns `{"title", "content"}` rules. The
rules are added to the custom rules the engine already loads for the
repository. HAI-11605 registers the registry's provider here.

## Example

```bash
curl -sS -X POST "$KORY_SOURCE_URL/api/reviews" \
  -H "Authorization: Bearer $KORY_SOURCE_REVIEW_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"repo\":\"hickoryai/kory-source\",\"pr_number\":42,\"head_sha\":\"$(git rev-parse HEAD)\"}"

curl -sS "$KORY_SOURCE_URL/api/reviews/rev_5f0c..." \
  -H "Authorization: Bearer $KORY_SOURCE_REVIEW_TOKEN"
```
