# Decision history import (HAI-12825)

Read-only imports that turn source history into immutable evidence and
proposed decisions. They never write to GitHub, Linear or the repository, and
they never approve anything.

All import and coverage commands need an enabled corpus policy (see
[decision-provenance.md](decision-provenance.md)) and
`REGISTRY_IMPORT_OPERATOR` naming an approved `operator`. Scope is checked
before the first source call: a repository or requested issue team outside the
manifest is refused. Linear documents are workspace-wide: the deployment must
approve workspace document access before enabling the sync credential. Shared-environment imports need their own
separate approval; nothing here runs automatically.

| Command | Reads |
|---|---|
| `python -m mira.registry.linear inventory` | Team list (to report unapproved teams), counts for manifest teams. |
| `python -m mira.registry.linear import [--team KEY] [--apply]` | Issues and issue comments of manifest teams, plus every workspace document (including initiative, cycle, release and mixed-team project documents); immutable revisions and proposed decisions with exact cited quotes. `--team` narrows issues and issue comments only. |
| `python -m mira.registry.github_history import [--repo owner/name] [--max-pulls N] [--apply]` | PR descriptions, conversation comments, reviews, inline review comments. REST GET only (`GITHUB_HISTORY_TOKEN`, read scope). |
| `python -m mira.registry.repository_history import --repo owner/name --checkout PATH [--commit SHA] [--max-commits N] [--apply]` | Markdown, SQL, data and template revisions along history; code comments/docstrings at the frozen commit; commit messages. Allowlisted read-only git commands. |

Without `--apply` each command is a dry run that writes nothing.

## Guarantees

- **Idempotent.** Evidence ids are content keys; a replay adds no rows. The
  same Markdown passage in a later revision adds supporting evidence to the
  existing decision, not a duplicate.
- **Resumable.** Each page (a Linear page, a PR, a file) commits its evidence
  and the advanced checkpoint in one transaction. A rate limit or crash leaves
  an `incomplete` run that the next invocation resumes.
- **Honest coverage.** `registry_import_runs.coverage` counts discovered,
  fetched, stored and duplicate evidence, extracted candidates, linked and
  unlinked anchors, attributed and unknown fields, excluded scope and
  unavailable items. Totals the source cannot report are `null`. A history
  walk bounded by `--max-commits` is never marked complete.
- **No inferred deletion.** A tombstone is written only after a complete
  enumeration or an authoritative deletion; a timeout or 403 is not one.
- **Inert content.** Imported text is evidence. Instructions inside it are
  never executed and change no status.

## Extraction

Repository history uses a two-stage extractor. `decision_passages.nominate`
only selects candidates: Markdown paragraphs/list items with decision/rationale
cues or ADR/decision/constraint/rationale headings, rationale-keyed data values,
and comments/text with broad choice,
constraint or rationale cues (including because, instead of, we use/keep/chose,
do not, never, always, must, trade-off, rationale, why, decided and ratified).
Adjacent line comments retain their choice/rationale context. Changelog, license,
release-note and generated boilerplate is excluded. Nomination never creates a
decision by itself.

`decision_llm.DecisionClassifier` deduplicates nominated passages across files
and revisions and packs them into batches of at most 16 passages or 12,000
source characters (`REGISTRY_EXTRACTION_BATCH_SIZE` and
`REGISTRY_EXTRACTION_BATCH_CHARS` configure these limits). Bounded lookahead combines small files into the same call
while retaining each file's provenance and atomic page checkpoint. Up to two
calls run concurrently by default; `REGISTRY_EXTRACTION_CONCURRENCY` configures
the limit. Every successful batch commits its cache before a failed sibling is
reported, so retries spend seats only on unfinished work. Call timeouts split
the pack in half under the same concurrency cap, down to a single passage.
A singleton that still times out is cached as skipped with its reason and
recorded in coverage/checkpoint details; it is never counted as a negative
classification or a completed decision. Budget expiry instead leaves work
resumable. Extractor version 3
uses short local response IDs while retaining SHA-256 cache keys.
It uses `mira.llm.router_acp.RouterAcpProvider`. Tracing follows that provider.
`MIRA_ROUTER_ACP_ALLOW_UNTRACED=1` runs classification with no collector.
Without that opt-out, a missing `LLM_OTLP_ENDPOINT` fails closed with
`router_acp_tracing_unavailable`. With the endpoint set and the opt-out unset,
the call stays traced. No provider API keys are used.
`REGISTRY_EXTRACTION_MODEL` selects a pinned router candidate; otherwise the
runtime indexing model, then runtime model is used. Unpinned/auto models are
refused so model-specific cache entries have stable meaning.

Each structured result has `is_decision`, `statement`, `quote`, `rationale` and
`confidence`. A positive result requires nonempty statement/rationale and a
verbatim supporting substring; unsupported quotes are rejected and the
rejection is cached. Malformed responses fail the unit without advancing its
checkpoint. Model statements populate decision bodies, and exact supporting
quotes/offsets remain in immutable evidence. Model identity is recorded on the
row and in a separate extractor attribution; source author/model attribution
stays separate. The model's numeric confidence is retained in the classification
cache and evidence reason, independently of the registry's computed confidence.

The additive `002_decision_classifications.sql` cache is scoped by corpus and
keyed by passage SHA-256, extractor version and pinned model. Positive, negative
and invalid-quote rejection results persist before evidence materialization,
so retrying a failed evidence transaction does not repeat model calls. Unchanged
passages reuse cache entries across files and imports. Exact completed scopes
reuse their completed run, and replays do not add classification, decision,
evidence or attribution rows. Changing content, extractor version or model
requires new classification. An older binary that does not know migration 002
refuses that database, as required by the migration ledger contract.

Dry runs report candidate passages, unique uncached passages, estimated logical
LLM calls, model and whether estimation completed. They do not construct a
provider, call a seat, write cache entries or materialize decisions. Estimates
exclude transport retries and are explicitly incomplete if source reads/parsing
fail. Tests use fake router transport/tracing, with no live calls in CI.

Linear and GitHub source extraction retains the existing deterministic
`imports.extract_decisions` path; the reusable classifier is available to other
source plugins. All extracted interpretations remain `proposed` with exact
cited evidence; source wording never grants registry approval.

## Anchors

- Inline review comments anchor to their path, line and commit.
- Code comments anchor to the innermost enclosing symbol from
  `mira.index.extract` at the frozen commit, else path-only.
- Markdown passages anchor to file and heading.
- `anchor_resolution.resolve` maps a historical anchor to a later commit:
  index anchor id, exact path and symbol, git rename, then `pending` (symbol
  gone, needs a person) or `dangling`. Unsupported languages are `unsupported`.

## Known gaps (reported, not imported)

Linear project objects; comments not on an issue; relations beyond the
first 50; edit history before the first observation (GitHub and Linear expose
only the current body); languages without a comment syntax here; git history
beyond `--max-commits`; cross-repository references outside the manifest.

## Legacy rows

`python -m mira.registry.provenance classify-legacy [--apply]` (operator
context) reports how many
mirrored Linear rows match a manifest team by ticket prefix; `apply=True`
classifies them, their comments, anchors and relationships. Unmatched rows
stay unclassified and denied. Run it only against an approved target and keep
the reconciliation report.

## Full repository history and coverage

`python -m mira.registry.checkout --repo owner/name --dest /tmp/checkout`
checks the runtime corpus manifest before GitHub authentication. It uses
`MIRA_GITHUB_APP_ID` and `MIRA_GITHUB_PRIVATE_KEY` to mint an installation token
for that repository with `contents: read`, then creates a full bare clone.
It does not use a webhook secret or persist the token in Git configuration.

Pass the returned frozen SHA to
`python -m mira.registry.repository_history import --repo owner/name --commit SHA --max-commits 0 --checkout /tmp/checkout`.
Add `--apply` only for an approved target and import action. Zero means unbounded;
every positive limit leaves the run incomplete. A completed unbounded walk
records its frozen SHA and commit timestamp in `registry_sync_state`.

`python -m mira.registry.coverage report` reads every current issue, issue-comment
and workspace document ID and reconciles against evidence in the caller's corpus.
It reports missing, stale and extra IDs, sorted-list SHA-256 digests, per-team
counts, source cursors and import runs. Documents also have a workspace total;
team counts may overlap for shared projects. Truncated project team labels are
reported separately and do not limit workspace document ingestion. Stale means
the source `updatedAt` is newer than the latest stored revision. Extra IDs are
reported for investigation, never treated as proof of deletion.

Use `--source linear` or `--source git` to narrow the report. For live Git
reconciliation add `--repo owner/name --checkout /tmp/checkout --commit SHA`.
This compares frozen files with a matching unbounded complete run's checkpoints,
and every reachable commit ID with stored commit-message evidence. Without a
checkout each repository is explicitly `unverified_no_checkout`, with stored
counts and runs only. Coverage reports write nothing and never perform imports.
A report alone is not proof that an import is complete or that publication is safe.

The first Linear run after upgrading from importer version 2 revisits prior
observations so unchanged descriptions and comments also get decision extraction;
subsequent runs resume the per-stream incremental cursors. Replays preserve
proposed status and never infer human approval.

Deployments may exclude GitHub PRs and reviews as decision sources; repository
history is independent of that importer. Image publication requires the separate
artifact and package-history audits and recorded maintainer approval in AGENTS.md.


## Extra repository decisions and source plugins

Repository importer version 4 includes SQL `--` and `/* */` comments, JSON/YAML/CSV/TOML
reason values, Jinja `{# #}`, HTML/SVG `<!-- -->`, CSS `/* */`, and plain `.txt`
passages. All these file types walk their revisions, including migration seed
and registry data in arbitrary directories. YAML retains `#` comments as well
as value scanning. Other supported code languages retain frozen-head scanning.

Reason/rationale/why/decision/_comment keys nominate rationale values regardless
of explicit decision wording. Other values need broad rationale/choice cues.
The model distinguishes actual adopted decisions from questions, descriptions,
hypothetical options and undecided statements. Imported text is never executed.
A data quote is the exact source slice (including lexical quoting or multiline
syntax), with its escaped key path in the evidence `section` field; CSV paths
use `/rows/<zero-based-data-row>/<column>`. Identical values at distinct key paths
are distinct decisions. SQL strings and dollar-quoted values are not comments.

The same inventory powers dry runs, import and coverage reconciliation. Dry runs
report markdown/code/sql/data/template file counts, skipped reasons, candidate
passages and estimated uncached LLM calls, without classifying or storing decisions. Dependency/generated directories, lockfiles,
minified files, generated-file markers, non-text encodings and NUL-containing
files are skipped explicitly. Malformed structured data and failed Git reads
leave an incomplete run with recorded errors. Each recorded error is logged at
WARNING as it is stored. Apply and dry-run stdout include `error_count` and the
first 20 errors, each truncated to 300 characters; the run row keeps the full list. Positive commit limits remain
incomplete; zero walks all reachable file revisions for the selected files.
The inventory selects files present at the frozen commit; files deleted before
that snapshot are outside this walk. Historical revisions of selected files
can retain decisions overwritten or removed in later versions.

### Incremental repository imports

After an unbounded, unfiltered backfill completes, use `--incremental` to
read changes after its saved Git commit. The cursor commit itself is excluded.
Omit `--max-commits` and `--path`. Add `--apply` to store decisions and advance
the cursor. The factory accepts `incremental=True`.

The importer reads commit messages within the range. It reads changed Markdown,
SQL, data and template revisions inside that range. It reads code comments and
docstrings at the range's final commit for paths the range changed. For a file
decision, immutable evidence stores the exact excerpt, its original offset and
the URL to its lines. Files without decisions create no source revisions.

An unfinished apply resumes its saved range before processing a later commit.
Use the same extractor configuration to resume. A time budget, classification
group failure, or evidence write failure leaves the range open. The importer
records file parse failures in `checkpoint.failed` and in the error summary.
The cursor advances once every unit is stored or has a recorded parse failure.

| Status | Meaning |
| --- | --- |
| `waiting_for_backfill` | No cursor exists. The importer performs no Git reads or writes. |
| `up_to_date` | The requested commit equals the cursor. |
| `cursor_not_ancestor` | The requested history does not contain the cursor. Check the checkout and commit before retrying. |
| `pending_range_ahead` | An unfinished range ends outside the requested history. Retry with its final commit or a descendant. |
| `cursor_changed` | Another run advanced the cursor. Retry from the new cursor. |

`source_importers.SourceImporter` defines `run(*, apply=False)`. A
`SourceImporterRegistry` registers factories, and `default_source_importers()`
registers lazy `linear` and `repository-history` adapters. A deployment can
register another ticket source without editing core code. Configuration and
credentials are supplied at runtime; each adapter retains the existing corpus
policy checks. No new ticket connector or scheduled job is installed.

```python
from mira.registry.source_importers import SourceSelection, default_source_importers

sources = default_source_importers()
sources.configure_sources("example/widgets", [
    SourceSelection("repository-history", {
        "access": operator_access, "checkout": checkout_path, "max_commits": 0,
    }),
    SourceSelection("linear", {"access": operator_access, "client": linear_client}),
])
report = sources.run_all("example/widgets", registry)  # dry runs only
# A caller can sources.register("other-tickets", custom_factory) at runtime.
```

Imported proposed decision rows carry the source system, stable source URL,
source author identity in `author` and `decided_by`, and source occurrence time in
`decided_at`. Repository passages use Git blame for their line author/date and
Git log for historical revisions; URLs name the frozen commit, path and exact
line range. A stated `decided/ratified/approved YYYY-MM-DD` date takes precedence
at UTC midnight. Otherwise time means the source's recorded occurrence, not a
claim that the original decision was made at exactly that instant. Git identity
is an inferred decision author, with that uncertainty retained in attribution;
agent markers remain separate from source accounts. Linear retains issue,
comment or document author/time and source URL. Unknown identities remain
unknown instead of being replaced with the importer.

Replaying repository v4 or Linear v4 revisits existing evidence and backfills
missing provenance without approving decisions. Existing lifecycle approvers
and statuses are preserved; after a registry lifecycle action `decided_by` and
`decided_at` retain their lifecycle meaning, while source attribution and
immutable evidence retain the original author/time. Upgraded importer versions
use fresh scopes so old checkpoints cannot suppress the new extraction.


## Bounded planning and resumable execution

Use repeatable `--path <file-or-directory>` filters and `--max-commits 2` for a
bounded preview before estimating a large backfill. Git receives the history
limit before walking, rather than reading all history and slicing afterward.
`--max-seconds 900` optionally limits an execution attempt. Completed file pages
remain checkpointed; completed classification batches remain cached even when
a file page has not materialized. Resuming the same scope skips completed pages
and classifications. Path-filtered and positive-commit-limit runs remain
incomplete and never advance the full repository cursor.

Each apply prints `per_source` counts for the current execution attempt:
`nominated`, `grouped`, `classified_ok`, `classified_failed`, `classified_skipped`,
`stored` (accepted passages written with evidence, including replays), and
`stored_evidence` (new immutable source revisions). Classification counts include
cached outcomes. A failed group can contain completed cached batches, but none
of that group's file pages are stored until classification succeeds on resume.
Failed groups retain their nomination estimates in coverage. `slice_coverage`
reports this execution's increments, including `llm_calls`; `coverage` remains
cumulative across resumes. `errors_by_type` summarizes the full error list,
independently of the bounded error preview. These counters do not change the
import scope, extractor version or checkpoint identity.

An apply requires an explicitly configured router URL and authentication before
inventory or run creation. Missing configuration fails once and makes no model
call; imports never use the provider's loopback development fallback. Dry-run
planning remains independent of router configuration.

Classification retries workstation transport outages six times with 5, 10, 20,
30, 30 and 30 seconds of backoff. Every attempt rechecks the execution deadline
and reduces its call timeout to the remaining budget. No backoff starts when it
would exhaust that budget. Authentication, model mismatch, malformed output and
intrinsic call timeouts do not use this outage retry loop.

JSON values use JSON semantics, never YAML, with comments and trailing commas
accepted for JSONC configuration files. Quotes retain original lexical offsets.
A historical path absent at a deletion or rename commit is an explicit
`skipped_history_absent_path`; git object and command failures remain errors.

Code at the frozen commit shares one commit metadata lookup. Provenance lookup
runs one Git blame command for all extracted decision lines in each file revision,
with cached origins and commit metadata, instead of one command per decision.
Nomination uses indexed newline positions, position-based SQL/TOML scanning and
the safe C YAML loader when available, with a portable safe-loader fallback.

The runtime job must set `MIRA_TRACING_MODULE=mira.tracing.otlp` together with its
OTLP endpoint when classification is traced. `MIRA_ROUTER_ACP_ALLOW_UNTRACED=1`
skips that requirement. A direct router transport also needs the relay settings.
Live runtime estimates must distinguish
cold-cache seat work from source/Git processing time: the initial backfill can
require many bounded executions, whereas unchanged passages require no further
model calls. A bounded sample is not a complete history inventory or permission
to run a shared-environment import.


### Classification through the existing seat worker

Set `REGISTRY_EXTRACTION_TRANSPORT=worker` on the import job to use its existing
registry database connection instead of a direct relay credential. The default
remains the configured direct router transport. The worker mode does not need a
relay URL, a new secret, or a new IAM grant on the job. Classification still
honors the tracing opt-out above. Without the opt-out, the tracer must be
available.

### Standalone import, no outside worker

When the service has the bundled router, a mounted router configuration with
at least one account, and `KORY_SOURCE_AUTO_REVIEW=1`, the in-container worker
claims history-import batches as well as review calls. It answers them through
the local seat. A standalone deploy then needs no outside relay and no
`MIRA_ROUTER_ACP_URL`. Run the import with `REGISTRY_EXTRACTION_TRANSPORT=worker`.

Set `LLM_OTLP_ENDPOINT` and `MIRA_TRACING_MODULE=mira.tracing.otlp` to trace
those calls. Or set `MIRA_ROUTER_ACP_ALLOW_UNTRACED=1` to classify with no
collector. Leave the opt-out unset and omit the endpoint, and classification
fails closed with `router_acp_tracing_unavailable`. Hickory deploys keep the
endpoint set and the opt-out unset, so Hickory imports stay traced.

An outside worker and the in-container worker share the queue. The first claim
owns the batch. A second answer for that attempt is refused. A timeout requeues
the batch once. `router_acp_model_mismatch` stays retryable when model fallback
is on. After repeated outages the in-container worker stops claiming for a
minute.

Migration `003_import_work.sql` adds the additive classification queue. Each
batch binds a corpus, live policy hash, admitted repository import run, operator,
model, and deadline. Polls and answers recheck admission. The worker bearer still
has access only to the existing work routes, not corpus read or enqueue APIs.
Removing admission prevents both fetching prompts and submitting results.

Updated workers advertise `import_capacity=1` only while they have no import
batch in flight. Old workers receive review work only. The service always
returns pending review work first, including work excluded because it is already
in flight, before considering an import. Imports have one global in-flight
lease and do not consume the worker's configured review concurrency. An import
already forwarded to the relay is not preempted when a review arrives.

A timed-out lease or worker-reported timeout requeues the batch once. The second
timeout records a visible terminal failure. Attempt-specific work ids reject
late answers from the first attempt. The import deadline bounds queue waiting
and retries. If no worker claims the batch, the import reports
`no_worker_before_deadline`. Cancelling a waiter expires its pending work.

The seat worker runs from its own installed checkout. Deploying the service image
does not update that process. Update the worker checkout and restart only its
worker service through the deployment's supported operator mechanism before the
first apply. A relay restart is unnecessary. Keep the apply held when no supported
worker update mechanism is available.

## Slack decision import

An import operator can run `python -m mira.registry.slack import` to sample one
history page per approved channel. A dry run writes no registry data and calls no
model. Add `--apply` to store messages and propose decisions. An empty or missing
`slack_channels` list disables Slack access. The corpus policy lists Slack channel
IDs, for example `"slack_channels": ["CEXAMPLE"]`.

Use a workspace Slack app and set its bot token as `SLACK_HISTORY_TOKEN`. Grant
`channels:history`, `groups:history`, `channels:read`, `groups:read`, and
`users:read`. Add the bot to each approved channel through Slack. The importer
records a channel access error if the bot lacks access.

Set `SLACK_SIGNING_SECRET` on the service. Use `/api/registry/slack/events` on
that service as the Slack event request URL. Subscribe to `message.channels` and
`message.groups`. The service verifies each request and queues only events for
approved channels. Apply drains those events for the token's workspace.

```text
Slack channel history → Import job → Registry evidence → Proposed decisions
Slack message events → Signed inbox ↗
```

The first apply pages through channel history, then checks for messages that
arrived during the backfill. A later apply reads history after the exact Slack
message timestamp saved for that channel. A history failure leaves the last
completed cursor in place. The next apply resumes from the saved page
checkpoint. A job that finds the importer lock held makes no Slack calls.

Slack events supply replies on older threads, edits, and explicit deletions.
Keep events enabled during backfill to capture replies that arrive while it
runs. Historic thread replies require a separate backfill. File contents
stay in Slack. An edit adds an immutable revision and an explicit deletion
adds a tombstone. A failed history request records an error. A failed author
lookup stores the message with an unknown author in the attribution.

The same classifier used for repository history proposes decisions with an
exact quote, a permalink, the message timestamp, and the Slack author fields.
Apply keeps decisions proposed and sets their basis to migration.

## AGENTS.md hard rules (HAI-14726)

`mira.registry.agent_rules` mirrors hard rules written in `AGENTS.md` files
into the registry as `rule` objects anchored to the hook files that enforce
them. `AGENTS.md` stays the source of truth. Unlike the commands above, this
one runs on its own once it is configured, and it is off until then.

A bullet names a hook file when a backticked span is the hook's path, its
file name, or its file name without `.py`, and that file exists under the hook
directory. Files whose name starts with `_` are helpers, not hooks. A file
name that two hooks share names neither; only its full path does. Each bullet
that names a hook becomes one proposed rule with one path anchor per named
hook. Every other bullet is an exception: it is listed in the run report and
is not written. Paragraph lines that name a hook, and hooks no bullet names,
are counted and listed in the report only. Fenced code is skipped.

Runtime configuration (nothing is set in the image):

| Variable | Meaning |
|---|---|
| `REGISTRY_AGENT_RULES_REPOS` | Space-separated `owner/name` list. Unset: off. |
| `REGISTRY_AGENT_RULES_HOOK_DIR` | Hook directory inside each repository, for example `tools/hooks/rules`. Unset: off. |
| `REGISTRY_AGENT_RULES_MODE` | `dry-run` (default) stores the report only. `apply` writes registry rows. |
| `REGISTRY_CORPUS_ACCESS_POLICY`, `REGISTRY_IMPORT_OPERATOR` | The corpus policy and its approved import operator. A repository outside the manifest is refused before any GitHub call. |
| `MIRA_GITHUB_APP_ID`, `MIRA_GITHUB_PRIVATE_KEY` | The GitHub App that mints a read-only (`contents: read`) token for that one repository. |

Triggers, all calling the same `sync_repo`:

- the hourly registry job checks each repository's default-branch head. A
  head and mode that match the last finished run read nothing more. This path
  works on its own;
- a push to the default branch, through the registry's existing GitHub
  webhook observer. It fires only once the GitHub App is subscribed to `push`
  events;
- `POST /api/registry/agent-rules/sync` (admin) runs it now.
  `GET /api/registry/agent-rules/report` (admin who reads the corpus) returns
  the newest run per repository.

GitHub is read with REST only: the head commit, one recursive tree, and the
`AGENTS.md` blobs. No clone, no model call. A truncated tree, an HTTP error, a
timeout or any partial read writes and retires nothing; the run is saved as
failed and the next trigger retries.

Apply writes rows under the policy's corpus with source system
`agents_md_rule`, status `proposed`, repository scope, the bullet as the body
and the `AGENTS.md` blob URL at the head commit as the source URL. Imported
proposals do not expire. `registry_agent_rule_identities` keeps one identity
per rule, `registry_agent_rule_versions` every text version, and
`registry_agent_rule_runs` every run and its report. A run pairs bullets with
identities in this order:

1. same file, same text and hooks: unchanged;
2. same file and same hooks: an edit;
3. same file and same text with hook names masked: a hook rename;
4. the old hook paths are gone and their blobs sit at the paths the bullet now
   names: a hook move;
5. same text and hooks with the old `AGENTS.md` gone: the file moved;
6. a removed rule with the same file, text and hooks: it reappeared.

Anything else is a new rule. A new version is written only when the text, the
hook paths or the `AGENTS.md` path changed; the old version is kept. An edited
rule keeps its status, so an approved rule stays approved. A bullet that is
gone, or whose hook file is gone, retires its rule with `source_removed`. When
it comes back the rule returns to the status it had. A rule a person rejected
stays rejected.

Use `dry-run` first and approve the reported change set before setting
`REGISTRY_AGENT_RULES_MODE=apply`. Switching mode reruns at the same head.
