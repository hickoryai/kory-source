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
It uses `mira.llm.router_acp.RouterAcpProvider` with tracing required, regardless
of `MIRA_ROUTER_ACP_ALLOW_UNTRACED`. The runtime needs the relay settings and
tracing endpoint described by that provider; no provider API keys are used.
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
OTLP endpoint and relay configuration. Live runtime estimates must distinguish
cold-cache seat work from source/Git processing time: the initial backfill can
require many bounded executions, whereas unchanged passages require no further
model calls. A bounded sample is not a complete history inventory or permission
to run a shared-environment import.
