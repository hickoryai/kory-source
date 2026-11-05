# Decision provenance, authority, and corpus access (HAI-12825)

The registry keeps the original evidence behind each decision, who said it,
who ran or directed the agent that said it, who approved it, and who copied
and interpreted it. This page is the contract. The import commands are in
[decision-history-import.md](decision-history-import.md).

## Tables (migration `001_provenance`)

Migrations live in `src/mira/registry/migrations/NNN_name.sql`, apply once in
order, and are recorded in `registry_schema_migrations` with their SHA-256.
They are additive. A ledger row the code does not know, or an edited released
file, stops startup. Concurrent startups serialize (SQLite `BEGIN IMMEDIATE`,
PostgreSQL advisory lock).

| Table | Holds |
|---|---|
| `registry_source_revisions` | One immutable row per (source system, type, id, content SHA-256): original text, labels, payload, base/head/commit SHA, availability, corpus, import run. An edit adds a row. |
| `registry_decision_evidence` | Exact quote and offsets into a revision, with role (statement, directive, approval, correction, realization, support) and stated/inferred. Quotes are validated against the stored text. |
| `registry_attributions` | One row per role per evidence: original labels, normalized values, per-field state (`stated` / `inferred` / `unknown`) with evidence pointer. |
| `registry_decision_versions` | Append-only claim versions: scope, basis, approval status and evidence, realization, authority tier and policy version. |
| `registry_relationship_evidence` | One record per relationship submission, with its stance and clauses. |
| `registry_anchor_evidence` | Historical anchors: repository, observed commit, path, symbol, lines or Markdown heading, resolution status. Never overwritten when code moves. |
| `registry_import_runs` | Importer/extractor identity and version, frozen scope, checkpoint, coverage and error manifests with hashes. |
| `registry_corpus_policy`, `registry_corpus_members` | Audit copies of the deployment policy. Not authoritative. |

`registry_objects`, `registry_relationships`, `registry_comments` and
`registry_anchors` gain `corpus_id` (and objects `policy_version`,
`authority_tier`, `decision_revision`). `registry_objects` stays the current
projection.

## Attribution roles

`decision_author`, `source_account`, `invoking_actor`, `director`, `reviewer`,
`approver`, `importer`, `extractor`. Each row keeps `original_identity`,
`original_provider`, `original_bot`, `original_model`, `original_client`,
`original_environment` verbatim, and `normalized_identity`, `provider_family`,
`model_family`, `model_version`, `client_interface`, `execution_environment`
only where `attribution-normalize/1` has a mapping.

Rules:

- Unknown stays unknown: `NULL` with a reason in `field_states`, never `''`.
- A bot display name or account never yields a model, a version, or a human
  invoker. `Claude Code` names no single client interface.
- A GitHub account of type `User` is a person's account; the text it posted
  may be agent-authored, so authorship from the account alone is `inferred`.
- `Co-Authored-By:`, `Model:`, `Client:`, `Environment:`, `Session:`, `Run:`,
  `Invoked-By:`, `Directed-By:` and "Generated with" lines are self-reported
  statements by the text. They are `stated` (the source says so), not verified.
- Git author and committer are source accounts with `actor_kind` unknown.
- Importer and extractor identities are their own roles and never fill source
  model fields.

The labels in the ticket (`claude`, `opus`, `5.5`, `cli`, `dev-workstation`,
`zclaes@hickory.ai`, `automated-review`) are examples, not asserted
attributions.

## Authority policy `authority/1` (ratified 2026-11-05)

1. `explicit_human`: a person's attributable statement, or a scoped human
   approval naming the decision.
2. `human_directed_agent`: agent author with source-backed bounded human
   direction.
3. `bot_proposal`: autonomous suggestions, automated reviews, summaries, and
   unknown authors.
4. `inference`: code-derived or similarity. Never approval or supersession.

- A merge is realization only. A clean merge no longer approves an agent
  proposal and never applies a proposed supersession; a person must approve.
  (This changes `Registry._merge`; see `test_clean_merge_records_realization...`.)
- A PR approval review is approval of that PR revision, not of decisions in it.
- Newer, repeated, merged or more confident statements never silently override
  an approved decision. Only an approved `supersedes` or `corrects` edge does,
  and the older decision and evidence stay visible.
- Confidence is evidence certainty (high/medium/low/unknown with reasons),
  not authority, and does not decay with age.
- Legacy rows keep their status; `evidence_for` returns a legacy projection
  that marks quote, invoker and approval evidence unknown.

### Relation-specific cycles

- `supersedes` and `corrects` are directed ancestry (newer -> older). Their
  union must be a DAG. Proposed edges count; rejected, withdrawn and
  kept-existing edges are audit-only. Reactivating one re-validates.
  Validation and write are serialized (advisory lock / `BEGIN IMMEDIATE`), so
  concurrent inverse inserts admit exactly one.
- `compatible_with` and `conflicts_with` are symmetric. `Registry.relate`
  merges a B->A submission into an existing A->B pair; each submission keeps
  its own `registry_relationship_evidence` record. A self-pair is refused as
  meaningless input.
- `constrains`, `derived_from`, `relates`, and the narrows/widens conflict
  classifications are cross-links. Mixed cycles are valid; traversal uses
  visited sets.

## Corpus access boundary

Imported history (GitHub repositories and every Linear team in the manifest,
private teams included) is one corpus with one explicitly approved audience.
Engineering role, dashboard admin, `kory.source`, `kind=human`, or a model
header grant nothing.

`REGISTRY_CORPUS_ACCESS_POLICY` (deployment-owned JSON):

```json
{
  "schema": "kory-source-corpus-policy/1",
  "corpus_id": "hickory-complete-history",
  "policy_version": "2026-11-05.1",
  "enabled": true,
  "approval": {
    "reference": "<data-owner approval URL>",
    "approved_by": "<namespaced id>",
    "approved_at": "2026-11-05T00:00:00Z",
    "expires_at": "2027-02-01T00:00:00Z"
  },
  "repositories": [{"id": "<GitHub node id>", "full_name": "hickoryai/kory-source"}],
  "linear_teams": [{"id": "<Linear team id>", "key": "HAI"}],
  "readers": [{"namespace": "registry-jwt", "id": "<user UUID>", "approval_reference": "..."}],
  "import_operators": [{"namespace": "operator", "id": "<job identity>"}],
  "legacy_reviewer_sources": []
}
```

- Missing, invalid, disabled or expired policy denies all protected imports and
  reads. Wildcards are refused; every repository and team is enumerated.
- Reader namespaces: `registry-jwt` (verified JWT `sub`; the gateway signs the
  resolved owner UUID there) and `dashboard-account` (numeric account id).
  Display names and usernames are never keys.
- The policy is read per request: removing a member denies their next call even
  with an unexpired token.
- Every service read and mutation takes an `access` context. Without
  membership a protected object is `NotFound` with the same message as an
  absent one; counts, concepts, anchors, chains, relationships, queues and sync
  status exclude it.
- Unclassified imported rows (an imported source with no `corpus_id`) are denied
  to everyone, readers included, until `ProvenanceStore.classify_legacy` assigns
  manifest teams. `mira_learned_rule` mirrors and direct registry objects keep
  their prior visibility.
- The PR reviewer (prompt block, `search_decisions`, contradiction findings,
  uncited advisories) reads with no corpus context. Linear ticket mirrors reach
  an uncited advisory only if `legacy_reviewer_sources` lists `linear`.
- The proposal-time conflict judge compares only legacy-visible claims, so a
  judgment returned to an agent never quotes corpus text.

The SQL reader (`pg_read_all_data`) sees the whole database. It is not row
isolation and needs its own approval for every accessible table; this policy
neither grants nor implies it.

### Rollout note

Until a policy is deployed and `classify_legacy` has run with `--apply`, Linear
work mirrored by HAI-11606 is invisible to `get_context(ticket=...)` and the
dashboard. That is the intended fail-closed default, not a regression to fix
by widening access.
