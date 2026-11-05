# Source questions: "Does PR #X override any prior decisions?" (HAI-12825)

Read-only, cited answers over the provenance in
[decision-provenance.md](decision-provenance.md). Only approved complete-corpus
readers get an answer; everyone else gets the same `404` / `not found` for every
PR, object and evidence id.

## Surfaces

| Surface | Call |
|---|---|
| Dashboard | Registry → **Ask the source**; object panels show **Evidence and attribution** and **Decision history**. |
| HTTP | `GET /api/registry/prs/{owner}/{repo}/{number}/decisions[?expected_head_sha=]` |
| HTTP | `POST /api/registry/questions` `{question, repo?, pr_number?, paths?, symbols?, as_of_commit?}` |
| HTTP | `GET /api/registry/objects/{id}/evidence`, `/objects/{id}/history`, `/evidence/{evidence_id}`, `/coverage` |
| MCP | `explain_pr_decisions(repo, pr_number, expected_head_sha?)`, `ask_source_question(question, repo?, pr_number?, paths?, symbols?, as_of_commit?)` |

Dashboard calls derive access from the session's account id; MCP calls from the
verified JWT `sub`, re-checked against the current policy on every call. The
gateway's `kory.source` capability and the caller's `kind` and model are
necessary to reach the tool but never sufficient for corpus data.

## How a PR is judged

1. **Freeze.** The review API's GET-only `SnapshotReader` captures base, head,
   merge base, title, body and the changed files, with renames and deletions
   read from the frozen diff. If `expected_head_sha` differs from the frozen
   head, the answer says so and is `insufficient_evidence`.
2. **Candidates.** Decisions anchored on any old or new changed path (live
   anchors and historical anchor evidence); decisions stated in the PR's own
   imported description, comments and reviews; decisions recorded on tickets
   the PR names, labeled `linked ticket (unanchored)`.
3. **Graph.** `history_graph.walk` follows every `supersedes`, `corrects`,
   `conflicts_with`, `compatible_with`, `constrains`, `derived_from` and
   `relates` edge both ways, expands each node once, and reports depth/node
   limits and hidden links. It does not use `chain()`'s first edge.
4. **Per-decision verdicts.**

   | Verdict | Means |
   |---|---|
   | `confirmed_supersession` | An active `supersedes`/`corrects` edge from a PR decision that a person approved. |
   | `proposed_supersession` | Such an edge exists but nobody has approved it. Not an override. |
   | `conflict` | The PR's decision conflicts with it. Behavior may change contrary to it; that is not authorization. |
   | `compatible` | Recorded as compatible. |
   | `no_supported_override_found` | The PR touches its anchored code; nothing recorded overrides it. |
   | `insufficient_evidence` | Uncertain anchor, ticket-only link, or missing evidence. |

5. **Overall verdict** is the strongest finding, except that a global
   `no_supported_override_found` is never returned while coverage is
   incomplete (GitHub import for the repo not `complete`, unavailable patches,
   truncated graph, hidden links, head race): that becomes
   `insufficient_evidence` with the limits listed.
6. **Citations** per decision: exact quotes with evidence ids and offsets,
   source revision (URL, type, commit/base/head, content hash, import run),
   every attribution role with original and normalized labels plus
   `unknown_fields` / `inferred_fields`, and a separate importer/extractor
   section. Legacy rows are marked legacy with unknown quote and approval.
7. **Narrative**: deterministic sentences over the findings, each listing the
   evidence ids it rests on; `validate_citations` refuses an id not in the
   answer. No model writes the answer.

## Other questions

- Repo plus paths or symbols: decision history for that code, oldest first by
  ancestry, with citations.
- Anything else, or a PR number without a repository: `needs_context`, naming
  what is missing. The service never guesses a repository and never runs an
  open-ended agent task.

## Never

Questions never import, approve, supersede, re-anchor, trigger reviews, post to
GitHub, or write to Linear. Tests assert every provenance table's row count is
unchanged after a question.
