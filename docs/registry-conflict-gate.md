# Registry contradiction gate

The hickory-ai push hook (out of this repository) calls this contract before
`git push` and `gh pr create`. It asks whether the current agent session still
holds a conflict with an approved decision. This service does not install that
hook.

An answer is stored on the proposal's session. A second session does not
inherit it. `defer` answers the session and leaves the proposal `proposed`,
so a later merge still will not auto-approve it.

## Assumption

A review-time hit does not write a registry object. The review finding is the
whole output of that path. Writing a reviewer-generated proposal would add
agent-authored churn to the approval queue. The finding quotes both registry
ids when a proposal already exists, and otherwise the approved id plus the PR
clause. It always quotes both clauses, at blocker severity.

## Does this session hold an unresolved conflict?

MCP tool `session_conflicts`. The caller is a registry JWT (`Authorization:
Bearer`), agent or human. Agent tokens can read. They cannot answer.

```json
{ "session": "sess-a" }
```

Response `structuredContent`:

```json
{
  "session": "sess-a",
  "unresolved": true,
  "conflicts": [
    {
      "object_id": "reg_…",
      "existing_id": "reg_…",
      "session": "sess-a",
      "relation": "narrows",
      "new_clause": "does not grant engineers edit on Artifact-made Apps",
      "existing_clause": "engineers hold full collaborator rights on every Interactive App",
      "answer": null
    }
  ]
}
```

`unresolved` is the bit the hook blocks on. It is false once this session's
proposal has been answered `keep`, `supersede`, or `defer`. Conflicts are
omitted once answered.

HTTP, same body, dashboard session required by `/api/*` like the other
registry reads:

```
GET /api/registry/session-conflicts?session=sess-a
```

The query parameter is the session id. It is not a path segment, because a
session id may contain `/`.

## Answering

MCP tool `resolve_conflict`, human JWT only. The author of the proposal
cannot answer. HTTP is admin-only, same as approve and reject:

```
POST /api/registry/objects/{id}/resolve
{ "choice": "keep" | "supersede" | "defer" }
```

- `keep`: the new object becomes `rejected` with reason
  `conflicts_with_approved`. The approved object stays approved.
- `supersede`: the new object becomes `approved` and the old one
  `superseded`. `decided_by` is the person who answered.
- `defer`: the new object stays `proposed`. The conflict stays open for
  merge. Only this session stops reporting it as unresolved.

## Prompt counter

```
GET /api/registry/conflict-prompts
```

MCP tool `conflict_prompt_counts` returns the same object:

```json
{
  "prompts": 4,
  "answers": { "keep": 1, "supersede": 0, "defer": 1 },
  "by_session": [{ "session": "sess-a", "prompts": 2 }]
}
```

`prompts` counts seat judge calls, including review-time calls. Review-time
calls are keyed `review:{owner}/{repo}#{pr}`. `answers` is the current
keep / supersede / defer split, one row per proposal session.
