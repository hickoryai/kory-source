# Validating decision history (HAI-12825)

How a person checks that the registry's provenance and PR answers are right
before any registry transition (HAI-12532 / HAI-12533) relies on them. Passing
tests or a merged PR does not mark HAI-12825 done; a named human's readiness
acceptance does.

## Gates, in order, each separately approved

1. Data owner approves the enumerated corpus (both repositories, every Linear
   team by id) and each application reader. The approval reference goes into
   `REGISTRY_CORPUS_ACCESS_POLICY`.
2. Each SQL impersonator is approved for **every** table in the database the
   reader can open, not just registry tables (`pg_read_all_data`).
3. Deployment of the additive schema and service, then gateway exposure.
4. Shared-environment imports, each run's coverage reviewed.
5. Human readiness acceptance with the evidence below.

## Checklist

- [ ] Negative access: unlisted engineer with `kory.source`, unlisted Kory Code
      owner, dashboard admin not listed, expired and revoked members, forged
      body fields, guessed ids — all get the uniform denial
      (`tests/test_registry_access.py`, `tests/test_source_questions.py`).
- [ ] Open an imported decision; compare its quote and source URL with the
      GitHub / Linear / commit text.
- [ ] Inspect decision author, source account, invoker, reviewer/approver and
      importer/extractor separately; check original vs normalized model labels
      and the unknown fields.
- [ ] Follow a code anchor across a rename; inspect older locations and any
      `pending` / `dangling` anchors.
- [ ] Ask "Does PR #X override any prior decisions?" for a confirmed, a
      proposed/conflicting, a compatible and an insufficient-evidence example;
      check every cited evidence id and the base/head SHAs.
- [ ] Replay an import: no new source revisions, no altered approvals.
- [ ] Inspect `/api/registry/coverage`: incomplete runs, unknown totals and
      excluded scope are stated.

## Real-history corpus (to be pinned with the coordinator)

- hickoryai/kory-source#11: a human overrode import scope to all teams. The PR
  body alone is not proof of the original instruction; expect unverified
  human approval until the linked Linear evidence is imported.
- hickoryai/kory-source#8: contradiction and merge behavior; separate the PR
  narrative from evidence of the product decision.
- hickoryai/kory-source#13: advisory citations; verify no fabricated approval
  or supersession from advisory wording.
- One human-confirmed hickory-ai supersession, one compatible change, one
  unrelated PR, chosen from real history. Do not manufacture a positive
  example when the record is incomplete.

Record each case's PR URL, frozen head, verdict, cited evidence ids and the
reviewer's agreement in the restricted audit artifact, never in a public link.
