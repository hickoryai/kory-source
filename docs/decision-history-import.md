# Decision history import (HAI-12825)

Read-only imports that turn source history into immutable evidence and
proposed decisions. They never write to GitHub, Linear or the repository, and
they never approve anything.

All three need an enabled corpus policy (see
[decision-provenance.md](decision-provenance.md)) and
`REGISTRY_IMPORT_OPERATOR` naming an approved `operator`. Scope is checked
before the first source call: a repository or team outside the manifest is
refused and never fetched. Shared-environment imports need their own
separate approval; nothing here runs automatically.

| Command | Reads |
|---|---|
| `python -m mira.registry.linear inventory` | Team list (to report unapproved teams), counts for manifest teams. |
| `python -m mira.registry.linear import [--team KEY] [--apply]` | Issues and comments of manifest teams; each description and comment also stored as a revision with creator / user / botActor attribution. |
| `python -m mira.registry.github_history import [--repo owner/name] [--max-pulls N] [--apply]` | PR descriptions, conversation comments, reviews, inline review comments. REST GET only (`GITHUB_HISTORY_TOKEN`, read scope). |
| `python -m mira.registry.repository_history import --repo owner/name --checkout PATH [--commit SHA] [--max-commits N] [--apply]` | Markdown revisions along history, code comments/docstrings at the frozen commit, commit messages. Allowlisted read-only git commands. |

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

`imports.extract_decisions` is deterministic (`kory-source.decision-extractor/1`,
no model): explicit commitments ("we decided", "the decision was to",
"Decision:") and items under a "Decisions" / ADR heading, skipping questions,
negations, hypotheticals, fenced code and blockquotes. Offsets slice the
original text exactly. Every extracted decision is `proposed`, cites its
quote, carries an `extractor` attribution separate from the author, and gets
an authority tier from its `decision_author` / `director` attributions.

## Anchors

- Inline review comments anchor to their path, line and commit.
- Code comments anchor to the innermost enclosing symbol from
  `mira.index.extract` at the frozen commit, else path-only.
- Markdown passages anchor to file and heading.
- `anchor_resolution.resolve` maps a historical anchor to a later commit:
  index anchor id, exact path and symbol, git rename, then `pending` (symbol
  gone, needs a person) or `dangling`. Unsupported languages are `unsupported`.

## Known gaps (reported, not imported)

Linear documents and projects; comments not on an issue; relations beyond the
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
