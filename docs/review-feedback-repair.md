# Review feedback repair

Classification keeps a comment as `human_review` only when its account is not
Bot, its login has no `[bot]` suffix, it is neither a configured AI reviewer
nor the posting bot, and its text has no leading robot marker (after whitespace
and quote markers). Other comments remain as `bot_review`, with their actors.
Kory Source's own comments are audit-only and excluded from synthesis.

Configure the policy with normal runtime configuration:

```yaml
feedback:
  known_ai_reviewers:
    - greptile-apps
    - copilot-pull-request-reviewer
    - claude
    - coderabbitai
    - kory-source
  human_weight: 1.0
  acted_bot_weight: 0.5
  bot_weight: 0.1
```

Weights must satisfy `bot_weight < acted_bot_weight < human_weight`. Ignored
bot signal defaults to one tenth of human signal; verified human action gets
half. Approval remains independent: synthesized rules still need admin
approval. Confidence orders active rules and scales registry mirror confidence.
A bot rule contradicting human feedback is suppressed rather than combined
with it. The existing indexing model compares topics; invalid or unavailable
comparison results withhold bot rules until comparison succeeds. Review models,
passes, posting and review prompt templates are unchanged.

Each new rule records the exact feedback-event IDs returned and validated
from synthesis. A historical rule has no source IDs; its original implicit
confidence was 1.0. Repair lowers legacy
human-pattern rules whose source evidence cannot be established, preserves original confidence, and
marks `uncertain: may include bot comments`. A comparison with surviving human
feedback can restore original confidence; explicit contradiction suppresses it.
No uncertain lineage is represented as exact provenance.

## After merge, dev only

Do not run repair before the implementation merges. Do not deploy or mutate
production. Obtain an independently verified production connection and a dev
connection through the normal runtime credential flow. The operator sets
`REVIEW_REPAIR_PROD_DATABASE_URL` and `REVIEW_REPAIR_DEV_DATABASE_URL`; credentials
must not be committed or printed. The production connection is used in a
read-only transaction without initializing the application schema. Any
misclassified production row stops the operation. Missing production access
also stops the operation.

First produce a dry-run report outside the checkout:

```sh
python -m mira.analysis.repair_review_feedback \
  --bot-login review-app --report /secure/runtime/feedback-plan.jsonl
```

After reviewing the report, run the same command with a new report path and
`--apply`. It inventories every repository in the dev database, with no row
limit, reclassifies rows in place, updates rule evidence, and synchronizes
existing registry mirrors so suppression cannot be bypassed there. It never
deletes feedback. Each report contains full before-state snapshots, every row
and rule changed, original/new confidence, corroborating/contradicting human
event IDs, comparison failures, and before/after signal counts per repository.
Reports contain private runtime data: retain them outside source/build inputs.
The tool refuses a report path inside this checkout.

A partial run leaves an outcome for each completed change and can be planned
again. Preserve its report. Original rule confidence remains stored across
reruns. To reverse a confidence change, restore `original_confidence` using the
rule evidence API and retain the report; to reverse classification, restore the
row's reported before signal/category/own-comment values. Admin approval and
active status are preserved. No relay restart is needed.
