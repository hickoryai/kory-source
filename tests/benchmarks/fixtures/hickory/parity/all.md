# Kory Source vs Greptile parity report (HAI-11607)

**Verdict: inconclusive** (arm `registry_off`)

- Units scored: 46 of 46; Greptile missing: 0 (never counted as a win).
- Δ severity-weighted true defects per unit (Kory − Greptile): mean -0.8406, 95% CI [-1.0974, -0.5504]; non-inferiority margin −0.2217.
- Judge: grok@59aa6fc18f3f4b929dd9d59a6f3bfc11/grok-4.7, unknown (families ['xai']; reviewer families ['anthropic', 'openai', 'xai']).
- Inconclusive because: judge shares a model family with a reviewer call; judge failed on 30% of units > 5%.

## Per arm

| arm | Kory defect F1 | Greptile defect F1 | golden F1 | golden recall | Kory P | Greptile P | Kory evidence | coverage | latency mean s | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| registry_on | 33.1867 ± 14.9089 | 86.6933 ± 2.3501 | 7.3867 ± 4.449 | 0.0433 | 0.8148 | 0.8571 | 3.0 | 0.5797 | 136.8069 | inconclusive |
| registry_off | 29.8967 ± 7.1065 | 87.1867 ± 1.4296 | 5.0733 ± 4.467 | 0.0303 | 0.9474 | 0.8687 | 3.0 | 1.0 | 146.298 | inconclusive |

## Registry A/B

- on: 33.1867 ± 14.9089 (runs [40.0, 12.5, 47.06])
- off: 29.8967 ± 7.1065 (runs [33.33, 36.36, 20.0])
- rule: keep on iff mean(on) - mean(off) > 2 * pstdev(off); blocks seen: 0
- decision: **off** (no unit received a registry block, so the arms are identical)

## Disagreement and cross-check

- both_not_real: 0
- both_real: 6
- both_uncertain: 1
- greptile_only_not_real: 13
- greptile_only_real: 80
- greptile_only_uncertain: 131
- kory_only_not_real: 1
- kory_only_real: 12
- kory_only_uncertain: 21
- not_real_fixed_later: 5
- not_real_total: 14
- real_fixed_later: 68
- real_total: 98
- uncertain_fixed_later: 78
- uncertain_total: 153

## Seat usage

- review: 1154 calls, 1016 seat sessions, models {'claude@22afd9be/opus': 11, 'claude@22afd9be/claude-fable-5-1[1m]': 211, 'claude@068f70d4/claude-fable-5-1[1m]': 171, 'codex@6bbbc1eb678f4f6784dd3d50c65872bd/gpt-6-astra': 81, 'claude@068f70d4/opus': 396, 'grok@59aa6fc18f3f4b929dd9d59a6f3bfc11/grok-4.7': 145, 'unknown': 138, 'codex@6bbbc1eb678f4f6784dd3d50c65872bd/gpt-5.6-terra': 1}
- judge: 319 calls, 242 seat sessions, models {'grok@59aa6fc18f3f4b929dd9d59a6f3bfc11/grok-4.7': 242, 'unknown': 77}
- metered: 1297 of 1500 seat sessions, 0 refused at the cap

`fixed_later` is biased toward Greptile (authors fixed what Greptile showed them) and can include base-branch merges into the PR's own files.
