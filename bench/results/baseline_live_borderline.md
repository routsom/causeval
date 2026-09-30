# Baseline variance (live, borderline items)

- **title**: Baseline variance (live, borderline items)
- **generated_at**: 2026-09-30T20:19:14.998057+00:00
- **judge_model**: claude-haiku-4-5-20251001
- **temperature**: 1.0
- **repeats**: 5
- **n_items**: 5
- **threshold**: 0.7
- **note**: Live Claude Haiku 4.5 at temperature 1.0 on deliberately borderline items, to surface real judge flakiness.

| metric | n | R | mean [95% CI] | mean within-item var | pass/fail flip frac | flaky items | failed |
|---|---|---|---|---|---|---|---|
| AnswerRelevancyMetric | 5 | 5 | 0.833 [0.633, 1.000] | 0.0000 | 0.00 | 0 | 0 |
| ContextualRelevancyMetric | 5 | 5 | 0.173 [0.000, 0.373] | 0.0011 | 0.00 | 0 | 0 |
| FaithfulnessMetric | 5 | 5 | 1.000 [1.000, 1.000] | 0.0000 | 0.00 | 0 | 0 |

> A nonzero flip fraction means a single DeepEval sample would pass or fail the same item depending on luck. This is the motivation for repeated sampling and CIs.
