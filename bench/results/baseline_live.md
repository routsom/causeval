# Baseline variance (live)

- **title**: Baseline variance (live)
- **generated_at**: 2026-09-30T20:12:56.415813+00:00
- **judge_model**: claude-haiku-4-5-20251001
- **repeats**: 5
- **n_items**: 6
- **threshold**: 0.7
- **note**: Live judge calls (Anthropic Claude Haiku 4.5). Small run for demonstration.

| metric | n | R | mean [95% CI] | mean within-item var | pass/fail flip frac | flaky items | failed |
|---|---|---|---|---|---|---|---|
| AnswerRelevancyMetric | 6 | 5 | 1.000 [1.000, 1.000] | 0.0000 | 0.00 | 0 | 0 |
| ContextualRelevancyMetric | 6 | 5 | 1.000 [1.000, 1.000] | 0.0000 | 0.00 | 0 | 0 |
| FaithfulnessMetric | 6 | 5 | 1.000 [1.000, 1.000] | 0.0000 | 0.00 | 0 | 0 |

> A nonzero flip fraction means a single DeepEval sample would pass or fail the same item depending on luck. This is the motivation for repeated sampling and CIs.
