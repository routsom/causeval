# Baseline variance (offline synthetic smoke)

- **title**: Baseline variance (offline synthetic smoke)
- **generated_at**: 2026-09-29T09:22:02.698862+00:00
- **judge_model**: synthetic (seeded)
- **repeats**: 10
- **threshold**: 0.7
- **note**: Synthetic data; not a real judge. Use run_live for the committed deliverable.

| metric | n | R | mean [95% CI] | mean within-item var | pass/fail flip frac | flaky items | failed |
|---|---|---|---|---|---|---|---|
| AnswerRelevancyMetric | 30 | 10 | 0.661 [0.623, 0.701] | 0.0149 | 0.93 | 10 | 0 |
| ContextualRelevancyMetric | 30 | 10 | 0.714 [0.675, 0.751] | 0.0117 | 0.80 | 10 | 0 |
| FaithfulnessMetric | 30 | 10 | 0.692 [0.661, 0.723] | 0.0135 | 0.93 | 16 | 0 |

> A nonzero flip fraction means a single DeepEval sample would pass or fail the same item depending on luck. This is the motivation for repeated sampling and CIs.
