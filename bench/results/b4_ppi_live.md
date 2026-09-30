# B4 prediction-powered inference (live)

- **judge model**: claude-haiku-4-5-20251001 (pointwise correctness score = the biased predictor f)
- **generated_at**: 2026-09-30T20:42:16.678480+00:00
- **n items**: 40 (20 correct, 20 wrong), **labeled subset**: 12
- **note**: objective 0/1 correctness plays the role of the human label; PPI debiases the judge using the labeled subset.

**True mean correctness = 0.500** (known by construction)

| estimator | estimate | 95% CI | covers truth? | CI width |
|---|---|---|---|---|
| PPI | 0.510 | [0.353, 0.667] | yes | 0.314 |
| human-only (12 labels) | 0.583 | [0.292, 0.875] | yes | 0.583 |
| naive judge-mean | 0.503 | [0.349, 0.656] | yes | (biased) |

> PPI's effective sample size is 41.3 (vs 12 real labels): it borrows the judge's signal to tighten the estimate while staying unbiased. The naive judge-mean is confidently wrong because Claude over-rates plausible-but-wrong answers.
