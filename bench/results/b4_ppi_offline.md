# B4 prediction-powered inference (offline synthetic)

- **generated_at**: 2026-09-29T19:45:11.187903+00:00
- **n_total**: 1000
- **n_labeled**: 100
- **judge_bias**: 0.1
- **theta_true**: 0.5169063382672536
- **note**: Synthetic judge with systematic bias; not a real LLM.

**theta (true human mean) = 0.5169**

| estimator | coverage of theta | mean 95% CI width |
|---|---|---|
| PPI | 0.990 | 0.0612 |
| human-only | 0.975 | 0.1116 |
| naive judge mean | 0.000 | (biased) |

> B4 target: PPI covers theta ~95% and is narrower than human-only at equal label count; the naive judge mean covers poorly because of its systematic bias.