# B3 judge bias recovery (offline synthetic)

- **generated_at**: 2026-09-29T19:45:05.079976+00:00
- **n_pairs**: 80
- **n_items**: 80
- **injected**: {'position': 0.15, 'verbosity': 0.08}
- **note**: Synthetic biased judges; not a real LLM.

| probe | injected | estimate | 95% CI | flagged | true in CI |
|---|---|---|---|---|---|
| position | +0.150 | +0.156 | [+0.078, +0.230] | True | True |
| verbosity | +0.080 | +0.070 | [+0.053, +0.085] | True | True |
| formatting | +0.000 | +0.011 | [-0.005, +0.026] | False | True |
| authorship | +0.000 | -0.008 | [-0.023, +0.007] | False | True |

> B3 target: recover injected biases inside the CI (~95% coverage over sims); non-injected ~0.