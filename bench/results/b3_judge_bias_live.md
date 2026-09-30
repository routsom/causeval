# B3 judge bias (live)

- **judge_model**: claude-haiku-4-5-20251001
- **generated_at**: 2026-09-30T20:26:11.587123+00:00
- **n_pairs (position)**: 12   **n_items (pointwise)**: 12
- **note**: Real judge; biases are measured, not injected. Effect flagged when its CI excludes 0 and exceeds the tolerance.

| probe | measured effect | 95% CI | flagged | note |
|---|---|---|---|---|
| position | -0.292 | [-0.422, -0.126] | YES | order-consistency=0.42 |
| verbosity | -0.061 | [-0.086, -0.024] | YES |  |
| formatting | -0.021 | [-0.042, +0.002] | no |  |
| authorship | -0.025 | [-0.097, -0.001] | no |  |

> Position effect = P(judge picks the first-presented answer) - 0.5, from judging every pair in both orders. Verbosity/formatting/authorship = mean score(transform) - score(original).
