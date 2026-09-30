# B6 IRT pruning (live, real systems)

- **generated_at**: 2026-09-30T20:48:37.961288+00:00
- **systems**: 6 real Claude models   **items**: 30 (hard short-answer)
- **prune target**: 50% of items, by 2PL Fisher information

**Pruning to 15/30 items keeps the system ranking: Kendall tau = 1.000**

| system | accuracy (full set) |
|---|---|
| claude-sonnet-4-5 | 1.00 |
| claude-opus-4-8 | 0.97 |
| claude-haiku-4-5 | 0.93 |
| claude-sonnet-5 | 0.93 |
| claude-opus-4-5 | 0.93 |
| claude-fable-5 | 0.93 |

> IRT scores each item's difficulty and discrimination, then keeps the most informative half. If the ranking of systems on the pruned set matches the full set (Kendall tau >= 0.9), you can evaluate future systems on half the items at half the cost.
