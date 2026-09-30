# B5 agent attribution (live)

- **agent model**: claude-haiku-4-5 (real Claude tool-agent; single-pass, default temperature)
- **generated_at**: 2026-09-30T21:17:04.186064+00:00
- **tasks**: 5 priced-purchase tasks; a wrong tool argument is injected at a known step

**Decisive-step accuracy (== injected fault step): 0.800 (4/5)**

| task | injected fault step | decisive step found | outcome |
|---|---|---|---|
| widgetx6 | 0 | 0 | hit |
| gadgetx5 | 1 | 1 | hit |
| gizmox3 | 2 | None | miss |
| sprocketx4 | 1 | None | setup-invalid (oracle failed or fault didn't break it) |
| cogx7 | 0 | 0 | hit |
| widgetx9 | 2 | 2 | hit |

- decisive-step histogram: {'tool_call/price': 2, 'tool_call/mul': 1, 'none': 1, 'tool_call/tax': 1}

> Counterfactual replay re-runs the agent many times from each step's prefix, with and without the oracle step, and names the earliest step whose fix restores success. Here it localizes an injected wrong-argument fault in a real Claude tool-agent.
