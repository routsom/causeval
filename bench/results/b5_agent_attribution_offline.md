# B5 agent attribution (offline synthetic)

- **generated_at**: 2026-09-29T20:37:18.958005+00:00
- **n_tasks**: 40
- **note**: Scripted reference agent in a toy tool env; not a real LLM.

**Decisive-step accuracy (== injected fault step): 1.000**

- decisive-step histogram: {'tool_call/lookup': 12, 'tool_call/unit_converter': 11, 'tool_call/calculator': 17}
- injected fault-step counts: {0: 12, 1: 17, 2: 11}

> B5 target: decisive step equals the injected fault step in >= 90% of tasks.