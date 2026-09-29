"""A minimal fake agent + tool backend for later attribution tests (SPEC §3.6).

Kept intentionally lightweight and self-contained: the real ``Trace``/``Step`` schemas and
the ``AgentHarness`` protocol land in Phase 5. This fake uses plain dicts as steps so it can
exist before that module does. It runs a scripted plan of tool calls and can be forced to
resume from a prefix, which is what step-level counterfactual replay needs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Step = dict[str, Any]  # {"kind", "tool", "args", "output"}


class ToolBackend:
    """Maps tool name -> callable. Deterministic; no side effects."""

    def __init__(self, tools: dict[str, Callable[..., Any]]) -> None:
        self._tools = tools

    def call(self, tool: str, **args: Any) -> Any:
        if tool not in self._tools:
            raise KeyError(f"unknown tool {tool!r}")
        return self._tools[tool](**args)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)


class FakeAgentHarness:
    """Runs a fixed plan of ``(tool, args)`` steps, optionally forced from a prefix.

    ``success`` is defined by a user predicate over the final output. A fault can be
    injected at a known step index (wrong argument) so attribution tests have a ground
    truth decisive step.
    """

    def __init__(
        self,
        plan: list[tuple[str, dict[str, Any]]],
        *,
        success_fn: Callable[[str, Any], bool],
        fault_step: int | None = None,
        fault_args: dict[str, Any] | None = None,
    ) -> None:
        self.plan = plan
        self.success_fn = success_fn
        self.fault_step = fault_step
        self.fault_args = fault_args or {}

    async def run(
        self,
        task: str,
        *,
        forced_prefix: list[Step] | None = None,
        tools: ToolBackend,
    ) -> list[Step]:
        trace: list[Step] = list(forced_prefix or [])
        start = len(trace)
        for i in range(start, len(self.plan)):
            tool, args = self.plan[i]
            if i == self.fault_step:
                args = {**args, **self.fault_args}
            output = tools.call(tool, **args)
            trace.append({"kind": "tool_call", "tool": tool, "args": args, "output": output})
        return trace

    def success(self, task: str, trace: list[Step]) -> bool:
        final = trace[-1]["output"] if trace else None
        return self.success_fn(task, final)
