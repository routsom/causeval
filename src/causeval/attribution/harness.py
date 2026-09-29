"""Agent harness + tool backend protocols (SPEC 3.6).

The user implements an :class:`AgentHarness` that can (a) run a task, optionally forced to
resume from a recorded prefix, and (b) judge success. Counterfactual replay drives it: it
re-runs from ``prefix<k`` many times to estimate success probabilities. A reference
implementation lives in ``bench/agent_attribution.py``.

``ToolBackend`` is the surface the agent calls tools through; :class:`DictToolBackend` is a
simple deterministic one. :class:`ToolSpec` carries the replay policy flags used by the
cassette (``safe_live`` for read-only tools, ``side_effecting`` for tools never called live).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from causeval.attribution.trace import Step, Trace


@runtime_checkable
class ToolBackend(Protocol):
    def call(self, tool: str, **args: Any) -> Any: ...

    @property
    def names(self) -> list[str]: ...


@dataclass(frozen=True)
class ToolSpec:
    """A tool plus its replay policy. ``safe_live`` read-only tools may run on novel inputs."""

    name: str
    fn: Callable[..., Any]
    safe_live: bool = False
    side_effecting: bool = False


class DictToolBackend:
    """A deterministic tool backend backed by a name -> callable map."""

    def __init__(self, tools: dict[str, Callable[..., Any]]) -> None:
        self._tools = tools

    def call(self, tool: str, **args: Any) -> Any:
        if tool not in self._tools:
            raise KeyError(f"unknown tool {tool!r}")
        return self._tools[tool](**args)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)


class AgentHarness(Protocol):
    async def run(self, task: str, *, forced_prefix: list[Step], tools: ToolBackend) -> Trace: ...

    def success(self, task: str, trace: Trace) -> bool: ...
