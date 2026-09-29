"""Agent trace schema (SPEC 3.6).

A :class:`Trace` is an ordered list of :class:`Step`s (``llm_call``, ``tool_call``,
``retrieval``, ``handoff``), each with its inputs, output, and timing. This is the internal
representation that counterfactual replay and step-level blame operate on.

Importers turn external formats into a ``Trace``: :func:`from_records` (the generic path used
by the reference harness and tests) and :func:`from_deepeval_trace` (best-effort, defensive --
DeepEval's trace API is read through the adapter, never guessed). OpenTelemetry GenAI spans
arrive in Phase 7.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

StepKind = Literal["llm_call", "tool_call", "retrieval", "handoff"]


class Step(BaseModel):
    """One step of an agent run. ``output`` and ``args`` values must be JSON-serializable."""

    schema_version: str = "0.1"
    index: int
    kind: StepKind
    tool: str | None = None  # tool/retriever name for tool_call/retrieval
    args: dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    latency_s: float | None = None


class Trace(BaseModel):
    steps: list[Step] = Field(default_factory=list)
    final_output: Any = None

    @property
    def final(self) -> Any:
        """The declared final output, or the last step's output when unset."""
        if self.final_output is not None:
            return self.final_output
        return self.steps[-1].output if self.steps else None

    def prefix(self, k: int) -> list[Step]:
        """The first ``k`` steps (the natural prefix before step ``k``)."""
        return list(self.steps[:k])


def canonical_args(args: dict[str, Any]) -> str:
    """Deterministic key for a tool call: sorted-key JSON, non-JSON values via ``str``."""
    return json.dumps(args, sort_keys=True, default=str, separators=(",", ":"))


def from_records(records: list[dict[str, Any]], *, final_output: Any = None) -> Trace:
    """Build a Trace from a list of step dicts (``kind``, ``tool``, ``args``, ``output``)."""
    steps = [
        Step(
            index=i,
            kind=r.get("kind", "tool_call"),
            tool=r.get("tool"),
            args=r.get("args", {}),
            output=r.get("output"),
            latency_s=r.get("latency_s"),
        )
        for i, r in enumerate(records)
    ]
    return Trace(steps=steps, final_output=final_output)


def from_deepeval_trace(trace: Any) -> Trace:  # pragma: no cover - needs a live DeepEval trace
    """Best-effort import of a DeepEval trace object into a :class:`Trace`.

    Reads span attributes defensively (``getattr``) because DeepEval's trace API changes across
    versions; per CLAUDE.md rule 9 the installed source is the source of truth, so this stays a
    thin, forgiving adapter rather than a hard-coded schema.
    """
    spans = getattr(trace, "spans", None) or getattr(trace, "children", None) or []
    records: list[dict[str, Any]] = []
    for span in spans:
        name = getattr(span, "name", None) or getattr(span, "type", "llm_call")
        kind: StepKind = "tool_call" if "tool" in str(name).lower() else "llm_call"
        records.append(
            {
                "kind": kind,
                "tool": getattr(span, "name", None),
                "args": getattr(span, "input", {}) or {},
                "output": getattr(span, "output", None),
            }
        )
    return from_records(records, final_output=getattr(trace, "output", None))
