"""B5 -- agent attribution benchmark (SPEC 6).

Claim: on a failed agent run with a fault injected at a known step ``k`` (a wrong tool
argument), counterfactual replay names step ``k`` as the decisive step. Accept: decisive step
equals ``k`` in >= 90% of tasks.

Toy tool environment (deterministic, read-only): a calculator, a key/value lookup, and a unit
converter. The reference agent runs a fixed plan whose later steps consume earlier outputs (so
a wrong early argument poisons the chain). Success is the final answer matching the expected
value, with a small independent execution-noise flip so rollouts vary and the CIs are real.

Offline uses this scripted agent; the live version swaps in a real model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from causeval.attribution.attribute import (
    AttributionResult,
    attribute_failure,
    decisive_step_histogram,
)
from causeval.attribution.harness import DictToolBackend, ToolBackend
from causeval.attribution.trace import Step, Trace

RESULTS_DIR = Path("bench/results")

# sentinel: "use the previous step's output as this argument".
PREV = "__PREV__"

_LOOKUP_TABLE = {"alpha": 5.0, "beta": 8.0, "gamma": 3.0, "delta": 11.0}


def _calculator(op: str, a: float, b: float) -> float:
    return a + b if op == "add" else a * b


def _lookup(key: str) -> float:
    return _LOOKUP_TABLE.get(key, -999.0)  # a wrong key yields an obviously wrong value


def _unit_converter(value: float, factor: float) -> float:
    return value * factor


def toy_tools() -> DictToolBackend:
    return DictToolBackend(
        {"calculator": _calculator, "lookup": _lookup, "unit_converter": _unit_converter}
    )


PlanStep = tuple[str, dict[str, Any]]


@dataclass
class ReferenceAgent:
    """Runs a fixed plan; can be forced from a prefix; success is noisy so rollouts vary.

    ``plan`` steps may use :data:`PREV` for an argument to consume the previous step's output.
    A fault at ``fault_step`` overrides that step's args with ``fault_override``.
    """

    plan: list[PlanStep]
    expected: float
    fault_step: int | None = None
    fault_override: dict[str, Any] = field(default_factory=dict)
    flip: float = 0.1  # P(success flips away from the deterministic correctness)
    seed: int = 0
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    async def run(self, task: str, *, forced_prefix: list[Step], tools: ToolBackend) -> Trace:
        steps: list[Step] = [s.model_copy(deep=True) for s in (forced_prefix or [])]
        for i in range(len(steps), len(self.plan)):
            tool, raw_args = self.plan[i]
            args = dict(raw_args)
            if PREV in args.values():
                prev_out = steps[-1].output if steps else 0.0
                args = {k: (prev_out if v == PREV else v) for k, v in args.items()}
            if i == self.fault_step:
                args = {**args, **self.fault_override}
            output = tools.call(tool, **args)
            steps.append(Step(index=i, kind="tool_call", tool=tool, args=args, output=output))
        return Trace(steps=steps, final_output=steps[-1].output if steps else None)

    def success(self, task: str, trace: Trace) -> bool:
        correct = abs(float(trace.final) - self.expected) < 1e-9
        flip = self._rng.random() < self.flip
        return (not correct) if flip else correct


def build_task(seed: int) -> tuple[ReferenceAgent, ReferenceAgent, Trace, Trace, int]:
    """Build a (faulty agent, oracle agent, failed trace, oracle trace, fault_step) tuple.

    Task: ``result = (lookup(key) + c) * f``. The 3-step plan is lookup -> add -> multiply;
    the fault is a wrong argument at a randomly chosen step.
    """
    rng = np.random.default_rng(seed)
    key = str(rng.choice(list(_LOOKUP_TABLE)))
    c = float(rng.integers(1, 9))
    f = float(rng.integers(2, 5))
    base = _LOOKUP_TABLE[key]
    expected = (base + c) * f

    plan: list[PlanStep] = [
        ("lookup", {"key": key}),
        ("calculator", {"op": "add", "a": PREV, "b": c}),
        ("unit_converter", {"value": PREV, "factor": f}),
    ]
    fault_step = int(rng.integers(0, len(plan)))
    # a plausible wrong argument for the chosen step.
    overrides: dict[int, dict[str, Any]] = {
        0: {"key": "wrong_key"},
        1: {"b": c + 3.0},
        2: {"factor": f + 2.0},
    }
    fault_override = overrides[fault_step]

    oracle_agent = ReferenceAgent(plan=plan, expected=expected, fault_step=None, seed=seed)
    faulty_agent = ReferenceAgent(
        plan=plan,
        expected=expected,
        fault_step=fault_step,
        fault_override=fault_override,
        seed=seed + 1,
    )
    return faulty_agent, oracle_agent, Trace(), Trace(), fault_step


async def _record_traces(faulty: ReferenceAgent, oracle: ReferenceAgent) -> tuple[Trace, Trace]:
    tools = toy_tools()
    failed = await faulty.run("task", forced_prefix=[], tools=tools)
    oracle_trace = await oracle.run("task", forced_prefix=[], tools=tools)
    return failed, oracle_trace


async def run_one(
    seed: int, *, tau: float = 0.2, r_refine: int = 20
) -> tuple[int, AttributionResult]:
    """Build a task, attribute its failure, and return (true fault_step, result)."""
    faulty, oracle, _, _, fault_step = build_task(seed)
    failed_trace, oracle_trace = await _record_traces(faulty, oracle)
    result = await attribute_failure(
        faulty,
        "task",
        failed_trace,
        oracle_trace,
        toy_tools(),
        tau=tau,
        r_refine=r_refine,
    )
    return fault_step, result


async def coverage(*, n_tasks: int = 40, seed: int = 0, tau: float = 0.2) -> dict[str, Any]:
    """Fraction of tasks where the decisive step equals the injected fault step."""
    hits = 0
    results: list[AttributionResult] = []
    fault_steps: list[int] = []
    for t in range(n_tasks):
        fault_step, result = await run_one(seed + t, tau=tau)
        results.append(result)
        fault_steps.append(fault_step)
        if result.decisive_step == fault_step:
            hits += 1
    return {
        "n_tasks": n_tasks,
        "accuracy": hits / n_tasks,
        "histogram": decisive_step_histogram(results),
        "fault_step_counts": {
            int(k): int(v) for k, v in zip(*np.unique(fault_steps, return_counts=True), strict=True)
        },
    }


def run_offline(*, n_tasks: int = 40, seed: int = 0) -> tuple[dict[str, Any], Path, Path]:
    import asyncio

    cov = asyncio.run(coverage(n_tasks=n_tasks, seed=seed))
    payload = {
        "meta": {
            "title": "B5 agent attribution (offline synthetic)",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_tasks": n_tasks,
            "note": "Scripted reference agent in a toy tool env; not a real LLM.",
        },
        **cov,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "b5_agent_attribution_offline.json"
    md_path = RESULTS_DIR / "b5_agent_attribution_offline.md"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(_render_markdown(payload))
    return payload, json_path, md_path


def _render_markdown(payload: dict[str, Any]) -> str:
    m = payload["meta"]
    lines = [f"# {m['title']}", ""]
    for k, v in m.items():
        if k != "title":
            lines.append(f"- **{k}**: {v}")
    lines += [
        "",
        f"**Decisive-step accuracy (== injected fault step): {payload['accuracy']:.3f}**",
        "",
        f"- decisive-step histogram: {payload['histogram']}",
        f"- injected fault-step counts: {payload['fault_step_counts']}",
        "",
        "> B5 target: decisive step equals the injected fault step in >= 90% of tasks.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="causeval.bench.agent_attribution", description=__doc__)
    parser.add_argument("--n-tasks", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    payload, json_path, md_path = run_offline(n_tasks=args.n_tasks, seed=args.seed)
    print(f"decisive-step accuracy = {payload['accuracy']:.3f}  (target >= 0.9)")
    print(f"wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
