"""Step-level counterfactual attribution for a failed agent run (SPEC 3.6).

For a failed task and a step ``k`` the estimand is::

    effect_k = P(success | prefix<k, do(step_k = oracle)) - P(success | prefix<k, natural step_k)

Both terms are estimated with ``R`` rollouts from the **same** recorded prefix -- never
compared against the single original failure, which may be bad luck. The oracle for step ``k``
comes from a reference trajectory, a human edit, or a stronger-model repair (recorded, since
they differ in evidential strength).

Search: a coarse screen of every step at ``r_screen`` rollouts, then refine the top
``top_k`` steps at ``r_refine``. The **decisive step** is the earliest step whose effect CI
lower bound exceeds ``tau`` -- earliest, because fixing an upstream cause also repairs every
downstream step, and we want the root, not a symptom.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel
from scipy import stats as sps

from causeval.attribution.harness import AgentHarness, ToolBackend
from causeval.attribution.trace import Step, StepKind, Trace

OracleSource = Literal["reference", "human", "stronger_model"]


@dataclass
class AttributionCase:
    """Everything :func:`attribute_failure` needs; what a CLI ``case`` factory returns."""

    harness: AgentHarness
    task: str
    failed_trace: Trace
    oracle_trace: Trace
    tools: ToolBackend
    tau: float = 0.2
    r_refine: int = 20
    oracle_source: OracleSource = "reference"


class StepEffect(BaseModel):
    schema_version: str = "0.1"
    step_index: int
    kind: StepKind
    tool: str | None = None
    p_oracle: float
    p_natural: float
    effect: float
    ci_low: float
    ci_high: float
    ci_level: float = 0.95
    r_oracle: int
    r_natural: int
    fidelity: str = "recorded"


class AttributionResult(BaseModel):
    schema_version: str = "0.1"
    task: str
    oracle_source: OracleSource
    tau: float
    effects: list[StepEffect]
    decisive_step: int | None  # earliest step with CI lower bound > tau, else None

    def summary(self) -> str:
        lines = [
            f"Attribution for task {self.task!r} (oracle={self.oracle_source}, tau={self.tau})"
        ]
        for e in self.effects:
            mark = "  <-- DECISIVE" if e.step_index == self.decisive_step else ""
            lines.append(
                f"  step {e.step_index} [{e.kind}{'/' + e.tool if e.tool else ''}]: "
                f"effect={e.effect:+.3f} [{int(e.ci_level * 100)}% CI {e.ci_low:+.3f}, "
                f"{e.ci_high:+.3f}] (P_oracle={e.p_oracle:.2f}, P_nat={e.p_natural:.2f}){mark}"
            )
        if self.decisive_step is None:
            lines.append("  (no decisive step: no effect CI clears tau)")
        return "\n".join(lines)


def _effect_ci(
    oracle: list[float], natural: list[float], *, level: float
) -> tuple[float, float, float, float]:
    """Difference of two independent success rates with a normal-approx CI (p_o, p_n, lo, hi)."""
    po = float(np.mean(oracle)) if oracle else 0.0
    pn = float(np.mean(natural)) if natural else 0.0
    effect = po - pn
    z = float(sps.norm.ppf(1 - (1 - level) / 2))
    se = float(np.sqrt(po * (1 - po) / len(oracle) + pn * (1 - pn) / len(natural)))
    return po, pn, effect - z * se, effect + z * se


async def _rollout_successes(
    harness: AgentHarness, task: str, prefix: list[Step], tools: ToolBackend, r: int
) -> list[float]:
    out: list[float] = []
    for _ in range(r):
        trace = await harness.run(task, forced_prefix=list(prefix), tools=tools)
        out.append(1.0 if harness.success(task, trace) else 0.0)
    return out


async def attribute_failure(
    harness: AgentHarness,
    task: str,
    failed_trace: Trace,
    oracle_trace: Trace,
    tools: ToolBackend,
    *,
    tau: float = 0.2,
    r_screen: int = 2,
    r_refine: int = 20,
    top_k: int = 3,
    oracle_source: OracleSource = "reference",
    level: float = 0.95,
) -> AttributionResult:
    """Attribute a failed run to its decisive step by counterfactual replay."""
    n = min(len(failed_trace.steps), len(oracle_trace.steps))

    async def effect_at(k: int, r: int) -> tuple[float, float, float, float]:
        prefix = failed_trace.prefix(k)
        oracle_step = oracle_trace.steps[k]
        oracle_run = await _rollout_successes(harness, task, [*prefix, oracle_step], tools, r)
        natural_run = await _rollout_successes(harness, task, prefix, tools, r)
        po, pn, lo, hi = _effect_ci(oracle_run, natural_run, level=level)
        return po, pn, lo, hi

    # 1. coarse screen every step at r_screen.
    screen: list[float] = []
    for k in range(n):
        po, pn, _, _ = await effect_at(k, r_screen)
        screen.append(po - pn)

    # 2. refine the top_k most promising steps at r_refine (others keep the screen R).
    refine = set(sorted(range(n), key=lambda k: screen[k], reverse=True)[:top_k])

    effects: list[StepEffect] = []
    for k in range(n):
        r = r_refine if k in refine else r_screen
        po, pn, lo, hi = await effect_at(k, r)
        step = failed_trace.steps[k]
        effects.append(
            StepEffect(
                step_index=k,
                kind=step.kind,
                tool=step.tool,
                p_oracle=po,
                p_natural=pn,
                effect=po - pn,
                ci_low=lo,
                ci_high=hi,
                ci_level=level,
                r_oracle=r,
                r_natural=r,
            )
        )

    decisive = next((e.step_index for e in effects if e.ci_low > tau), None)
    return AttributionResult(
        task=task,
        oracle_source=oracle_source,
        tau=tau,
        effects=effects,
        decisive_step=decisive,
    )


def decisive_step_histogram(results: list[AttributionResult]) -> dict[str, int]:
    """Histogram of decisive steps across tasks, keyed by ``<kind>`` or ``<kind>/<tool>``."""
    hist: dict[str, int] = {}
    for res in results:
        if res.decisive_step is None:
            hist["none"] = hist.get("none", 0) + 1
            continue
        e = res.effects[res.decisive_step]
        key = f"{e.kind}/{e.tool}" if e.tool else e.kind
        hist[key] = hist.get(key, 0) + 1
    return hist
