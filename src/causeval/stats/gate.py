"""Three-valued, non-inferiority CI gates.

Given a comparison's CI for ``Delta = candidate - baseline`` and a tolerance ``margin``:

* ``regression``    if ``ci_high < -margin``  (confidently worse than tolerance)
* ``pass``          if ``ci_low  > -margin``  (confidently not worse than tolerance)
* ``inconclusive``  otherwise

CLI exit codes: 0 pass, 1 regression, 2 inconclusive. When several metrics gate together,
Holm-adjusted confidence levels are applied in :mod:`causeval.stats.compare` before the
verdict is computed here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from causeval.core.schemas import Comparison

Verdict = Literal["pass", "regression", "inconclusive"]

EXIT_PASS = 0
EXIT_REGRESSION = 1
EXIT_INCONCLUSIVE = 2


def gate_verdict(ci_low: float, ci_high: float, margin: float) -> Verdict:
    """Three-valued non-inferiority verdict from a Delta CI and tolerance ``margin``."""
    if ci_high < -margin:
        return "regression"
    if ci_low > -margin:
        return "pass"
    return "inconclusive"


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, returned in the input order."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


class GateReport(BaseModel):
    """Outcome of gating one or more comparisons."""

    overall: Verdict
    exit_code: int
    comparisons: list[Comparison]


def overall_verdict(verdicts: list[Verdict]) -> Verdict:
    """The family verdict: any regression dominates, else any inconclusive, else pass."""
    if any(v == "regression" for v in verdicts):
        return "regression"
    if any(v == "inconclusive" for v in verdicts):
        return "inconclusive"
    return "pass"


def exit_code_for(verdict: Verdict) -> int:
    return {
        "pass": EXIT_PASS,
        "regression": EXIT_REGRESSION,
        "inconclusive": EXIT_INCONCLUSIVE,
    }[verdict]


def gate(comparisons: list[Comparison]) -> GateReport:
    """Aggregate comparison verdicts into a :class:`GateReport` with a CLI exit code.

    The per-comparison verdicts are taken as-is (they already reflect any Holm-adjusted
    confidence levels chosen at ``compare`` time).
    """
    verdicts: list[Verdict] = [c.verdict for c in comparisons]
    overall = overall_verdict(verdicts)
    return GateReport(
        overall=overall,
        exit_code=exit_code_for(overall),
        comparisons=comparisons,
    )
