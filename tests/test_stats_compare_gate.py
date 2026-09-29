"""B1 gate error-rate + power simulations, and compare/gate unit tests."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from causeval.core.errors import InsufficientDataError
from causeval.core.schemas import Measurement
from causeval.stats.compare import compare, compare_pass_rates, mcnemar_exact_p
from causeval.stats.gate import gate, gate_verdict, holm_adjust

SIGMA_B = 0.15
SIGMA_W = 0.10


def _paired_runs(
    *, n: int, r: int, true_delta: float, rng: np.random.Generator, metric: str = "M"
) -> tuple[list[Measurement], list[Measurement]]:
    base: list[Measurement] = []
    cand: list[Measurement] = []
    for i in range(n):
        effect = rng.normal(0.0, SIGMA_B)
        for rep in range(r):
            bs = effect + rng.normal(0.0, SIGMA_W)
            cs = effect + true_delta + rng.normal(0.0, SIGMA_W)
            base.append(Measurement(item_id=f"i{i}", metric=metric, repeat_index=rep, score=bs))
            cand.append(Measurement(item_id=f"i{i}", metric=metric, repeat_index=rep, score=cs))
    return base, cand


# ---- unit tests -----------------------------------------------------------------


def test_gate_verdict_three_valued() -> None:
    assert gate_verdict(-0.10, -0.05, margin=0.02) == "regression"  # ci_high < -margin
    assert gate_verdict(-0.01, 0.03, margin=0.02) == "pass"  # ci_low > -margin
    assert gate_verdict(-0.05, 0.03, margin=0.02) == "inconclusive"


def test_holm_adjust_matches_known_values() -> None:
    adj = holm_adjust([0.01, 0.04, 0.03])
    # sorted: 0.01(*3)=0.03, 0.03(*2)=0.06, 0.04(*1)=0.04 -> monotone: 0.03,0.06,0.06
    assert adj[0] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.06)
    assert adj[1] == pytest.approx(0.06)


def test_mcnemar_symmetry_and_extremes() -> None:
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(5, 5) == 1.0
    assert mcnemar_exact_p(10, 0) < 0.01  # all discordances one way


def test_compare_requires_matched_ids() -> None:
    rng = np.random.default_rng(0)
    base, cand = _paired_runs(n=10, r=2, true_delta=0.0, rng=rng)
    cand = [m for m in cand if m.item_id != "i0"]  # drop an item from candidate
    with pytest.raises(InsufficientDataError):
        compare(base, cand, margin=0.02, rng=rng)


def test_compare_allow_partial_reports_dropped() -> None:
    rng = np.random.default_rng(0)
    base, cand = _paired_runs(n=10, r=2, true_delta=0.0, rng=rng)
    cand = [m for m in cand if m.item_id != "i0"]
    (comp,) = compare(base, cand, margin=0.02, rng=rng, allow_partial=True)
    assert comp.n_items == 9
    assert comp.n_dropped == 1


def test_compare_delta_sign_and_gate_report() -> None:
    rng = np.random.default_rng(1)
    base, cand = _paired_runs(n=60, r=5, true_delta=0.08, rng=rng)
    (comp,) = compare(base, cand, margin=0.02, rng=rng)
    assert comp.delta == pytest.approx(0.08, abs=0.03)
    report = gate([comp])
    assert report.overall == comp.verdict
    assert report.exit_code in (0, 1, 2)


# ---- B1 simulations -------------------------------------------------------------


@pytest.mark.sim
def test_gate_false_regression_rate_at_delta_zero() -> None:
    """With true Delta = 0 and a positive margin, false regressions are rare (<= 0.05)."""
    n_sims = 200
    margin = 0.02
    regressions = 0
    for s in range(n_sims):
        rng = np.random.default_rng(10_000 + s)
        base, cand = _paired_runs(n=50, r=3, true_delta=0.0, rng=rng)
        (comp,) = compare(base, cand, margin=margin, rng=rng, n_boot=400)
        if comp.verdict == "regression":
            regressions += 1
    rate = regressions / n_sims
    assert rate <= 0.05, f"false-regression rate {rate} exceeds 0.05"


@pytest.mark.sim
def test_gate_detects_true_regression_power() -> None:
    """A real 0.10 regression is detected with high power."""
    n_sims = 150
    margin = 0.02
    detected = 0
    for s in range(n_sims):
        rng = np.random.default_rng(20_000 + s)
        base, cand = _paired_runs(n=50, r=3, true_delta=-0.10, rng=rng)
        (comp,) = compare(base, cand, margin=margin, rng=rng, n_boot=400)
        if comp.verdict == "regression":
            detected += 1
    power = detected / n_sims
    assert power >= 0.85, f"power {power} too low to detect a 0.10 regression"


@pytest.mark.sim
def test_gate_declares_pass_when_non_inferior() -> None:
    """With Delta = 0 and a 0.05 tolerance, the gate confidently passes most of the time."""
    n_sims = 150
    margin = 0.05
    passed = 0
    for s in range(n_sims):
        rng = np.random.default_rng(30_000 + s)
        base, cand = _paired_runs(n=50, r=3, true_delta=0.0, rng=rng)
        (comp,) = compare(base, cand, margin=margin, rng=rng, n_boot=400)
        if comp.verdict == "pass":
            passed += 1
    rate = passed / n_sims
    assert rate >= 0.85, f"pass rate {rate} too low for a clearly non-inferior candidate"


@pytest.mark.sim
def test_power_curve_increases_with_effect_size() -> None:
    """B1 power curve: detection probability rises monotonically with |delta|."""
    n_sims = 120
    margin = 0.0  # detect any true difference
    deltas = [0.0, 0.03, 0.06, 0.10]
    powers = []
    for delta in deltas:
        detected = 0
        for s in range(n_sims):
            rng = np.random.default_rng(50_000 + int(delta * 1000) * 1000 + s)
            base, cand = _paired_runs(n=50, r=3, true_delta=-delta, rng=rng)
            (comp,) = compare(base, cand, margin=margin, rng=rng, n_boot=300)
            if comp.verdict == "regression":
                detected += 1
        powers.append(detected / n_sims)
    # Near-zero effect -> low detection (~one-sided alpha); large effect -> high power.
    assert powers[0] <= 0.10
    assert powers[-1] >= 0.90
    assert all(a <= b + 0.05 for a, b in itertools.pairwise(powers))


def test_compare_pass_rates_runs() -> None:
    rng = np.random.default_rng(3)
    base, cand = _paired_runs(n=40, r=5, true_delta=0.0, rng=rng)
    # threshold near the mean so items straddle it
    comp, mcp = compare_pass_rates(base, cand, metric="M", threshold=0.0, margin=0.1, rng=rng)
    assert comp.metric == "M[pass@0]"
    assert 0.0 <= mcp <= 1.0
