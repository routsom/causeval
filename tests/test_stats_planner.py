"""Planner: the recommended (n, R) actually achieves the target power (SPEC section 7)."""

from __future__ import annotations

import numpy as np
import pytest

from causeval.core.schemas import Measurement
from causeval.stats.estimate import cluster_bootstrap_ci
from causeval.stats.planner import (
    VarianceForPlanning,
    estimate_variance_for_planning,
    plan_power,
)

SIGMA_W = 0.10


def test_plan_power_shrinks_n_as_repeats_grow() -> None:
    var = VarianceForPlanning(var_diff_between=0.0, var_within_a=0.01, var_within_b=0.01)
    opts = {o.n_repeats: o for o in plan_power(delta=0.05, variance=var, r_options=(1, 3, 10))}
    # More repeats -> less within noise per item -> fewer items needed.
    assert opts[1].n_items > opts[3].n_items > opts[10].n_items
    assert all(o.est_cost_usd is None for o in opts.values())


def test_plan_power_includes_cost_when_price_given() -> None:
    var = VarianceForPlanning(var_diff_between=0.0, var_within_a=0.01, var_within_b=0.01)
    (opt,) = plan_power(delta=0.05, variance=var, r_options=(3,), cost_per_call_usd=0.001)
    assert opt.est_cost_usd == pytest.approx(opt.total_calls * 0.001)


@pytest.mark.sim
def test_planned_sample_size_hits_target_power() -> None:
    """Simulate at the planner's recommended n and confirm ~80% detection of `delta`."""
    delta = 0.05
    r = 3
    var = VarianceForPlanning(
        var_diff_between=0.0, var_within_a=SIGMA_W**2, var_within_b=SIGMA_W**2
    )
    (opt,) = plan_power(delta=delta, variance=var, power=0.8, r_options=(r,))
    n = opt.n_items

    n_sims = 250
    detected = 0
    for s in range(n_sims):
        rng = np.random.default_rng(40_000 + s)
        diffs = np.empty(n)
        for i in range(n):
            # item effect cancels in the paired difference -> only within noise remains
            base = rng.normal(0.0, SIGMA_W, size=r)
            cand = delta + rng.normal(0.0, SIGMA_W, size=r)
            diffs[i] = float(cand.mean() - base.mean())
        lo, _ = cluster_bootstrap_ci(diffs, level=0.95, n_boot=400, rng=rng)
        if lo > 0.0:  # CI excludes 0 on the positive side -> effect detected
            detected += 1
    power = detected / n_sims
    # Target 0.80; binomial tolerance over 250 sims keeps this non-flaky.
    assert 0.72 <= power <= 0.92, f"achieved power {power} off target for n={n}, R={r}"


def test_estimate_variance_for_planning_from_pilot() -> None:
    rng = np.random.default_rng(0)
    base: list[Measurement] = []
    cand: list[Measurement] = []
    for i in range(40):
        effect = rng.normal(0.5, 0.15)
        for rep in range(4):
            base.append(
                Measurement(
                    item_id=f"i{i}",
                    metric="M",
                    repeat_index=rep,
                    score=effect + rng.normal(0, SIGMA_W),
                )
            )
            cand.append(
                Measurement(
                    item_id=f"i{i}",
                    metric="M",
                    repeat_index=rep,
                    score=effect + 0.03 + rng.normal(0, SIGMA_W),
                )
            )
    v = estimate_variance_for_planning(base, cand, metric="M")
    assert v.var_within_a == pytest.approx(SIGMA_W**2, rel=0.4)
    assert v.var_within_b == pytest.approx(SIGMA_W**2, rel=0.4)
    # item effects cancel in the difference, so between-diff variance is near 0
    assert v.var_diff_between < 0.01
