"""Sample-size / power planning from a pilot run.

For a paired comparison, the variance of the estimated difference is

    Var(Delta_hat) ~= (sigma^2_diff_between + sigma^2_within_A / R + sigma^2_within_B / R) / n

so the item count needed to detect a true difference ``delta`` at two-sided level
``1 - alpha`` with power ``1 - beta`` is

    n = ceil( (z_{1-alpha/2} + z_{1-beta})^2 * V_R / delta^2 )   where
    V_R = sigma^2_diff_between + (sigma^2_within_A + sigma^2_within_B) / R.

The planner returns a small table of ``(n, R, expected CI half-width, calls, cost)`` options
so the user can trade repeats against items. Prices, if supplied, come from user config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

from causeval.core.schemas import Measurement
from causeval.stats.aggregate import group_scores
from causeval.stats.estimate import variance_components


@dataclass(frozen=True)
class VarianceForPlanning:
    var_diff_between: float
    var_within_a: float
    var_within_b: float


@dataclass(frozen=True)
class PlanOption:
    n_items: int
    n_repeats: int
    expected_ci_halfwidth: float
    total_calls: int
    est_cost_usd: float | None


def _per_item_diff_variance(v: VarianceForPlanning, r: int) -> float:
    return v.var_diff_between + (v.var_within_a + v.var_within_b) / r


def plan_power(
    *,
    delta: float,
    variance: VarianceForPlanning,
    power: float = 0.8,
    level: float = 0.95,
    r_options: tuple[int, ...] = (1, 3, 5, 10),
    calls_per_item_per_repeat: int = 1,
    cost_per_call_usd: float | None = None,
) -> list[PlanOption]:
    """Return ``(n, R, half-width, calls, cost)`` options to detect ``delta`` at ``power``."""
    if delta <= 0:
        raise ValueError("delta (the effect size to detect) must be positive")
    z_alpha = float(sps.norm.ppf(1 - (1 - level) / 2))
    z_beta = float(sps.norm.ppf(power))
    factor = (z_alpha + z_beta) ** 2

    options: list[PlanOption] = []
    for r in r_options:
        v_r = _per_item_diff_variance(variance, r)
        n = max(2, math.ceil(factor * v_r / (delta**2)))
        half_width = z_alpha * math.sqrt(v_r / n)
        # A comparison runs both baseline and candidate: 2 runs * n * R * calls-per-repeat.
        total_calls = 2 * n * r * calls_per_item_per_repeat
        cost = total_calls * cost_per_call_usd if cost_per_call_usd is not None else None
        options.append(
            PlanOption(
                n_items=n,
                n_repeats=r,
                expected_ci_halfwidth=half_width,
                total_calls=total_calls,
                est_cost_usd=(round(cost, 6) if cost is not None else None),
            )
        )
    return options


def estimate_variance_for_planning(
    baseline: list[Measurement],
    candidate: list[Measurement],
    *,
    metric: str,
) -> VarianceForPlanning:
    """Estimate the planning variance components from a paired pilot.

    ``sigma^2_diff_between`` is recovered from the observed spread of per-item mean
    differences minus the within contribution at the pilot's repeat count, truncated at 0.
    """
    base_scores, _ = group_scores(baseline, metric=metric)
    cand_scores, _ = group_scores(candidate, metric=metric)
    matched = sorted(set(base_scores) & set(cand_scores))
    if len(matched) < 2:
        raise ValueError(f"metric {metric!r}: need >= 2 matched items to plan from a pilot")

    _, within_a, _ = variance_components({i: base_scores[i] for i in matched})
    _, within_b, _ = variance_components({i: cand_scores[i] for i in matched})

    r_pilot = max(
        min(len(base_scores[i]) for i in matched),
        1,
    )
    diffs = np.array(
        [float(np.mean(cand_scores[i])) - float(np.mean(base_scores[i])) for i in matched]
    )
    observed_diff_var = float(diffs.var(ddof=1))
    var_diff_between = max(observed_diff_var - (within_a + within_b) / r_pilot, 0.0)
    return VarianceForPlanning(
        var_diff_between=var_diff_between,
        var_within_a=within_a,
        var_within_b=within_b,
    )
