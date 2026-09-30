"""Unit + simulation tests for observational AIPW estimation (SPEC 3.8)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from causeval.stats.observational import aipw_ate, estimate_effect


def _confounded(rng: np.random.Generator, n: int, tau: float):
    """Confounded logs: covariates drive both which version served and the metric.

    True ATE = tau. The naive difference-in-means is biased because high-X units both get the
    treatment more often and score higher regardless.
    """
    x = rng.normal(size=(n, 3))
    logits = 1.2 * x[:, 0] - 0.8 * x[:, 1]
    t = (rng.uniform(size=n) < expit(logits)).astype(float)
    y = tau * t + 1.5 * x[:, 0] + 0.7 * x[:, 1] - 0.5 * x[:, 2] + rng.normal(0, 0.5, n)
    return t, y, x


def test_aipw_recovers_effect_point() -> None:
    rng = np.random.default_rng(0)
    t, y, x = _confounded(rng, 3000, tau=0.5)
    ate, lo, hi, _ = aipw_ate(t, y, x, rng=rng)
    assert lo <= 0.5 <= hi
    # naive is clearly biased upward here.
    naive = y[t == 1].mean() - y[t == 0].mean()
    assert abs(naive - 0.5) > abs(ate - 0.5)


def test_estimate_effect_reports_refuters_and_warning() -> None:
    rng = np.random.default_rng(1)
    t, y, x = _confounded(rng, 2000, tau=0.3)
    res = estimate_effect(t, y, x, seed=1)
    names = {r.name for r in res.refuters}
    assert names == {"placebo_treatment", "random_common_cause", "data_subset"}
    placebo = next(r for r in res.refuters if r.name == "placebo_treatment")
    assert placebo.passed  # placebo effect CI covers 0
    assert "unobserved confounding" in res.warning
    assert res.n_treated == int(np.asarray(t).sum())


@pytest.mark.sim
def test_aipw_coverage_beats_naive() -> None:
    """Across sims AIPW covers the true ATE ~95%; the naive difference covers poorly."""
    tau = 0.4
    n_sims = 150
    aipw_hits = 0
    naive_hits = 0
    for s in range(n_sims):
        rng = np.random.default_rng(1000 + s)
        t, y, x = _confounded(rng, 1500, tau=tau)
        _ate, lo, hi, _ = aipw_ate(t, y, x, rng=rng)
        aipw_hits += int(lo <= tau <= hi)
        # a naive normal CI on the difference in means.
        y1, y0 = y[t == 1], y[t == 0]
        diff = y1.mean() - y0.mean()
        se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
        naive_hits += int(diff - 1.96 * se <= tau <= diff + 1.96 * se)
    assert aipw_hits / n_sims >= 0.88  # lenient offline bound
    assert naive_hits / n_sims <= 0.5  # confounded -> poor coverage


@pytest.mark.sim
def test_placebo_is_null_under_no_effect() -> None:
    rng = np.random.default_rng(7)
    t, y, x = _confounded(rng, 2500, tau=0.0)
    _ate, lo, hi, _ = aipw_ate(t, y, x, rng=rng)
    assert lo <= 0.0 <= hi  # AIPW finds ~no effect when there is none
