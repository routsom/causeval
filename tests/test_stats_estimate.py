"""B1 (stats coverage) + variance-recovery simulation tests for `stats.estimate`.

The offline fast versions use <=200 sims with lenient binomial-tolerance bounds so they
never flake; the strict [0.93, 0.97] coverage target from SPEC B1 is the nightly goal
(more sims), asserted here with a wider band.
"""

from __future__ import annotations

import numpy as np
import pytest

from causeval.core.schemas import Measurement
from causeval.stats.estimate import cluster_bootstrap_ci, estimate, variance_components

MU = 0.6
SIGMA_B = 0.15
SIGMA_W = 0.10


# Representative subset of the B1 (n, R) grid for the fast offline suite; the full grid at
# >=1000 sims with the strict [0.93, 0.97] band is the nightly target (SPEC B1 / section 7).
@pytest.mark.sim
@pytest.mark.parametrize(
    ("n", "r"),
    [(20, 1), (20, 10), (50, 3), (200, 10)],
)
def test_cluster_bootstrap_ci_coverage(n: int, r: int) -> None:
    """Studentized 95% CI covers the true mean at ~0.95 across the B1 (n, R) grid."""
    rng = np.random.default_rng(1234 + n * 10 + r)
    n_sims = 200
    # Item mean = item effect + mean of R repeat noises; Var = sigma_b^2 + sigma_w^2 / R.
    item_mean_sd = np.sqrt(SIGMA_B**2 + SIGMA_W**2 / r)
    covered = 0
    for _ in range(n_sims):
        item_means = MU + rng.normal(0.0, item_mean_sd, size=n)
        lo, hi = cluster_bootstrap_ci(item_means, level=0.95, n_boot=800, rng=rng)
        if lo <= MU <= hi:
            covered += 1
    coverage = covered / n_sims
    # Binomial tolerance around 0.95 for 200 sims (never-flake band).
    assert 0.92 <= coverage <= 0.99, f"coverage {coverage} out of band for n={n}, R={r}"


@pytest.mark.sim
def test_variance_components_recovery() -> None:
    """ANOVA method-of-moments recovers the injected between/within components."""
    rng = np.random.default_rng(0)
    n, r = 400, 8
    scores_by_item = {}
    for i in range(n):
        item_effect = rng.normal(MU, SIGMA_B)
        scores_by_item[f"i{i}"] = list(item_effect + rng.normal(0.0, SIGMA_W, size=r))
    between, within, icc = variance_components(scores_by_item)
    assert between == pytest.approx(SIGMA_B**2, rel=0.25)
    assert within == pytest.approx(SIGMA_W**2, rel=0.15)
    expected_icc = SIGMA_B**2 / (SIGMA_B**2 + SIGMA_W**2)
    assert icc == pytest.approx(expected_icc, abs=0.05)


def test_variance_components_single_repeat_has_zero_within() -> None:
    # R == 1: within-item noise is unidentifiable -> reported as 0.
    between, within, icc = variance_components({"a": [0.2], "b": [0.8], "c": [0.5]})
    assert within == 0.0
    assert between > 0.0
    assert icc == 1.0


def _measurements(scores_by_item: dict[str, list[float]], *, threshold: float) -> list[Measurement]:
    out = []
    for item_id, scores in scores_by_item.items():
        for rep, s in enumerate(scores):
            out.append(
                Measurement(
                    item_id=item_id,
                    metric="M",
                    repeat_index=rep,
                    score=s,
                    passed=s >= threshold,
                )
            )
    return out


def test_estimate_end_to_end_wiring() -> None:
    rng = np.random.default_rng(7)
    scores = {f"i{i}": list(rng.uniform(0, 1, size=5)) for i in range(30)}
    ms = _measurements(scores, threshold=0.5)
    est = estimate(ms, rng=rng, threshold=0.5)
    assert est.metric == "M"
    assert est.n_items == 30 and est.n_repeats == 5
    assert est.ci_low < est.estimate < est.ci_high
    assert 0.0 <= est.estimate <= 1.0
    assert est.small_sample is False  # n == 30


def test_estimate_small_sample_reports_t_interval() -> None:
    rng = np.random.default_rng(7)
    scores = {f"i{i}": [0.4, 0.6, 0.5] for i in range(10)}
    ms = _measurements(scores, threshold=0.5)
    est = estimate(ms, rng=rng, threshold=0.5)
    assert est.small_sample is True
    assert est.t_ci_low is not None and est.t_ci_high is not None


def test_estimate_flaky_items_flagged() -> None:
    rng = np.random.default_rng(0)
    # item "flip" passes half the time; "stable" always passes.
    ms = [
        Measurement(item_id="flip", metric="M", repeat_index=r, score=s, passed=(s >= 0.5))
        for r, s in enumerate([0.2, 0.9, 0.1, 0.8, 0.95, 0.05])
    ]
    ms += [
        Measurement(item_id="stable", metric="M", repeat_index=r, score=0.9, passed=True)
        for r in range(6)
    ]
    est = estimate(ms, rng=rng, threshold=0.5)
    assert "flip" in est.flaky_items
    assert "stable" not in est.flaky_items


def test_estimate_excludes_failed_measurements() -> None:
    rng = np.random.default_rng(0)
    ms = [Measurement(item_id="a", metric="M", repeat_index=0, score=0.7, passed=True)]
    ms.append(Measurement(item_id="a", metric="M", repeat_index=1, error="boom"))
    est = estimate(ms, rng=rng, threshold=0.5)
    assert est.n_failed == 1
    assert est.n_items == 1
