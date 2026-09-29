"""Unit + simulation tests for PPI (SPEC 3.5). B4 coverage lives in test_b4_ppi.py."""

from __future__ import annotations

import numpy as np
import pytest

from causeval.core.errors import InsufficientDataError
from causeval.judge_audit.ppi import optimal_lambda, ppi_mean, ppi_mean_ci


def _synthetic(rng: np.random.Generator, n_lab: int, n_unlab: int, bias: float):
    """Judge f = Y + bias + noise; Y ~ Bernoulli-ish quality in [0,1]."""
    n = n_lab + n_unlab
    y_all = rng.uniform(0, 1, size=n)
    f_all = np.clip(y_all + bias + rng.normal(0, 0.1, size=n), 0, 1)
    return y_all[:n_lab], f_all[:n_lab], f_all[n_lab:], float(y_all.mean())


def test_lambda_one_recovers_classic_ppi() -> None:
    rng = np.random.default_rng(0)
    yl, fl, fu, _ = _synthetic(rng, 50, 500, 0.2)
    est = ppi_mean_ci(yl, fl, fu, lam=1.0)
    theta = 1.0 * fu.mean() + (yl - fl).mean()
    assert est.estimate == pytest.approx(theta)
    assert est.lam == 1.0


def test_lambda_zero_is_human_only_point_estimate() -> None:
    rng = np.random.default_rng(1)
    yl, fl, fu, _ = _synthetic(rng, 40, 400, 0.3)
    est = ppi_mean_ci(yl, fl, fu, lam=0.0)
    assert est.estimate == pytest.approx(float(np.mean(yl)))


def test_optimal_lambda_zero_when_predictor_constant() -> None:
    y = np.array([0.0, 1.0, 0.5, 0.2])
    f = np.array([0.5, 0.5, 0.5, 0.5])
    assert optimal_lambda(y, f, 100) == 0.0


def test_ppi_narrower_than_human_only() -> None:
    rng = np.random.default_rng(2)
    yl, fl, fu, _ = _synthetic(rng, 50, 2000, 0.15)
    est = ppi_mean_ci(yl, fl, fu)
    assert est.ci_width < est.human_only_ci_width
    assert est.effective_n > est.n_labeled


def test_random_sample_guard_blocks_non_subset() -> None:
    rng = np.random.default_rng(3)
    yl, fl, fu, _ = _synthetic(rng, 10, 100, 0.1)
    with pytest.raises(InsufficientDataError, match="subset of the recorded random sample"):
        ppi_mean(
            yl,
            fl,
            fu,
            labeled_ids=[f"x{i}" for i in range(10)],
            sampling_plan_ids=[f"y{i}" for i in range(10)],
        )


def test_random_sample_guard_requires_plan() -> None:
    rng = np.random.default_rng(4)
    yl, fl, fu, _ = _synthetic(rng, 10, 100, 0.1)
    with pytest.raises(InsufficientDataError, match="recorded random sampling plan"):
        ppi_mean(yl, fl, fu)
    # override lets it through
    est = ppi_mean(yl, fl, fu, i_know_the_labels_are_random=True)
    assert est.n_labeled == 10


def test_crosscheck_against_ppi_py() -> None:
    ppi_py = pytest.importorskip("ppi_py")
    rng = np.random.default_rng(5)
    yl, fl, fu, _ = _synthetic(rng, 60, 600, 0.2)
    ours = ppi_mean_ci(yl, fl, fu, lam=1.0, level=0.95)
    # ppi_py's power-tuning kwarg is ``lhat`` (fix at 1.0 = classic PPI for an exact match).
    lo, hi = ppi_py.ppi_mean_ci(yl, fl, fu, alpha=0.05, lhat=1.0)
    assert ours.ci_low == pytest.approx(float(lo), abs=1e-6)
    assert ours.ci_high == pytest.approx(float(hi), abs=1e-6)
