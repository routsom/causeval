"""Unit + simulation tests for conformal selective judging (SPEC 3.5, 7)."""

from __future__ import annotations

import numpy as np
import pytest

from causeval.judge_audit.conformal import (
    agreement_confidence,
    apply_threshold,
    calibrate_threshold,
    clopper_pearson_upper,
)


def test_agreement_confidence() -> None:
    assert agreement_confidence(["a", "a", "a"]) == 1.0
    assert agreement_confidence(["a", "a", "b"]) == pytest.approx(2 / 3)
    assert agreement_confidence([]) == 0.0


def test_clopper_pearson_edges() -> None:
    assert clopper_pearson_upper(0, 0, delta=0.1) == 1.0
    assert clopper_pearson_upper(5, 5, delta=0.1) == 1.0
    # 0 errors in 100 -> a small but positive upper bound
    u = clopper_pearson_upper(0, 100, delta=0.05)
    assert 0.0 < u < 0.05


def test_calibrate_prefers_more_coverage_when_safe() -> None:
    # confidence perfectly separates: high-confidence items never err.
    rng = np.random.default_rng(0)
    conf = np.concatenate([rng.uniform(0.8, 1.0, 200), rng.uniform(0.0, 0.5, 200)])
    err = np.concatenate([np.zeros(200), np.ones(200)])  # only low-confidence items err
    t = calibrate_threshold(conf, err, alpha=0.1, delta=0.1)
    assert t.calibrated
    assert t.cal_error_upper <= 0.1
    # it should accept roughly the clean high-confidence half.
    assert 0.4 <= t.cal_coverage <= 0.55


def test_calibrate_abstains_when_nothing_is_safe() -> None:
    rng = np.random.default_rng(1)
    conf = rng.uniform(0, 1, 100)
    err = np.ones(100)  # judge always wrong -> no safe threshold
    t = calibrate_threshold(conf, err, alpha=0.05, delta=0.1)
    assert not t.calibrated
    assert t.threshold == float("inf")


def test_apply_threshold_routes_low_confidence_to_human() -> None:
    # calibrate on a real set (clean high-confidence, erroring low-confidence).
    rng = np.random.default_rng(2)
    conf = np.concatenate([rng.uniform(0.7, 1.0, 300), rng.uniform(0.0, 0.4, 300)])
    err = np.concatenate([np.zeros(300), np.ones(300)])
    t = calibrate_threshold(conf, err, alpha=0.2, delta=0.2)
    assert t.calibrated
    res = apply_threshold(["a", "b", "c"], [0.99, 0.2, 0.92], t, verdicts=[1.0, 0.0, 1.0])
    by_id = {v.item_id: v for v in res.verdicts}
    assert by_id["b"].accepted is False and by_id["b"].verdict is None
    assert by_id["a"].accepted is True and by_id["a"].verdict == 1.0


def _draw(rng: np.random.Generator, n: int):
    """Confidence ~ U(0,1); P(judge errs) = 0.6*(1-conf)^2 (monotone decreasing in conf)."""
    conf = rng.uniform(0.0, 1.0, n)
    p_err = 0.6 * (1.0 - conf) ** 2
    err = (rng.uniform(0, 1, n) < p_err).astype(float)
    return conf, err, p_err


@pytest.mark.sim
def test_ltt_risk_control() -> None:
    """Across sims, the true error among accepted exceeds alpha at most ~delta of the time."""
    alpha, delta = 0.1, 0.1
    n_sims = 200
    rng = np.random.default_rng(7)
    violations = 0
    covered = []
    for _ in range(n_sims):
        conf, err, _ = _draw(rng, 500)
        t = calibrate_threshold(conf, err, alpha=alpha, delta=delta)
        if not t.calibrated:
            continue
        # true error among accepted, from a large fresh holdout at this threshold.
        h_conf, h_err, _ = _draw(rng, 20000)
        accepted = h_conf >= t.threshold
        true_err = float(h_err[accepted].mean()) if accepted.any() else 0.0
        violations += int(true_err > alpha)
        covered.append(float(accepted.mean()))
    # LTT guarantees P(true error > alpha) <= delta; allow binomial slack offline.
    assert violations / n_sims <= delta + 0.05
    # and the rule should actually accept a useful fraction, not abstain trivially.
    assert np.mean(covered) > 0.3
