"""Unit tests for judge calibration (SPEC 3.5)."""

from __future__ import annotations

import numpy as np
import pytest

from causeval.core.errors import InsufficientDataError
from causeval.judge_audit.calibration import (
    IsotonicMap,
    _pava,
    calibrate,
    cohen_kappa,
    expected_calibration_error,
)


def test_pava_is_monotone_nondecreasing() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0])  # not monotone
    _xs, fitted = _pava(x, y)
    assert np.all(np.diff(fitted) >= -1e-12)
    assert fitted[0] <= fitted[-1]


def test_isotonic_map_predict_clamps() -> None:
    m = IsotonicMap(x=[0.2, 0.5, 0.9], y=[0.1, 0.5, 0.95])
    assert m.predict(-1.0) == 0.1  # left clamp
    assert m.predict(2.0) == 0.95  # right clamp
    assert 0.1 <= m.predict(0.4) <= 0.5  # interpolated


def test_cohen_kappa_bounds() -> None:
    a = np.array([1, 1, 0, 0])
    assert cohen_kappa(a, a) == pytest.approx(1.0)
    # independent-ish -> near 0
    assert cohen_kappa(np.array([1, 0, 1, 0]), np.array([0, 1, 0, 1])) == pytest.approx(-1.0)


def test_perfect_calibration_low_ece() -> None:
    fitted = np.array([0.05, 0.15, 0.85, 0.95])
    human = np.array([0, 0, 1, 1])
    assert expected_calibration_error(fitted, human, n_bins=10) < 0.2


def test_calibrate_recovers_monotone_relationship() -> None:
    rng = np.random.default_rng(0)
    js = rng.uniform(0, 1, 400)
    # human passes with probability increasing in judge score.
    human = (rng.uniform(0, 1, 400) < js).astype(float)
    rep = calibrate(js, human, threshold=0.5, rng=rng)
    assert rep.spearman > 0.3
    assert rep.ece_ci_low <= rep.ece <= rep.ece_ci_high
    # isotonic map should be non-decreasing.
    assert np.all(np.diff(rep.isotonic.y) >= -1e-12)


def test_calibrate_needs_two_items() -> None:
    with pytest.raises(InsufficientDataError):
        calibrate([0.5], [1.0])
