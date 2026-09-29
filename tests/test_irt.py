"""Unit + simulation tests for 2PL IRT and pruning (SPEC 3.3, 6)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import pearsonr

from causeval.bench.irt_bench import simulate_2pl
from causeval.stats.irt import IRTModel, fisher_information, fit_2pl, prune, rank_systems


def test_fit_requires_five_systems() -> None:
    with pytest.raises(ValueError, match="at least 5 systems"):
        fit_2pl(np.ones((10, 4)))


def test_fit_shapes_and_prob() -> None:
    x, _, _, _ = simulate_2pl(20, 8, seed=0)
    model = fit_2pl(x)
    assert len(model.discrimination) == 20
    assert len(model.difficulty) == 20
    assert len(model.ability) == 8
    p = model.prob(0, 0.0)
    assert 0.0 <= p <= 1.0


def test_prob_monotonic_in_ability() -> None:
    model = IRTModel(
        discrimination=[1.5], difficulty=[0.0], ability=[0.0], n_iter=1, converged=True
    )
    assert model.prob(0, -2.0) < model.prob(0, 0.0) < model.prob(0, 2.0)


def test_fisher_information_peaks_at_difficulty() -> None:
    model = IRTModel(
        discrimination=[2.0], difficulty=[0.5], ability=[0.0], n_iter=1, converged=True
    )
    at_b = fisher_information(model, 0, np.array([0.5]))
    away = fisher_information(model, 0, np.array([3.0]))
    assert at_b > away  # 2PL information is maximal at theta == difficulty


def test_prune_selects_most_informative_items() -> None:
    # item 0 is highly discriminating near the abilities; item 1 is flat/uninformative.
    model = IRTModel(
        discrimination=[2.5, 0.1],
        difficulty=[0.0, 0.0],
        ability=[0.0, 0.1, -0.1],
        n_iter=1,
        converged=True,
    )
    assert prune(model, 1) == [0]


def test_rank_systems_orders_by_score() -> None:
    # system 2 solves everything, system 0 nothing.
    x = np.array([[0, 1, 1], [0, 0, 1], [0, 1, 1]], dtype=float)
    ranks = rank_systems(x)
    assert ranks[2] == 0  # best -> rank 0
    assert ranks[0] == 2  # worst -> rank 2


@pytest.mark.sim
def test_2pl_parameter_recovery() -> None:
    x, a, b, theta = simulate_2pl(80, 500, seed=0)
    model = fit_2pl(x)
    et = np.asarray(model.ability)
    flip = np.sign(float(pearsonr(theta, et)[0])) or 1.0
    corr_b = abs(float(pearsonr(b, flip * np.asarray(model.difficulty))[0]))
    corr_a = abs(float(pearsonr(a, np.asarray(model.discrimination))[0]))
    corr_t = abs(float(pearsonr(theta, et)[0]))
    assert corr_b >= 0.9
    assert corr_a >= 0.9
    assert corr_t >= 0.9


def test_crosscheck_against_girth() -> None:
    girth = pytest.importorskip("girth")
    x, a, b, _ = simulate_2pl(60, 300, seed=0)
    est = girth.twopl_mml(x.astype(int))
    # different estimator (marginal MLE), so expect strong but not exact agreement.
    assert abs(float(pearsonr(b, est["Difficulty"])[0])) >= 0.85
    assert abs(float(pearsonr(a, est["Discrimination"])[0])) >= 0.75
