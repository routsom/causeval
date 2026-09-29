"""Unit + coverage-simulation tests for adaptive repeat allocation (SPEC 3.3, Phase 6)."""

from __future__ import annotations

import numpy as np
import pytest

from causeval.stats.adaptive import allocate_repeats, repeats_per_item
from causeval.stats.estimate import cluster_bootstrap_ci


def test_allocation_sums_to_budget() -> None:
    alloc = allocate_repeats([1.0, 2.0, 3.0], 60)
    assert alloc.sum() == 60


def test_allocation_proportional_to_sd() -> None:
    alloc = allocate_repeats([1.0, 3.0], 40)
    # the noisier item gets ~3x the extras.
    assert alloc[1] > alloc[0]
    assert alloc[0] + alloc[1] == 40


def test_zero_variance_items_get_nothing() -> None:
    alloc = allocate_repeats([0.0, 0.0, 2.0], 30)
    assert alloc[0] == 0 and alloc[1] == 0
    assert alloc[2] == 30


def test_all_zero_variance_allocates_nothing() -> None:
    assert allocate_repeats([0.0, 0.0], 20).sum() == 0


def test_cap_redistributes_overflow() -> None:
    alloc = allocate_repeats([1.0, 1.0, 1.0], 30, max_per_item=5)
    assert alloc.max() <= 5
    assert alloc.sum() == 15  # capped at 5 each


def test_repeats_per_item_adds_base() -> None:
    reps = repeats_per_item([1.0, 2.0], 30, base_repeats=2)
    assert reps.min() >= 2
    assert reps.sum() == 2 * 2 + 30


@pytest.mark.sim
def test_adaptive_allocation_keeps_coverage() -> None:
    """Data-dependent allocation must not break CI coverage of the grand mean (B1 bounds)."""
    rng = np.random.default_rng(0)
    n_items = 25
    # fixed item population: true means and heterogeneous within-item noise.
    mu = rng.uniform(0.2, 0.8, n_items)
    sd = rng.uniform(0.05, 0.5, n_items)
    truth = float(mu.mean())

    n_sims = 200
    base, extra = 3, 60
    hits = 0
    for _ in range(n_sims):
        # phase 1: base repeats -> estimate each item's sd.
        base_draws = rng.normal(mu[:, None], sd[:, None], size=(n_items, base))
        est_sd = base_draws.std(axis=1, ddof=1)
        # phase 2: allocate extra repeats by the *estimated* sd, then draw them.
        alloc = allocate_repeats(est_sd, extra)
        item_means = []
        for i in range(n_items):
            total = base + int(alloc[i])
            draws = rng.normal(mu[i], sd[i], size=total)
            item_means.append(float(draws.mean()))
        lo, hi = cluster_bootstrap_ci(np.array(item_means), rng=rng, n_boot=800)
        hits += int(lo <= truth <= hi)

    coverage = hits / n_sims
    assert coverage >= 0.90  # lenient offline bound; strict [0.93, 0.97] is nightly
