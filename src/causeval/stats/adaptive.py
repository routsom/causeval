"""Adaptive repeat allocation (SPEC 3.3 adaptive, Phase 6).

Repeated sampling is only worth it where the judge/app is actually noisy. Given a fixed budget
of extra measurements, :func:`allocate_repeats` spends them where they shrink the estimator's
variance most -- Neyman allocation, proportional to each item's within-item standard deviation.

The grand-mean estimator (mean of item means) stays unbiased under any allocation: giving an
item more repeats only sharpens its own item mean, it does not tilt the average. Still, because
the allocation is *data-dependent* (it uses observed variances), the coverage claim must be
confirmed by simulation -- see ``tests/test_adaptive.py``.
"""

from __future__ import annotations

import numpy as np


def allocate_repeats(
    item_sd: np.ndarray | list[float],
    extra_budget: int,
    *,
    max_per_item: int | None = None,
) -> np.ndarray:
    """Neyman allocation of ``extra_budget`` repeats across items, proportional to sd.

    Returns an integer array of extra repeats per item (summing to at most ``extra_budget``).
    Items with zero variance get none. Largest-remainder rounding keeps the total exact; an
    optional ``max_per_item`` cap redistributes the overflow to the next-noisiest items.
    """
    sd = np.asarray(item_sd, dtype=float)
    n = sd.size
    alloc = np.zeros(n, dtype=int)
    if extra_budget <= 0 or n == 0:
        return alloc

    total_sd = float(sd.sum())
    if total_sd <= 0:
        return alloc  # nothing is noisy; no repeats help

    remaining = extra_budget
    active = np.ones(n, dtype=bool)
    while remaining > 0 and active.any():
        weights = np.where(active, sd, 0.0)
        wsum = float(weights.sum())
        if wsum <= 0:
            break
        raw = remaining * weights / wsum
        floor = np.floor(raw).astype(int)
        # distribute the floors, then the largest remainders.
        give = floor.copy()
        leftover = remaining - int(give.sum())
        if leftover > 0:
            order = np.argsort(-(raw - floor))
            for idx in order[:leftover]:
                if active[idx]:
                    give[idx] += 1
        alloc += give

        if max_per_item is None:
            remaining = 0
            break
        # enforce the cap; free up the overflow to reallocate next pass.
        over = np.maximum(alloc - max_per_item, 0)
        alloc = np.minimum(alloc, max_per_item)
        active = alloc < max_per_item
        remaining = int(over.sum())

    return alloc


def repeats_per_item(
    item_sd: np.ndarray | list[float],
    extra_budget: int,
    *,
    base_repeats: int = 1,
    max_per_item: int | None = None,
) -> np.ndarray:
    """Total repeats per item: a ``base_repeats`` floor plus the Neyman-allocated extras."""
    extra = allocate_repeats(item_sd, extra_budget, max_per_item=max_per_item)
    return base_repeats + extra
