"""Selective judging with a distribution-free error guarantee (SPEC 3.5, Learn-then-Test).

We accept the judge's verdict only when it is confident enough, and route the rest to a human.
The confidence threshold is calibrated so that, with probability >= 1-delta over the
calibration draw, the error rate among accepted items is <= alpha.

Confidence source:
  * provider logprobs when available (pass them in directly), else
  * agreement across ``k`` repeated judge samples -- the fraction voting for the modal verdict
    (:func:`agreement_confidence`).

Calibration (Learn-then-Test, Angelopoulos et al. 2110.01052): over a pre-specified grid of
candidate thresholds we form, for each, a Clopper-Pearson upper confidence bound on the error
rate among accepted items. A threshold is *safe* if that bound is <= alpha at a
Bonferroni-corrected level ``delta / n_grid`` (which controls the family-wise error over the
grid, so selecting the best safe threshold does not inflate the violation rate). We return the
lowest safe threshold -- maximal coverage subject to the guarantee.

A fixed grid with Bonferroni is used rather than a fixed-sequence-from-the-top test: the
highest thresholds accept too few items to certify (a wide CP bound), which would stop a
top-down sequence immediately and abstain on everything.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel
from scipy import stats as sps


def agreement_confidence(verdicts: Sequence[Any]) -> float:
    """Fraction of ``k`` repeated judge verdicts that match the modal verdict."""
    if not verdicts:
        return 0.0
    counts = Counter(verdicts)
    return counts.most_common(1)[0][1] / len(verdicts)


def clopper_pearson_upper(n_errors: int, n_total: int, *, delta: float) -> float:
    """Upper (1-delta) Clopper-Pearson bound on a binomial error rate.

    With ``n_errors`` errors out of ``n_total``, returns the largest error probability
    consistent with the data at confidence ``1-delta``. ``n_total == 0`` -> 1.0 (no evidence).
    """
    if n_total == 0:
        return 1.0
    if n_errors == n_total:
        return 1.0
    return float(sps.beta.ppf(1.0 - delta, n_errors + 1, n_total - n_errors))


class ConformalThreshold(BaseModel):
    """The calibrated acceptance rule and its calibration-set diagnostics."""

    schema_version: str = "0.1"
    threshold: float  # accept items with confidence >= threshold; inf => abstain on all
    alpha: float  # target error rate among accepted
    delta: float  # confidence level of the guarantee (1 - delta)
    method: str = "learn_then_test_clopper_pearson"
    calibrated: bool  # False when no threshold met the guarantee
    cal_coverage: float  # fraction of the calibration set accepted at the threshold
    cal_error_rate: float  # empirical error among accepted on the calibration set
    cal_error_upper: float  # Clopper-Pearson upper bound at the threshold


def calibrate_threshold(
    confidence: Sequence[float],
    errors: Sequence[float],
    *,
    alpha: float = 0.1,
    delta: float = 0.1,
    n_grid: int = 20,
) -> ConformalThreshold:
    """Choose the lowest grid threshold whose accepted-error CP bound is <= alpha.

    ``errors[i]`` is 1.0 if the judge's verdict on calibration item ``i`` is wrong, else 0.0.
    Each grid threshold is tested at the Bonferroni level ``delta / n_grid``; the lowest safe
    one (maximal coverage) is returned. When none is safe, abstains on everything.
    """
    c = np.asarray(confidence, dtype=float)
    e = np.asarray(errors, dtype=float)
    if c.size != e.size:
        raise ValueError("confidence and errors must be the same length")

    delta_grid = delta / n_grid
    # pre-specified grid over the confidence range; test from lowest (max coverage) upward and
    # return the first safe threshold.
    grid = np.linspace(1.0 / n_grid, 1.0, n_grid)
    for t in grid:
        accepted = c >= t
        n = int(accepted.sum())
        if n == 0:
            continue
        k = int(e[accepted].sum())
        upper = clopper_pearson_upper(k, n, delta=delta_grid)
        if upper <= alpha:
            return ConformalThreshold(
                threshold=float(t),
                alpha=alpha,
                delta=delta,
                calibrated=True,
                cal_coverage=n / c.size,
                cal_error_rate=(k / n),
                cal_error_upper=upper,
            )
    return ConformalThreshold(
        threshold=float("inf"),
        alpha=alpha,
        delta=delta,
        calibrated=False,
        cal_coverage=0.0,
        cal_error_rate=0.0,
        cal_error_upper=1.0,
    )


class SelectiveVerdict(BaseModel):
    item_id: str
    accepted: bool
    confidence: float
    verdict: float | None = None  # judge score/verdict when accepted; None => routed to human


class SelectiveResult(BaseModel):
    schema_version: str = "0.1"
    threshold: ConformalThreshold
    verdicts: list[SelectiveVerdict]
    coverage: float  # fraction accepted on this set

    def summary(self) -> str:
        t = self.threshold
        head = "abstain-all (no safe threshold)" if not t.calibrated else f"conf>={t.threshold:.3f}"
        return (
            f"selective judging [{head}]: accept {self.coverage:.1%} "
            f"(guarantee: error<={t.alpha} w.p.>={1 - t.delta}); "
            f"cal error={t.cal_error_rate:.3f} (CP<= {t.cal_error_upper:.3f})"
        )


def apply_threshold(
    item_ids: Sequence[str],
    confidence: Sequence[float],
    threshold: ConformalThreshold,
    *,
    verdicts: Sequence[float] | None = None,
) -> SelectiveResult:
    """Accept items whose confidence meets the calibrated threshold; abstain on the rest."""
    out: list[SelectiveVerdict] = []
    n_accepted = 0
    for i, (iid, conf) in enumerate(zip(item_ids, confidence, strict=True)):
        accept = conf >= threshold.threshold
        n_accepted += int(accept)
        out.append(
            SelectiveVerdict(
                item_id=iid,
                accepted=accept,
                confidence=float(conf),
                verdict=(float(verdicts[i]) if (accept and verdicts is not None) else None),
            )
        )
    coverage = n_accepted / len(out) if out else 0.0
    return SelectiveResult(threshold=threshold, verdicts=out, coverage=coverage)
