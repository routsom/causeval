"""Calibrate a judge against human labels (SPEC 3.5).

Given paired ``(judge_score, human_label)`` on a labeled set, report how well the judge tracks
humans and produce a monotone map from judge score to ``P(human pass)`` for use by PPI and
gates.

  * agreement: Spearman correlation, Cohen's kappa (binarized at ``threshold``), 2x2 confusion;
  * :class:`IsotonicMap`: isotonic regression (pool-adjacent-violators) of the binary human
    pass indicator on the judge score -- a non-decreasing calibration curve, no sklearn needed;
  * expected calibration error (ECE) of that map, with a bootstrap CI.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy import stats as sps

from causeval.core.errors import InsufficientDataError


def _pava(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators: fit a non-decreasing step function of ``y`` on sorted ``x``.

    Returns ``(x_sorted, y_fitted)`` where ``y_fitted`` is the isotonic regression of ``y``
    against ``x`` (ties in ``x`` are averaged first). Weights are equal (one per point).
    """
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ys = y[order].astype(float)

    # blocks of (weighted) means, merged while they violate monotonicity.
    values: list[float] = []
    weights: list[float] = []
    for yi in ys:
        values.append(float(yi))
        weights.append(1.0)
        while len(values) > 1 and values[-2] > values[-1]:
            w = weights[-1] + weights[-2]
            v = (values[-1] * weights[-1] + values[-2] * weights[-2]) / w
            values.pop()
            weights.pop()
            values[-1] = v
            weights[-1] = w
    # expand block values back to per-point fitted values.
    fitted = np.empty_like(ys)
    i = 0
    for v, w in zip(values, weights, strict=True):
        fitted[i : i + int(w)] = v
        i += int(w)
    return xs, fitted


class IsotonicMap(BaseModel):
    """A fitted non-decreasing map ``judge score -> P(human pass)`` (step interpolation)."""

    schema_version: str = "0.1"
    x: list[float]  # sorted judge scores (knots)
    y: list[float]  # fitted P(pass), non-decreasing

    def predict(self, score: float) -> float:
        """P(human pass) at ``score`` via clamped step interpolation on the knots."""
        xs = np.asarray(self.x)
        ys = np.asarray(self.y)
        if xs.size == 0:
            return float("nan")
        return float(np.interp(score, xs, ys, left=ys[0], right=ys[-1]))


class CalibrationReport(BaseModel):
    schema_version: str = "0.1"
    n_items: int
    threshold: float
    spearman: float
    cohen_kappa: float
    # confusion at threshold, judge (rows) vs human (cols): [[tn, fp], [fn, tp]]
    confusion: list[list[int]]
    ece: float
    ece_ci_low: float
    ece_ci_high: float
    isotonic: IsotonicMap

    def summary(self) -> str:
        return (
            f"calibration (n={self.n_items}, thr={self.threshold}): "
            f"spearman={self.spearman:.3f} kappa={self.cohen_kappa:.3f} "
            f"ECE={self.ece:.3f} [{self.ece_ci_low:.3f}, {self.ece_ci_high:.3f}]"
        )


def cohen_kappa(judge_pass: np.ndarray, human_pass: np.ndarray) -> float:
    """Cohen's kappa for two binary raters (chance-corrected agreement)."""
    n = judge_pass.size
    if n == 0:
        return float("nan")
    po = float(np.mean(judge_pass == human_pass))
    pj1 = float(np.mean(judge_pass))
    ph1 = float(np.mean(human_pass))
    pe = pj1 * ph1 + (1 - pj1) * (1 - ph1)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def expected_calibration_error(
    fitted_p: np.ndarray, human_pass: np.ndarray, *, n_bins: int = 10
) -> float:
    """Binned ECE: mean over bins of |mean predicted P - empirical pass rate|, size-weighted."""
    if fitted_p.size == 0:
        return float("nan")
    bins = np.clip((fitted_p * n_bins).astype(int), 0, n_bins - 1)
    total = fitted_p.size
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        m = int(mask.sum())
        if m == 0:
            continue
        conf = float(fitted_p[mask].mean())
        acc = float(human_pass[mask].mean())
        ece += (m / total) * abs(conf - acc)
    return ece


def calibrate(
    judge_scores: np.ndarray | list[float],
    human_labels: np.ndarray | list[float],
    *,
    threshold: float = 0.5,
    n_bins: int = 10,
    n_boot: int = 1000,
    level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> CalibrationReport:
    """Fit the calibration map and report agreement + ECE with a bootstrap CI.

    ``human_labels`` may be continuous or binary; they are binarized at ``threshold`` for
    kappa/confusion and for the isotonic target (``human >= threshold`` = pass).
    """
    rng = rng if rng is not None else np.random.default_rng()
    js = np.asarray(judge_scores, dtype=float)
    hl = np.asarray(human_labels, dtype=float)
    if js.size != hl.size:
        raise ValueError("judge_scores and human_labels must be the same length")
    if js.size < 2:
        raise InsufficientDataError("calibration needs at least 2 labeled items")

    human_pass = (hl >= threshold).astype(int)
    judge_pass = (js >= threshold).astype(int)

    spearman = float(sps.spearmanr(js, hl).statistic) if np.ptp(js) > 0 else 0.0
    kappa = cohen_kappa(judge_pass, human_pass)

    tn = int(np.sum((judge_pass == 0) & (human_pass == 0)))
    fp = int(np.sum((judge_pass == 1) & (human_pass == 0)))
    fn = int(np.sum((judge_pass == 0) & (human_pass == 1)))
    tp = int(np.sum((judge_pass == 1) & (human_pass == 1)))

    xs, fitted = _pava(js, human_pass.astype(float))
    iso = IsotonicMap(x=xs.tolist(), y=fitted.tolist())
    fitted_p = np.array([iso.predict(s) for s in js])
    ece = expected_calibration_error(fitted_p, human_pass, n_bins=n_bins)

    # bootstrap the ECE over items (refit the map each replicate).
    boot_ece = np.empty(n_boot)
    n = js.size
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bx, bfit = _pava(js[idx], human_pass[idx].astype(float))
        bmap = IsotonicMap(x=bx.tolist(), y=bfit.tolist())
        bp = np.array([bmap.predict(s) for s in js[idx]])
        boot_ece[b] = expected_calibration_error(bp, human_pass[idx], n_bins=n_bins)
    a = (1 - level) / 2
    lo, hi = float(np.quantile(boot_ece, a)), float(np.quantile(boot_ece, 1 - a))

    return CalibrationReport(
        n_items=int(n),
        threshold=threshold,
        spearman=spearman,
        cohen_kappa=kappa,
        confusion=[[tn, fp], [fn, tp]],
        ece=ece,
        ece_ci_low=lo,
        ece_ci_high=hi,
        isotonic=iso,
    )
