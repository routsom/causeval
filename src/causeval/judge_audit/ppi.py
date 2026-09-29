"""Prediction-powered inference for the human-quality mean (SPEC 3.5, PPI/PPI++).

The judge is a cheap, biased predictor ``f(X)`` of the expensive human label ``Y``. Naively
averaging judge scores gives a tight but *wrong* CI (it estimates the judge's mean, not the
human mean). Averaging the few human labels is unbiased but wide. PPI combines them: it
debiases the judge mean using the labeled subset, keeping the unbiased target while borrowing
the judge's precision.

Estimand: ``theta = E[Y]``, the mean human score over the full dataset.

Estimator (PPI++ with power tuning ``lambda``)::

    theta_hat = lambda * mean(f(X_unlabeled)) + mean(Y_labeled - lambda * f(X_labeled))

with variance ``Var(Y - lambda f)/n + lambda^2 Var(f)/N_unlabeled`` and a normal CI. The
optimal ``lambda`` minimises that variance:

    lambda* = Cov(Y, f) / (Var(f) * (1 + n/N_unlabeled))

(``lambda = 1`` recovers classic PPI; ``lambda = 0`` recovers the human-only mean.)

Cross-checked numerically against ``ppi_py`` (PyPI ``ppi-python``) in the tests. Validity
requires the labeled subset to be a *random* sample -- enforced against a recorded
``sampling`` plan by :func:`ppi_mean`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from scipy import stats as sps

from causeval.core.errors import InsufficientDataError

# a 1-D list of floats or a numpy array of them.
Floats = Sequence[float] | npt.NDArray[np.float64]


class PPIEstimate(BaseModel):
    """A PPI estimate of the human mean, with the human-only baseline for comparison."""

    schema_version: str = "0.1"
    estimand: str = "mean human score over the full dataset"
    estimate: float
    ci_low: float
    ci_high: float
    ci_level: float = 0.95
    method: str  # e.g. "ppi++_lambda=0.83"
    lam: float
    n_labeled: int
    n_unlabeled: int
    # human-only (classic) mean over the labeled sample, for the width comparison.
    human_only_estimate: float
    human_only_ci_low: float
    human_only_ci_high: float
    # effective labeled sample size: n_labeled * Var(Y)/Var_pp -- how many human labels a
    # human-only estimator would need to match PPI's precision.
    effective_n: float

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low

    @property
    def human_only_ci_width(self) -> float:
        return self.human_only_ci_high - self.human_only_ci_low


def optimal_lambda(y_labeled: np.ndarray, f_labeled: np.ndarray, n_unlabeled: int) -> float:
    """PPI++ power-tuning ``lambda*`` = Cov(Y,f) / (Var(f) (1 + n/N)); 0 if Var(f)=0."""
    n = y_labeled.size
    var_f = float(np.var(f_labeled, ddof=1)) if n > 1 else 0.0
    if var_f == 0.0 or n_unlabeled == 0:
        return 0.0
    cov = float(np.cov(y_labeled, f_labeled, ddof=1)[0, 1])
    return cov / (var_f * (1.0 + n / n_unlabeled))


def ppi_mean_ci(
    y_labeled: Floats,
    f_labeled: Floats,
    f_unlabeled: Floats,
    *,
    level: float = 0.95,
    lam: float | None = None,
) -> PPIEstimate:
    """Compute the PPI++ estimate and CI for the human mean.

    ``y_labeled`` and ``f_labeled`` are aligned (human label and judge score on the labeled
    items); ``f_unlabeled`` is the judge score on the remaining items. Pass ``lam`` to fix the
    power-tuning weight; by default it is chosen optimally from the labeled data.
    """
    y = np.asarray(y_labeled, dtype=float)
    fl = np.asarray(f_labeled, dtype=float)
    fu = np.asarray(f_unlabeled, dtype=float)
    n = y.size
    if n < 2:
        raise InsufficientDataError("PPI needs at least 2 labeled items")
    if fl.size != n:
        raise ValueError("y_labeled and f_labeled must be the same length")
    n_unlabeled = fu.size

    lam = optimal_lambda(y, fl, n_unlabeled) if lam is None else lam

    # unlabeled term uses the unlabeled predictions; with none, it drops out and this reduces
    # to the classic labeled rectifier (still unbiased, just no borrowed precision).
    mean_fu = float(fu.mean()) if n_unlabeled > 0 else float(fl.mean())
    rectifier = y - lam * fl
    theta = lam * mean_fu + float(rectifier.mean())

    var_rectifier = float(np.var(rectifier, ddof=1)) / n
    var_unlabeled = (lam**2) * float(np.var(fu, ddof=1)) / n_unlabeled if n_unlabeled > 1 else 0.0
    se = float(np.sqrt(var_rectifier + var_unlabeled))
    z = float(sps.norm.ppf(1 - (1 - level) / 2))

    # human-only baseline over the labeled sample.
    ho_mean = float(y.mean())
    ho_se = float(np.std(y, ddof=1) / np.sqrt(n))
    var_y = float(np.var(y, ddof=1))
    var_pp = se**2
    effective_n = (var_y / var_pp) if var_pp > 0 else float("inf")

    return PPIEstimate(
        estimate=theta,
        ci_low=theta - z * se,
        ci_high=theta + z * se,
        ci_level=level,
        method=f"ppi++_lambda={lam:.3f}",
        lam=lam,
        n_labeled=n,
        n_unlabeled=n_unlabeled,
        human_only_estimate=ho_mean,
        human_only_ci_low=ho_mean - z * ho_se,
        human_only_ci_high=ho_mean + z * ho_se,
        effective_n=effective_n,
    )


def ppi_mean(
    y_labeled: Floats,
    f_labeled: Floats,
    f_unlabeled: Floats,
    *,
    labeled_ids: Sequence[str] | None = None,
    sampling_plan_ids: Sequence[str] | None = None,
    i_know_the_labels_are_random: bool = False,
    level: float = 0.95,
    lam: float | None = None,
) -> PPIEstimate:
    """PPI mean with the random-sample guard (SPEC: non-random labels break PPI).

    Refuses to run unless the labeled ids are a subset of a recorded random sampling plan,
    or ``i_know_the_labels_are_random`` is set.
    """
    if not i_know_the_labels_are_random:
        if sampling_plan_ids is None or labeled_ids is None:
            raise InsufficientDataError(
                "PPI requires a recorded random sampling plan (from judge_audit.sampling); "
                "pass sampling_plan_ids + labeled_ids, or i_know_the_labels_are_random=True"
            )
        if not set(labeled_ids).issubset(set(sampling_plan_ids)):
            raise InsufficientDataError(
                "labeled items are not a subset of the recorded random sample; PPI is invalid"
            )
    return ppi_mean_ci(y_labeled, f_labeled, f_unlabeled, level=level, lam=lam)
