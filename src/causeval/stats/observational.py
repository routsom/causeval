"""Observational causal estimation from production logs (SPEC 3.8, Phase 7, optional).

Sometimes you can't A/B test two versions; you only have logs where the version that served
each request was chosen non-randomly (by traffic rules, cohorts, time). Naively differencing
the metric across versions then confounds the version effect with whoever got each version.

This module estimates the average effect of the version on a metric with **AIPW** (augmented
inverse-propensity weighting), a doubly-robust estimator: it is consistent if *either* the
propensity model ``e(x) = P(version=1 | x)`` *or* the outcome models ``mu_t(x) = E[Y | x, t]``
are right. Nuisances are fit by K-fold cross-fitting so the estimator stays root-n valid
without assuming the models are perfect.

It always runs DoWhy-style refuters (placebo treatment, random common cause, data subset) and
carries a fixed warning: no observational method can rule out *unobserved* confounding.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy import stats as sps
from scipy.special import expit

UNCONFOUNDEDNESS_WARNING = (
    "Observational estimate: assumes no unobserved confounding, which cannot be verified from "
    "data. Refuters check robustness, not identification. Prefer a randomized A/B test."
)


class RefuterResult(BaseModel):
    name: str
    estimate: float
    passed: bool  # did the refutation behave as a valid estimator should?
    detail: str


class ObservationalEffect(BaseModel):
    schema_version: str = "0.1"
    estimand: str = "average treatment effect of version on the metric (ATE)"
    ate: float
    ci_low: float
    ci_high: float
    ci_level: float = 0.95
    method: str
    n: int
    n_treated: int
    naive_diff: float  # mean(Y|T=1) - mean(Y|T=0), the confounded comparison
    refuters: list[RefuterResult]
    warning: str = UNCONFOUNDEDNESS_WARNING

    def summary(self) -> str:
        lines = [
            f"Observational effect (AIPW): ATE={self.ate:+.4f} "
            f"[{int(self.ci_level * 100)}% CI {self.ci_low:+.4f}, {self.ci_high:+.4f}] "
            f"(n={self.n}, treated={self.n_treated})",
            f"  naive difference-in-means (confounded): {self.naive_diff:+.4f}",
        ]
        for r in self.refuters:
            lines.append(f"  refuter {r.name}: {r.estimate:+.4f} -> {'ok' if r.passed else 'FLAG'}")
        lines.append(f"  ! {self.warning}")
        return "\n".join(lines)


def _with_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def _fit_logistic(
    x: np.ndarray, y: np.ndarray, *, ridge: float = 1e-3, iters: int = 50
) -> np.ndarray:
    """Ridge-penalised logistic regression by IRLS/Newton. ``x`` includes an intercept column."""
    d = x.shape[1]
    w = np.zeros(d)
    for _ in range(iters):
        p = expit(x @ w)
        wgt = np.clip(p * (1 - p), 1e-6, None)
        grad = x.T @ (p - y) + ridge * w
        hess = (x * wgt[:, None]).T @ x + ridge * np.eye(d)
        step = np.linalg.solve(hess, grad)
        w = w - step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w


def _ridge_ols(x: np.ndarray, y: np.ndarray, *, ridge: float = 1e-3) -> np.ndarray:
    """Ridge least squares. ``x`` includes an intercept column (not penalised heavily here)."""
    d = x.shape[1]
    return np.linalg.solve(x.T @ x + ridge * np.eye(d), x.T @ y)


def _crossfit_nuisances(
    t: np.ndarray, y: np.ndarray, x: np.ndarray, *, n_folds: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-fold predictions of e(x), mu0(x), mu1(x) by K-fold cross-fitting."""
    n = len(t)
    xb = _with_intercept(x)
    folds = np.array_split(rng.permutation(n), n_folds)
    e = np.zeros(n)
    mu0 = np.zeros(n)
    mu1 = np.zeros(n)
    for fold in folds:
        mask = np.ones(n, dtype=bool)
        mask[fold] = False  # training complement
        w = _fit_logistic(xb[mask], t[mask])
        e[fold] = np.clip(expit(xb[fold] @ w), 0.02, 0.98)
        tr0 = mask & (t == 0)
        tr1 = mask & (t == 1)
        b0 = _ridge_ols(xb[tr0], y[tr0]) if tr0.sum() > xb.shape[1] else np.zeros(xb.shape[1])
        b1 = _ridge_ols(xb[tr1], y[tr1]) if tr1.sum() > xb.shape[1] else np.zeros(xb.shape[1])
        mu0[fold] = xb[fold] @ b0
        mu1[fold] = xb[fold] @ b1
    return e, mu0, mu1


def _aipw_scores(
    t: np.ndarray, y: np.ndarray, e: np.ndarray, mu0: np.ndarray, mu1: np.ndarray
) -> np.ndarray:
    """Per-unit AIPW influence scores; their mean is the doubly-robust ATE."""
    scores = (mu1 - mu0) + t / e * (y - mu1) - (1 - t) / (1 - e) * (y - mu0)
    return np.asarray(scores, dtype=float)


def aipw_ate(
    treatment: np.ndarray | list[float],
    outcome: np.ndarray | list[float],
    covariates: np.ndarray | list[list[float]],
    *,
    n_folds: int = 2,
    level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float, np.ndarray]:
    """Cross-fit AIPW ATE. Returns (ate, ci_low, ci_high, per-unit scores)."""
    rng = rng if rng is not None else np.random.default_rng()
    t = np.asarray(treatment, dtype=float)
    y = np.asarray(outcome, dtype=float)
    x = np.atleast_2d(np.asarray(covariates, dtype=float))
    if x.shape[0] != len(t):
        x = x.T  # accept (features, n) too
    e, mu0, mu1 = _crossfit_nuisances(t, y, x, n_folds=n_folds, rng=rng)
    psi = _aipw_scores(t, y, e, mu0, mu1)
    ate = float(psi.mean())
    se = float(psi.std(ddof=1) / np.sqrt(len(psi)))
    z = float(sps.norm.ppf(1 - (1 - level) / 2))
    return ate, ate - z * se, ate + z * se, psi


def estimate_effect(
    treatment: np.ndarray | list[float],
    outcome: np.ndarray | list[float],
    covariates: np.ndarray | list[list[float]],
    *,
    n_folds: int = 2,
    level: float = 0.95,
    seed: int = 0,
) -> ObservationalEffect:
    """AIPW ATE plus DoWhy-style refuters (placebo, random common cause, data subset)."""
    rng = np.random.default_rng(seed)
    t = np.asarray(treatment, dtype=float)
    y = np.asarray(outcome, dtype=float)
    x = np.atleast_2d(np.asarray(covariates, dtype=float))
    if x.shape[0] != len(t):
        x = x.T
    n = len(t)

    ate, lo, hi, _ = aipw_ate(t, y, x, n_folds=n_folds, level=level, rng=rng)
    half_width = max(hi - ate, 1e-6)

    # placebo: shuffle treatment -> a valid estimator should find ~0.
    t_placebo = rng.permutation(t)
    ate_placebo, plo, phi, _ = aipw_ate(t_placebo, y, x, n_folds=n_folds, level=level, rng=rng)
    placebo = RefuterResult(
        name="placebo_treatment",
        estimate=ate_placebo,
        passed=bool(plo <= 0.0 <= phi),
        detail="ATE on permuted treatment should have a CI covering 0",
    )

    # random common cause: add a noise covariate -> the estimate should barely move.
    x_rcc = np.column_stack([x, rng.normal(size=n)])
    ate_rcc, *_ = aipw_ate(t, y, x_rcc, n_folds=n_folds, level=level, rng=rng)
    rcc = RefuterResult(
        name="random_common_cause",
        estimate=ate_rcc,
        passed=bool(abs(ate_rcc - ate) <= half_width),
        detail="adding an irrelevant covariate should not change the ATE",
    )

    # data subset: re-estimate on 70% -> should be stable.
    idx = rng.choice(n, size=int(0.7 * n), replace=False)
    ate_sub, *_ = aipw_ate(t[idx], y[idx], x[idx], n_folds=n_folds, level=level, rng=rng)
    subset = RefuterResult(
        name="data_subset",
        estimate=ate_sub,
        passed=bool(abs(ate_sub - ate) <= 2 * half_width),
        detail="estimate on a random 70% subset should be stable",
    )

    naive = float(y[t == 1].mean() - y[t == 0].mean())
    return ObservationalEffect(
        ate=ate,
        ci_low=lo,
        ci_high=hi,
        ci_level=level,
        method=f"aipw_crossfit_folds={n_folds}",
        n=n,
        n_treated=int(t.sum()),
        naive_diff=naive,
        refuters=[placebo, rcc, subset],
    )
