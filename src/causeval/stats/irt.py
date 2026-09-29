"""2PL item-response theory for evaluation sets (SPEC 3.3 irt, Phase 6).

Fits a two-parameter logistic model to a binary ``item x system`` matrix::

    P(system j solves item i) = sigmoid(a_i * (theta_j - b_i))

where ``a_i`` is item discrimination, ``b_i`` item difficulty, and ``theta_j`` system ability.
The point of fitting it is to *prune*: keep the items that carry the most information about
where the systems of interest sit, so a much smaller set ranks systems the same way.

The fit is a native joint MAP by L-BFGS (no dependency required), with an ``N(0, sd)`` prior on
the abilities to pin the scale/location the likelihood leaves free; ``girth`` is an optional
backend used only to cross-check the fit on simulated data (extra ``causeval[irt]``). Abilities
are standardised (mean 0, sd 1) at the end, with ``a``/``b`` rescaled to preserve the fitted
probabilities.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy.optimize import minimize
from scipy.special import expit

_EPS = 1e-9


class IRTModel(BaseModel):
    """Fitted 2PL parameters. ``discrimination``/``difficulty`` per item, ``ability`` per system."""

    schema_version: str = "0.1"
    discrimination: list[float]  # a_i, per item
    difficulty: list[float]  # b_i, per item
    ability: list[float]  # theta_j, per system
    n_iter: int
    converged: bool

    def prob(self, item: int, ability: float) -> float:
        a = self.discrimination[item]
        b = self.difficulty[item]
        return float(expit(a * (ability - b)))


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def _unpack(params: np.ndarray, n_items: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = params[:n_items]
    b = params[n_items : 2 * n_items]
    theta = params[2 * n_items :]
    return a, b, theta


def _neg_log_posterior(
    params: np.ndarray, x: np.ndarray, n_items: int, theta_prior_sd: float
) -> tuple[float, np.ndarray]:
    """Joint 2PL NLL with an N(0, sd) ability prior, plus its gradient (for L-BFGS)."""
    a, b, theta = _unpack(params, n_items)
    z = a[:, None] * (theta[None, :] - b[:, None])
    p = np.clip(expit(z), _EPS, 1 - _EPS)
    r = x - p  # residual, (n_items, n_systems)

    nll = -float(np.sum(x * np.log(p) + (1 - x) * np.log(1 - p)))
    # ability prior anchors the scale/location that the likelihood alone leaves free.
    prior = 0.5 * float(np.sum((theta / theta_prior_sd) ** 2))

    g_a = -np.sum(r * (theta[None, :] - b[:, None]), axis=1)
    g_b = a * np.sum(r, axis=1)
    g_theta = -np.sum(r * a[:, None], axis=0) + theta / theta_prior_sd**2
    grad = np.concatenate([g_a, g_b, g_theta])
    return nll + prior, grad


def fit_2pl(matrix: np.ndarray, *, max_iter: int = 500, theta_prior_sd: float = 1.0) -> IRTModel:
    """Joint MAP 2PL fit of a binary ``(n_items, n_systems)`` matrix via L-BFGS.

    An ``N(0, theta_prior_sd)`` prior on the abilities regularises the fit and pins the metric
    that the likelihood leaves free (a joint scale/location shift of ``theta`` with compensating
    ``a``/``b``). Abilities are standardised at the end for a canonical, comparable scale.
    """
    x = np.asarray(matrix, dtype=float)
    n_items, n_systems = x.shape
    if n_systems < 5:
        raise ValueError("2PL IRT needs at least 5 systems")

    a0 = np.ones(n_items)
    b0 = np.array([-_logit(np.clip(row.mean(), 0.02, 0.98)) for row in x])
    theta0 = _standardize(np.array([_logit(np.clip(col.mean(), 0.02, 0.98)) for col in x.T]))
    x0 = np.concatenate([a0, b0, theta0])
    bounds = [(0.05, 6.0)] * n_items + [(-6.0, 6.0)] * n_items + [(-6.0, 6.0)] * n_systems

    res = minimize(
        _neg_log_posterior,
        x0,
        args=(x, n_items, theta_prior_sd),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": max_iter},
    )
    a, b, theta = _unpack(res.x, n_items)

    # standardise abilities to a canonical scale, rescaling a/b to preserve a*(theta - b).
    mu, sd = float(theta.mean()), float(theta.std())
    if sd > _EPS:
        theta = (theta - mu) / sd
        a = a * sd
        b = (b - mu) / sd

    return IRTModel(
        discrimination=a.tolist(),
        difficulty=b.tolist(),
        ability=theta.tolist(),
        n_iter=int(res.nit),
        converged=bool(res.success),
    )


def _standardize(v: np.ndarray) -> np.ndarray:
    sd = float(v.std())
    return (v - v.mean()) / sd if sd > _EPS else v - v.mean()


def fisher_information(model: IRTModel, item: int, abilities: np.ndarray) -> float:
    """Total 2PL Fisher information of ``item`` over ``abilities``: sum a^2 p (1-p)."""
    a = model.discrimination[item]
    b = model.difficulty[item]
    p = expit(a * (abilities - b))
    return float(np.sum(a**2 * p * (1 - p)))


def prune(model: IRTModel, target_size: int, *, abilities: np.ndarray | None = None) -> list[int]:
    """Select ``target_size`` item indices maximising Fisher information over ``abilities``.

    ``abilities`` defaults to the fitted system abilities (the systems of interest). Items are
    independent contributors to total information, so the optimum is the top-``target_size`` by
    per-item information.
    """
    thetas = np.asarray(model.ability) if abilities is None else np.asarray(abilities)
    n_items = len(model.discrimination)
    target_size = max(1, min(target_size, n_items))
    info = np.array([fisher_information(model, i, thetas) for i in range(n_items)])
    return sorted(np.argsort(info)[::-1][:target_size].tolist())


def rank_systems(matrix: np.ndarray, items: list[int] | None = None) -> np.ndarray:
    """Rank systems by mean score over ``items`` (or all). Returns each system's rank index."""
    x = np.asarray(matrix, dtype=float)
    sub = x if items is None else x[items, :]
    scores = sub.mean(axis=0)
    # argsort of -scores gives ranking; convert to per-system rank position.
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(len(scores))
    return ranks
