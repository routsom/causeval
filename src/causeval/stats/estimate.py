"""Point estimate and uncertainty for a metric under one condition.

Estimand (default): ``mu = E_item[ E_repeat[ y ] ]`` -- the expected score of a randomly
drawn item under a random judge/app sample. Estimator: ``mu_hat = mean_i(mean_r y_ir)``.

Uncertainty is a cluster bootstrap over items (each item's repeats travel together), which
targets the variance of the mean of item means, ``(sigma^2_between + sigma^2_within/R)/n``.
Variance components come from a one-way random-effects ANOVA (method of moments).
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

from causeval.core.errors import InsufficientDataError
from causeval.core.schemas import Measurement, ScoreEstimate
from causeval.stats.aggregate import group_pass, group_scores


def cluster_bootstrap_ci(
    item_values: np.ndarray,
    *,
    level: float = 0.95,
    n_boot: int = 2000,
    rng: np.random.Generator,
    method: str = "studentized",
) -> tuple[float, float]:
    """Bootstrap CI for the mean over a resample of items (clusters).

    ``item_values`` is one value per item (e.g. per-item means); items are the resampling
    unit, so a caller that keeps each item's repeats together gets a clustered bootstrap.

    Methods:
      * ``"studentized"`` (default) -- bootstrap-t. Second-order accurate for the mean and
        the only method here that meets B1's [0.93, 0.97] coverage at n=20 (percentile and
        BCa undercover a symmetric statistic like the mean at small n). See PROGRESS.md.
      * ``"percentile"`` -- plain percentile interval.
      * ``"bca"`` -- bias-corrected-and-accelerated percentiles.
    """
    n = item_values.size
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        v = float(item_values[0])
        return (v, v)
    alpha = 1.0 - level
    idx = rng.integers(0, n, size=(n_boot, n))
    resamples = item_values[idx]
    boot = resamples.mean(axis=1)

    if method == "percentile":
        return (float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2)))
    if method == "bca":
        return _bca_interval(item_values, boot, alpha=alpha)
    if method == "studentized":
        return _studentized_interval(item_values, resamples, boot, alpha=alpha)
    raise ValueError(f"unknown CI method {method!r}")


def _studentized_interval(
    item_values: np.ndarray, resamples: np.ndarray, boot: np.ndarray, *, alpha: float
) -> tuple[float, float]:
    """Bootstrap-t interval for the mean (Efron & Tibshirani ch. 12)."""
    theta_hat = float(item_values.mean())
    n = item_values.size
    se = float(item_values.std(ddof=1)) / np.sqrt(n)
    if se == 0:
        return (theta_hat, theta_hat)
    boot_se = resamples.std(axis=1, ddof=1) / np.sqrt(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_star = np.where(boot_se > 0, (boot - theta_hat) / boot_se, np.nan)
    q_lo = float(np.nanquantile(t_star, alpha / 2))
    q_hi = float(np.nanquantile(t_star, 1 - alpha / 2))
    # Note the flip: the upper t-quantile maps to the lower confidence bound.
    return (theta_hat - q_hi * se, theta_hat - q_lo * se)


def _bca_interval(
    item_values: np.ndarray, boot: np.ndarray, *, alpha: float
) -> tuple[float, float]:
    """BCa interval for the mean (Efron). Jackknife acceleration uses leave-one-out means."""
    theta_hat = float(item_values.mean())
    n = item_values.size

    # Bias-correction z0 from the fraction of bootstrap replicates below the estimate.
    prop = float(np.mean(boot < theta_hat))
    prop = min(max(prop, 1.0 / (2 * boot.size)), 1.0 - 1.0 / (2 * boot.size))
    z0 = float(sps.norm.ppf(prop))

    # Acceleration via jackknife leave-one-out means.
    total = item_values.sum()
    jack = (total - item_values) / (n - 1)
    jack_mean = jack.mean()
    diff = jack_mean - jack
    denom = 6.0 * (float(np.sum(diff**2)) ** 1.5)
    a = float(np.sum(diff**3)) / denom if denom != 0 else 0.0

    z_lo = sps.norm.ppf(alpha / 2)
    z_hi = sps.norm.ppf(1 - alpha / 2)

    def adjusted(z: float) -> float:
        p = sps.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        return float(np.clip(p, 0.0, 1.0))

    lo = float(np.quantile(boot, adjusted(z_lo)))
    hi = float(np.quantile(boot, adjusted(z_hi)))
    return (lo, hi)


def variance_components(scores_by_item: dict[str, list[float]]) -> tuple[float, float, float]:
    """One-way random-effects ANOVA (method of moments). Returns (between, within, icc).

    Negative variance components are truncated at 0. With a single repeat per item
    (``R == 1``) within-item noise is unidentifiable, so ``within`` is reported as 0 and all
    variance is attributed to ``between``.
    """
    groups = [np.asarray(v, dtype=float) for v in scores_by_item.values() if len(v) > 0]
    k = len(groups)
    if k == 0:
        return (0.0, 0.0, 0.0)
    n_i = np.array([g.size for g in groups])
    total_n = int(n_i.sum())
    group_means = np.array([g.mean() for g in groups])
    grand_mean = float(np.concatenate(groups).mean())

    ss_between = float(np.sum(n_i * (group_means - grand_mean) ** 2))
    ss_within = float(sum(float(np.sum((g - g.mean()) ** 2)) for g in groups))

    if k == 1:
        # No between-item information; report only within.
        within = ss_within / (total_n - 1) if total_n > 1 else 0.0
        return (0.0, float(within), 0.0)

    ms_between = ss_between / (k - 1)
    df_within = total_n - k
    within = ss_within / df_within if df_within > 0 else 0.0
    # n0: the effective group size (== R when balanced).
    n0 = (total_n - float(np.sum(n_i**2)) / total_n) / (k - 1)
    n0 = max(n0, 1.0)
    between = (ms_between - within) / n0
    between = max(between, 0.0)
    denom = between + within
    icc = between / denom if denom > 0 else 0.0
    return (float(between), float(within), float(icc))


def estimate(
    measurements: list[Measurement],
    *,
    metric: str | None = None,
    condition: str | None = None,
    level: float = 0.95,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
    method: str = "studentized",
    threshold: float | None = None,
    flaky_low: float = 0.2,
    flaky_high: float = 0.8,
) -> ScoreEstimate:
    """Estimate ``mu`` with a cluster-bootstrap CI and variance decomposition.

    When fewer than 20 items are available, a t-interval on the item means is also reported
    (``t_ci_low``/``t_ci_high``) and ``small_sample`` is set.
    """
    rng = rng if rng is not None else np.random.default_rng()
    scores_by_item, n_failed = group_scores(measurements, metric=metric, condition=condition)
    if not scores_by_item:
        raise InsufficientDataError("no successful measurements to estimate from")

    metric_name = metric or measurements[0].metric
    cond = condition or "base"

    item_means = np.array([float(np.mean(v)) for v in scores_by_item.values()])
    n_items = item_means.size
    n_repeats = max(len(v) for v in scores_by_item.values())
    mu_hat = float(item_means.mean())

    ci_low, ci_high = cluster_bootstrap_ci(
        item_means, level=level, n_boot=n_boot, rng=rng, method=method
    )
    method_str = f"cluster_bootstrap_{method}_B={n_boot}"

    between, within, icc = variance_components(scores_by_item)

    passes_by_item = group_pass(
        measurements, metric=metric, condition=condition, threshold=threshold
    )
    flaky = sorted(
        item_id
        for item_id, v in passes_by_item.items()
        if v and flaky_low < (sum(v) / len(v)) < flaky_high
    )

    small = n_items < 20
    t_lo: float | None = None
    t_hi: float | None = None
    if small and n_items >= 2:
        sd = float(item_means.std(ddof=1))
        se = sd / np.sqrt(n_items)
        t_crit = float(sps.t.ppf(1 - (1 - level) / 2, df=n_items - 1))
        t_lo = mu_hat - t_crit * se
        t_hi = mu_hat + t_crit * se

    return ScoreEstimate(
        metric=metric_name,
        condition=cond,
        estimate=mu_hat,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=level,
        method=method_str,
        n_items=n_items,
        n_repeats=n_repeats,
        n_failed=n_failed,
        var_between=between,
        var_within=within,
        icc=icc,
        flaky_items=flaky,
        small_sample=small,
        t_ci_low=t_lo,
        t_ci_high=t_hi,
    )
