"""Paired candidate-vs-baseline comparison.

Estimand: ``Delta = E_item[ ybar_i^cand - ybar_i^base ]``, estimated by the mean of paired
per-item differences and a paired cluster bootstrap over items. Secondary evidence: a
Wilcoxon signed-rank p-value on the item-level differences. Across the metrics in one call,
Holm-Bonferroni sets each metric's confidence level before its verdict is gated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

from causeval.core.errors import InsufficientDataError
from causeval.core.schemas import Comparison, Measurement
from causeval.stats.aggregate import group_pass, group_scores
from causeval.stats.estimate import cluster_bootstrap_ci
from causeval.stats.gate import gate_verdict, holm_adjust


@dataclass(frozen=True)
class _PairedMetric:
    metric: str
    diffs: np.ndarray  # per-item candidate-minus-baseline mean difference
    delta: float
    wilcoxon_p: float
    n_items: int
    n_dropped: int


def _wilcoxon_p(diffs: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value; 1.0 when there is no signal to test."""
    if diffs.size == 0 or np.allclose(diffs, 0.0):
        return 1.0
    try:
        return float(sps.wilcoxon(diffs).pvalue)
    except ValueError:
        return 1.0


def _matched_items(
    base: Mapping[str, object],
    cand: Mapping[str, object],
    *,
    metric: str,
    allow_partial: bool,
) -> tuple[list[str], int]:
    base_ids, cand_ids = set(base), set(cand)
    if base_ids != cand_ids and not allow_partial:
        missing = base_ids ^ cand_ids
        raise InsufficientDataError(
            f"metric {metric!r}: baseline and candidate item ids differ "
            f"({len(missing)} unmatched); pass allow_partial=True to intersect"
        )
    matched = sorted(base_ids & cand_ids)
    if not matched:
        raise InsufficientDataError(f"metric {metric!r}: no matched item ids between runs")
    n_dropped = len(base_ids ^ cand_ids)
    return matched, n_dropped


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact (binomial) McNemar p-value for discordant counts ``b`` and ``c``."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2.0 * float(sps.binom.cdf(k, n, 0.5))
    return min(p, 1.0)


def compare(
    baseline: list[Measurement],
    candidate: list[Measurement],
    *,
    metrics: list[str] | None = None,
    margin: float,
    level: float = 0.95,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
    allow_partial: bool = False,
    family_correction: str = "holm",
) -> list[Comparison]:
    """Compare ``candidate`` against ``baseline`` per metric with paired bootstrap CIs.

    ``family_correction="holm"`` (default) allocates Holm-adjusted confidence levels across
    the compared metrics: the metric with the strongest signal (smallest Wilcoxon p) gets
    the widest interval, controlling the family-wise rate of the gate's claims. Use
    ``"none"`` to gate each metric at ``level`` independently.
    """
    rng = rng if rng is not None else np.random.default_rng()

    base_metrics = {m.metric for m in baseline}
    cand_metrics = {m.metric for m in candidate}
    names = metrics if metrics is not None else sorted(base_metrics & cand_metrics)
    if not names:
        raise InsufficientDataError("no metric is present in both runs")

    paired: list[_PairedMetric] = []
    for name in names:
        base_scores, _ = group_scores(baseline, metric=name)
        cand_scores, _ = group_scores(candidate, metric=name)
        matched, n_dropped = _matched_items(
            base_scores, cand_scores, metric=name, allow_partial=allow_partial
        )
        diffs = np.array(
            [float(np.mean(cand_scores[i])) - float(np.mean(base_scores[i])) for i in matched]
        )
        paired.append(
            _PairedMetric(
                metric=name,
                diffs=diffs,
                delta=float(diffs.mean()),
                wilcoxon_p=_wilcoxon_p(diffs),
                n_items=len(matched),
                n_dropped=n_dropped,
            )
        )

    # Holm allocation of confidence levels across metrics, ordered by Wilcoxon p.
    k = len(paired)
    alpha = 1.0 - level
    raw_p = [p.wilcoxon_p for p in paired]
    p_adjusted = holm_adjust(raw_p)
    if family_correction == "holm" and k > 1:
        ranks = {i: r for r, i in enumerate(sorted(range(k), key=lambda i: raw_p[i]))}
        alphas = [alpha / (k - ranks[i]) for i in range(k)]
    elif family_correction == "none":
        alphas = [alpha] * k
        p_adjusted = raw_p
    else:
        alphas = [alpha] * k  # holm with k == 1 leaves the level unchanged

    comparisons: list[Comparison] = []
    for i, pm in enumerate(paired):
        metric_level = 1.0 - alphas[i]
        child = np.random.default_rng(rng.integers(0, 2**63 - 1))
        ci_low, ci_high = cluster_bootstrap_ci(
            pm.diffs, level=metric_level, n_boot=n_boot, rng=child
        )
        verdict = gate_verdict(ci_low, ci_high, margin)
        comparisons.append(
            Comparison(
                metric=pm.metric,
                delta=pm.delta,
                ci_low=ci_low,
                ci_high=ci_high,
                ci_level=metric_level,
                p_value=pm.wilcoxon_p,
                p_adjusted=p_adjusted[i],
                method=f"paired_cluster_bootstrap_studentized_B={n_boot}; wilcoxon; holm",
                verdict=verdict,
                margin=margin,
                n_items=pm.n_items,
                n_dropped=pm.n_dropped,
            )
        )
    return comparisons


def compare_pass_rates(
    baseline: list[Measurement],
    candidate: list[Measurement],
    *,
    metric: str,
    threshold: float,
    margin: float,
    level: float = 0.95,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
    allow_partial: bool = False,
) -> tuple[Comparison, float]:
    """Paired comparison of per-item pass rates, plus a McNemar p on majority-vote pass.

    Returns ``(comparison, mcnemar_p)``. The comparison's ``Delta`` is the mean per-item
    difference in pass rate (candidate - baseline); ``mcnemar_p`` tests whether the
    majority-vote pass/fail label flips systematically between the two runs.
    """
    rng = rng if rng is not None else np.random.default_rng()
    base_pass = group_pass(baseline, metric=metric, threshold=threshold)
    cand_pass = group_pass(candidate, metric=metric, threshold=threshold)
    matched, n_dropped = _matched_items(
        base_pass, cand_pass, metric=metric, allow_partial=allow_partial
    )

    base_rate = {i: float(np.mean(base_pass[i])) for i in matched}
    cand_rate = {i: float(np.mean(cand_pass[i])) for i in matched}
    diffs = np.array([cand_rate[i] - base_rate[i] for i in matched])

    ci_low, ci_high = cluster_bootstrap_ci(diffs, level=level, n_boot=n_boot, rng=rng)

    # McNemar on majority-vote pass labels.
    b = c = 0
    for i in matched:
        bp = base_rate[i] > 0.5
        cp = cand_rate[i] > 0.5
        if bp and not cp:
            b += 1
        elif cp and not bp:
            c += 1
    mcnemar_p = mcnemar_exact_p(b, c)

    comparison = Comparison(
        metric=f"{metric}[pass@{threshold:g}]",
        delta=float(diffs.mean()),
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=level,
        p_value=mcnemar_p,
        p_adjusted=None,
        method=f"paired_pass_rate_bootstrap_B={n_boot}; mcnemar_exact",
        verdict=gate_verdict(ci_low, ci_high, margin),
        margin=margin,
        n_items=len(matched),
        n_dropped=n_dropped,
    )
    return comparison, mcnemar_p
