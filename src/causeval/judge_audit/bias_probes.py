"""Randomized interventions that measure judge bias (SPEC 3.5).

Each probe intervenes on the input to the judge in a way that should not change a fair
verdict, then measures the effect with a cluster-bootstrap CI over items. An effect whose CI
excludes 0 and whose magnitude exceeds a configured tolerance is flagged.

Probes here:
  * :func:`position_bias` -- pairwise judge, both orders. ``P(choose first) - 0.5`` plus the
    order-consistency rate. A position-biased judge favors whichever answer is shown first.
  * :func:`paired_score_effect` -- pointwise judge under a content-preserving transform.
    ``effect = mean_i(score(transform(a)) - score(a))``. Verbosity, formatting, and authorship
    probes are this with different transforms.

Judges are injected as async callables so the probes are testable offline with a fake judge.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from causeval.stats.estimate import cluster_bootstrap_ci

# picks the first-presented answer? (question, answer_first, answer_second) -> 1.0 / 0.0
PairwiseJudge = Callable[[str, str, str], Awaitable[float]]
# pointwise score in [0, 1]: (question, answer) -> float
PointwiseJudge = Callable[[str, str], Awaitable[float]]
# content-preserving edit: (question, answer) -> transformed answer
Transform = Callable[[str, str], str]


@dataclass(frozen=True)
class PairItem:
    item_id: str
    question: str
    answer_a: str
    answer_b: str


@dataclass(frozen=True)
class ScoreItem:
    item_id: str
    question: str
    answer: str


class BiasEstimate(BaseModel):
    """One probe's effect estimate with a paired cluster-bootstrap CI."""

    schema_version: str = "0.1"
    probe: str
    estimate: float  # e.g. P(choose first) - 0.5, or mean score delta under the transform
    ci_low: float
    ci_high: float
    ci_level: float = 0.95
    method: str
    n_items: int
    n_repeats: int
    tolerance: float
    # CI excludes 0 (an effect exists) *and* |estimate| exceeds tolerance.
    flagged: bool
    # probe-specific extras (e.g. {"order_consistency_rate": 0.82}).
    extra: dict[str, float] = {}

    @property
    def ci_excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0


def _flag(estimate: float, ci_low: float, ci_high: float, tolerance: float) -> bool:
    excludes_zero = ci_low > 0.0 or ci_high < 0.0
    return excludes_zero and abs(estimate) > tolerance


async def position_bias(
    pairs: list[PairItem],
    judge: PairwiseJudge,
    *,
    repeats: int = 1,
    tolerance: float = 0.05,
    level: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> BiasEstimate:
    """Judge each pair in both orders; report ``P(choose first) - 0.5`` and consistency.

    For a pair (a, b): ``o1 = judge picks first when shown (a, b)``; ``o2 = judge picks first
    when shown (b, a)``. Both are draws of "chose the first-presented answer", so their mean
    estimates ``P(choose first)``. The pair is *order-consistent* when the judge prefers the
    same underlying answer regardless of order, i.e. ``o1 != o2``.
    """
    rng = np.random.default_rng(seed)

    async def one(pair: PairItem) -> tuple[float, float]:
        firsts: list[float] = []
        consistent: list[float] = []
        for _ in range(repeats):
            o1 = await judge(pair.question, pair.answer_a, pair.answer_b)
            o2 = await judge(pair.question, pair.answer_b, pair.answer_a)
            firsts.append((o1 + o2) / 2.0)
            consistent.append(1.0 if o1 != o2 else 0.0)
        return float(np.mean(firsts)), float(np.mean(consistent))

    results = await asyncio.gather(*(one(p) for p in pairs))
    first_rates = np.array([r[0] for r in results], dtype=float)
    consistency = np.array([r[1] for r in results], dtype=float)

    # center on 0.5 so the estimand is the bias; the CI shifts with it.
    centered = first_rates - 0.5
    lo, hi = cluster_bootstrap_ci(centered, level=level, n_boot=n_boot, rng=rng)
    estimate = float(centered.mean())
    return BiasEstimate(
        probe="position",
        estimate=estimate,
        ci_low=lo,
        ci_high=hi,
        ci_level=level,
        method=f"pairwise_both_orders_cluster_bootstrap_B={n_boot}",
        n_items=len(pairs),
        n_repeats=repeats,
        tolerance=tolerance,
        flagged=_flag(estimate, lo, hi, tolerance),
        extra={"order_consistency_rate": float(consistency.mean())},
    )


async def paired_score_effect(
    items: list[ScoreItem],
    judge: PointwiseJudge,
    transform: Transform,
    *,
    probe: str,
    repeats: int = 1,
    tolerance: float = 0.05,
    level: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> BiasEstimate:
    """Mean per-item ``score(transform(answer)) - score(answer)`` with a cluster CI.

    Used for verbosity (pad with null text), formatting (markdown vs prose), and authorship
    (add a "response written by <model>" line) by passing the corresponding ``transform``.
    """
    rng = np.random.default_rng(seed)

    async def one(item: ScoreItem) -> float:
        transformed = transform(item.question, item.answer)
        deltas: list[float] = []
        for _ in range(repeats):
            base = await judge(item.question, item.answer)
            padded = await judge(item.question, transformed)
            deltas.append(padded - base)
        return float(np.mean(deltas))

    per_item = np.array(await asyncio.gather(*(one(it) for it in items)), dtype=float)
    lo, hi = cluster_bootstrap_ci(per_item, level=level, n_boot=n_boot, rng=rng)
    estimate = float(per_item.mean())
    return BiasEstimate(
        probe=probe,
        estimate=estimate,
        ci_low=lo,
        ci_high=hi,
        ci_level=level,
        method=f"paired_transform_cluster_bootstrap_B={n_boot}",
        n_items=len(items),
        n_repeats=repeats,
        tolerance=tolerance,
        flagged=_flag(estimate, lo, hi, tolerance),
    )


# ---- default content-preserving transforms --------------------------------------


def verbose_padding(question: str, answer: str) -> str:
    """Restate the question + polite filler (semantically null; verify with NLI in live runs)."""
    return (
        f"{answer} To restate the question you asked: {question} "
        "I hope this response is helpful and thorough."
    )


def markdown_formatting(_question: str, answer: str) -> str:
    """Same content, rendered as a Markdown bullet + bold lead-in."""
    return f"- **Answer:** {answer}"


def authorship_label(model: str) -> Transform:
    """Prefix a 'response written by <model>' line (the label is the only change)."""

    def _t(_question: str, answer: str) -> str:
        return f"[Response written by {model}]\n{answer}"

    return _t
