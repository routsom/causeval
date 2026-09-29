"""RAG causal grounding (SPEC 3.4.1) -- the flagship causal metric.

DeepEval's Faithfulness asks whether an answer is *entailed* by the retrieved context. It
cannot tell whether the answer actually *depends* on that context: a model answering from
parametric memory can look perfectly faithful. This module intervenes on the context and
measures whether the answer changes.

Conditions per item (each run with R repeats):
  * ``full``   -- all retrieved chunks (baseline)
  * ``none``   -- no context (what the model knows without retrieval)
  * ``loo_k``  -- all chunks except chunk k (per-chunk attribution)
  * ``cf``     -- the supporting chunk with one key fact edited to a false value

Metrics (each aggregated over items with a cluster-bootstrap CI):
  * Context Reliance     ``CR = P(correct|full) - P(correct|none)``
  * Counterfactual Adherence ``CA = P(follows_cf|cf)``  (the primary grounding signal)
  * Chunk effect         ``drop_k = P(correct|full) - P(correct|loo_k)``

Each item is classified grounded / parametric / confabulating / mixed, and flagged
``false_faithful`` when DeepEval Faithfulness passes yet the item is parametric.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, Field

from causeval.adapters.deepeval_import import LLMTestCase
from causeval.adapters.metric_factory import MetricSpec, a_measure_once
from causeval.core.provenance import build_provenance
from causeval.core.schemas import Measurement, Provenance, ScoreEstimate
from causeval.interventions.cf_gen import Counterfactual
from causeval.stats.estimate import cluster_bootstrap_ci

Classification = Literal["grounded", "parametric", "confabulating", "mixed"]
ConflictPolicy = Literal["follow_context", "flag_conflict"]

CorrectFn = Callable[[str, str | None], float]  # (answer, expected) -> [0, 1]
FollowsCfFn = Callable[[str, Counterfactual, ConflictPolicy], float]  # (answer, cf, policy)


class RAGApp(Protocol):
    """User app that bypasses its own retriever so causeval can inject contexts."""

    async def answer(self, question: str, contexts: list[str]) -> str: ...


# ---- outcome functions ----------------------------------------------------------


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def contains_outcome(answer: str, expected: str | None) -> float:
    """Deterministic default ``correct``: 1.0 if the expected value appears in the answer."""
    if expected is None:
        return float("nan")
    return 1.0 if _normalize(expected) in _normalize(answer) else 0.0


def follows_cf_outcome(
    answer: str, cf: Counterfactual, policy: ConflictPolicy = "follow_context"
) -> float:
    """Deterministic default ``follows_cf``.

    ``follow_context``: 1.0 if the answer states the counterfactual (false) value.
    ``flag_conflict``: 1.0 if the answer surfaces the conflict (mentions both values).
    """
    ans = _normalize(answer)
    has_cf = _normalize(cf.cf_value) in ans
    has_orig = _normalize(cf.original_value) in ans
    if policy == "flag_conflict":
        return 1.0 if (has_cf and has_orig) else 0.0
    return 1.0 if has_cf else 0.0


# ---- inputs and results ---------------------------------------------------------


@dataclass(frozen=True)
class GroundingItem:
    item_id: str
    question: str
    contexts: list[str]
    expected_output: str | None = None
    cf: Counterfactual | None = None


class ItemGrounding(BaseModel):
    item_id: str
    correct_full: float
    correct_none: float
    context_reliance: float  # correct_full - correct_none
    counterfactual_adherence: float | None  # None when no valid cf
    drop_by_chunk: dict[int, float] = Field(default_factory=dict)
    classification: Classification
    faithfulness_score: float | None = None
    false_faithful: bool = False


class GroundingResult(BaseModel):
    schema_version: str = "0.1"
    provenance: Provenance
    tau: float
    conflict_policy: ConflictPolicy
    estimates: list[ScoreEstimate]
    items: list[ItemGrounding]
    measurements: list[Measurement]
    class_counts: dict[str, int]
    n_cf_skipped: int
    false_faithful_items: list[str]

    def summary(self) -> str:
        lines = [f"RAG grounding (tau={self.tau}, policy={self.conflict_policy})"]
        for e in self.estimates:
            lines.append(
                f"  {e.metric}: {e.estimate:+.3f} "
                f"[{int(e.ci_level * 100)}% CI {e.ci_low:+.3f}, {e.ci_high:+.3f}] "
                f"(n={e.n_items}, R={e.n_repeats})"
            )
        counts = ", ".join(f"{k}={v}" for k, v in sorted(self.class_counts.items()))
        lines.append(f"  classes: {counts}")
        if self.false_faithful_items:
            lines.append(
                f"  FALSE-FAITHFUL ({len(self.false_faithful_items)}): "
                f"faithful yet parametric -> {self.false_faithful_items}"
            )
        if self.n_cf_skipped:
            lines.append(f"  ({self.n_cf_skipped} items had no valid counterfactual)")
        return "\n".join(lines)


def classify_item(
    *, ca: float | None, correct_full: float, correct_none: float, tau: float
) -> Classification:
    """grounded (CA>=tau) | parametric (correct w/o context) | confabulating | mixed."""
    if ca is not None and ca >= tau:
        return "grounded"
    if correct_none >= 0.5:
        return "parametric"
    if correct_full < 0.5:
        return "confabulating"
    return "mixed"


# ---- condition planning ---------------------------------------------------------


def _conditions_for(item: GroundingItem) -> dict[str, list[str]]:
    conditions: dict[str, list[str]] = {"full": list(item.contexts), "none": []}
    if len(item.contexts) >= 2:
        for k in range(len(item.contexts)):
            conditions[f"loo_{k}"] = [c for j, c in enumerate(item.contexts) if j != k]
    if item.cf is not None and item.cf.valid:
        cf_contexts = list(item.contexts)
        if 0 <= item.cf.chunk_index < len(cf_contexts):
            cf_contexts[item.cf.chunk_index] = item.cf.edited_chunk
            conditions["cf"] = cf_contexts
    return conditions


def _rate(
    measurements: list[Measurement], item_id: str, metric: str, condition: str
) -> float | None:
    vals = [
        m.score
        for m in measurements
        if m.item_id == item_id
        and m.metric == metric
        and m.condition == condition
        and m.score is not None
    ]
    return float(np.mean(vals)) if vals else None


def _aggregate(
    values: list[float],
    *,
    metric: str,
    n_repeats: int,
    level: float,
    n_boot: int,
    rng: np.random.Generator,
) -> ScoreEstimate:
    arr = np.array(values, dtype=float)
    lo, hi = cluster_bootstrap_ci(arr, level=level, n_boot=n_boot, rng=rng)
    return ScoreEstimate(
        metric=metric,
        condition="grounding",
        estimate=float(arr.mean()),
        ci_low=lo,
        ci_high=hi,
        ci_level=level,
        method=f"cluster_bootstrap_studentized_B={n_boot}",
        n_items=arr.size,
        n_repeats=n_repeats,
    )


async def a_ground(
    app: RAGApp,
    dataset: list[GroundingItem],
    *,
    repeats: int = 5,
    seed: int = 0,
    correct_fn: CorrectFn = contains_outcome,
    follows_cf_fn: FollowsCfFn = follows_cf_outcome,
    tau: float = 0.5,
    conflict_policy: ConflictPolicy = "follow_context",
    faithfulness_spec: MetricSpec | None = None,
    faithfulness_threshold: float = 0.5,
    level: float = 0.95,
    n_boot: int = 2000,
) -> GroundingResult:
    """Run the grounding conditions over ``dataset`` and compute CR, CA, drop_k, classes."""

    # 1. run every (item, condition, repeat) and score the outcome.
    async def one(item: GroundingItem, condition: str, contexts: list[str], r: int) -> Measurement:
        answer = await app.answer(item.question, contexts)
        if condition == "cf":
            assert item.cf is not None
            score = follows_cf_fn(answer, item.cf, conflict_policy)
            return Measurement(
                item_id=item.item_id,
                metric="follows_cf",
                condition="cf",
                repeat_index=r,
                score=score,
                reason=answer[:200],
            )
        score = correct_fn(answer, item.expected_output)
        return Measurement(
            item_id=item.item_id,
            metric="correct",
            condition=condition,
            repeat_index=r,
            score=score,
            reason=answer[:200],
        )

    tasks = [
        one(item, cond, contexts, r)
        for item in dataset
        for cond, contexts in _conditions_for(item).items()
        for r in range(repeats)
    ]
    measurements: list[Measurement] = list(await asyncio.gather(*tasks))

    # 2. optional DeepEval Faithfulness on the full-context answer (for false-faithful).
    faith_scores: dict[str, float | None] = {}
    if faithfulness_spec is not None:
        for item in dataset:
            answer = await app.answer(item.question, item.contexts)
            tc = LLMTestCase(
                input=item.question, actual_output=answer, retrieval_context=item.contexts
            )
            fm = await a_measure_once(faithfulness_spec, tc, item_id=item.item_id, condition="full")
            faith_scores[item.item_id] = fm.score

    # 3. per-item outcomes and classification.
    items: list[ItemGrounding] = []
    cr_values: list[float] = []
    ca_values: list[float] = []
    drop_values: dict[int, list[float]] = defaultdict(list)
    n_cf_skipped = 0
    false_faithful: list[str] = []

    for item in dataset:
        cf = _rate(measurements, item.item_id, "correct", "full") or 0.0
        cn = _rate(measurements, item.item_id, "correct", "none") or 0.0
        ca = _rate(measurements, item.item_id, "follows_cf", "cf")
        if item.cf is None or not item.cf.valid:
            n_cf_skipped += 1
        cr = cf - cn
        cr_values.append(cr)
        if ca is not None:
            ca_values.append(ca)

        drops: dict[int, float] = {}
        for k in range(len(item.contexts)):
            loo = _rate(measurements, item.item_id, "correct", f"loo_{k}")
            if loo is not None:
                drops[k] = cf - loo
                drop_values[k].append(cf - loo)

        classification = classify_item(ca=ca, correct_full=cf, correct_none=cn, tau=tau)
        fscore = faith_scores.get(item.item_id)
        is_false_faithful = (
            fscore is not None
            and fscore >= faithfulness_threshold
            and classification == "parametric"
        )
        if is_false_faithful:
            false_faithful.append(item.item_id)
        items.append(
            ItemGrounding(
                item_id=item.item_id,
                correct_full=cf,
                correct_none=cn,
                context_reliance=cr,
                counterfactual_adherence=ca,
                drop_by_chunk=drops,
                classification=classification,
                faithfulness_score=fscore,
                false_faithful=is_false_faithful,
            )
        )

    # 4. aggregate estimates with CIs.
    rng = np.random.default_rng(seed)
    estimates: list[ScoreEstimate] = [
        _aggregate(
            cr_values,
            metric="context_reliance",
            n_repeats=repeats,
            level=level,
            n_boot=n_boot,
            rng=rng,
        )
    ]
    if ca_values:
        estimates.append(
            _aggregate(
                ca_values,
                metric="counterfactual_adherence",
                n_repeats=repeats,
                level=level,
                n_boot=n_boot,
                rng=rng,
            )
        )
    for k in sorted(drop_values):
        estimates.append(
            _aggregate(
                drop_values[k],
                metric=f"chunk_effect_loo_{k}",
                n_repeats=repeats,
                level=level,
                n_boot=n_boot,
                rng=rng,
            )
        )

    class_counts = dict(Counter(it.classification for it in items))
    provenance = build_provenance(
        goldens=[{"item_id": it.item_id, "question": it.question} for it in dataset],
        seed=seed,
        judge_model=(str(faithfulness_spec.kwargs.get("model")) if faithfulness_spec else None),
    )
    return GroundingResult(
        provenance=provenance,
        tau=tau,
        conflict_policy=conflict_policy,
        estimates=estimates,
        items=items,
        measurements=measurements,
        class_counts=class_counts,
        n_cf_skipped=n_cf_skipped,
        false_faithful_items=false_faithful,
    )


def ground(app: RAGApp, dataset: list[GroundingItem], **kwargs: Any) -> GroundingResult:
    """Synchronous wrapper around :func:`a_ground`."""
    return asyncio.run(a_ground(app, dataset, **kwargs))
