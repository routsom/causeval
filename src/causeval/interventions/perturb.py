"""Perturbation engine (SPEC 3.4.2).

Runs metamorphic relations from :mod:`causeval.checks.relations` against an app and measures
how the output responds. For each relation we report, over items with a cluster-bootstrap CI:

  * **invariance rate** -- fraction of items whose answer meaning is unchanged (an injectable
    equivalence check; deterministic normalized-equality by default, NLI/judge in live runs);
  * **metric effect** -- the paired ``Delta = score(perturbed) - score(original)`` on an
    outcome metric;
  * **counterfactual fairness gap** -- for attribute-swap relations, the per-pair ``Delta``,
    Holm-adjusted across the attribute-swap family.

Every generated perturbation passes the relation's validity check; a no-op (e.g. an
attribute token absent, or no context to insert a distractor into) is dropped and counted,
never measured as if it were a real perturbation.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel
from scipy import stats as sps

from causeval.checks.deterministic import normalized_equivalence
from causeval.checks.relations import Example, MetamorphicRelation, RelationKind
from causeval.core.provenance import build_provenance
from causeval.core.schemas import Measurement, Provenance
from causeval.interventions.rag import contains_outcome
from causeval.stats.estimate import cluster_bootstrap_ci
from causeval.stats.gate import holm_adjust

CorrectFn = Callable[[str, str | None], float]
EquivalenceFn = Callable[[str, str], bool]


class PerturbApp(Protocol):
    """User app; same shape as ``RAGApp`` so a RAG app can be perturbation-tested directly."""

    async def answer(self, question: str, contexts: list[str]) -> str: ...


@dataclass(frozen=True)
class PerturbItem:
    item_id: str
    question: str
    contexts: list[str] = field(default_factory=list)
    expected_output: str | None = None


class PerturbationEffect(BaseModel):
    schema_version: str = "0.1"
    relation: str
    kind: RelationKind
    n_items: int
    n_invalid_items: int  # items with no valid perturbation draw (dropped)
    invariance_rate: float
    metric_delta: float  # mean_i(score_perturbed - score_original)
    ci_low: float
    ci_high: float
    ci_level: float = 0.95
    method: str
    attribute_pair: tuple[str, str] | None = None
    p_adjusted: float | None = None  # Holm across the attribute-swap family (fairness gap)
    flagged: bool = False  # CI excludes 0 -> the app is sensitive to this perturbation

    @property
    def ci_excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0


class PerturbationResult(BaseModel):
    schema_version: str = "0.1"
    provenance: Provenance
    effects: list[PerturbationEffect]
    measurements: list[Measurement]

    def summary(self) -> str:
        lines = ["Perturbation report:"]
        for e in self.effects:
            flag = "  <-- SENSITIVE" if e.flagged else ""
            padj = f", p_adj={e.p_adjusted:.3f}" if e.p_adjusted is not None else ""
            lines.append(
                f"  {e.relation} [{e.kind}]: Delta={e.metric_delta:+.3f} "
                f"[{int(e.ci_level * 100)}% CI {e.ci_low:+.3f}, {e.ci_high:+.3f}], "
                f"invariance={e.invariance_rate:.2f} "
                f"(n={e.n_items}, invalid={e.n_invalid_items}{padj}){flag}"
            )
        return "\n".join(lines)


def _wilcoxon_p(diffs: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p on nonzero diffs; 1.0 when all diffs are 0."""
    nz = diffs[diffs != 0]
    if nz.size == 0:
        return 1.0
    try:
        return float(sps.wilcoxon(nz).pvalue)
    except ValueError:
        return 1.0


async def a_perturb(
    app: PerturbApp,
    dataset: list[PerturbItem],
    relations: list[MetamorphicRelation],
    *,
    repeats: int = 1,
    seed: int = 0,
    correct_fn: CorrectFn = contains_outcome,
    equivalence_fn: EquivalenceFn = normalized_equivalence,
    level: float = 0.95,
    n_boot: int = 2000,
) -> PerturbationResult:
    """Measure each relation's invariance, metric effect, and (for swaps) fairness gap."""
    rng = np.random.default_rng(seed)
    measurements: list[Measurement] = []

    # per relation -> per item lists of (orig_score, pert_score, equivalent) over valid repeats.
    per_item: dict[str, dict[str, list[tuple[float, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    async def run_one(rel: MetamorphicRelation, item: PerturbItem, r: int) -> None:
        original = Example(question=item.question, contexts=list(item.contexts))
        perturbed = rel.transform(original, rng)
        if not rel.validity(original, perturbed):
            return  # no-op / inapplicable: dropped (counted later as invalid)
        orig_ans = await app.answer(original.question, original.contexts)
        pert_ans = await app.answer(perturbed.question, perturbed.contexts)
        s_orig = correct_fn(orig_ans, item.expected_output)
        s_pert = correct_fn(pert_ans, item.expected_output)
        equivalent = 1.0 if equivalence_fn(orig_ans, pert_ans) else 0.0
        per_item[rel.name][item.item_id].append((s_orig, s_pert, equivalent))
        measurements.append(
            Measurement(
                item_id=item.item_id,
                metric="correct",
                condition=f"{rel.name}:perturbed",
                repeat_index=r,
                score=s_pert,
                reason=pert_ans[:200],
                metadata={"equivalent": equivalent, "delta": s_pert - s_orig},
            )
        )

    await asyncio.gather(
        *(run_one(rel, item, r) for rel in relations for item in dataset for r in range(repeats))
    )

    # aggregate per relation.
    effects: list[PerturbationEffect] = []
    swap_indices: list[int] = []
    swap_pvalues: list[float] = []
    for rel in relations:
        item_map = per_item[rel.name]
        deltas: list[float] = []
        invariances: list[float] = []
        for item in dataset:
            draws = item_map.get(item.item_id)
            if not draws:
                continue
            arr = np.array(draws, dtype=float)
            deltas.append(float(arr[:, 1].mean() - arr[:, 0].mean()))
            invariances.append(float(arr[:, 2].mean()))
        n_valid = len(deltas)
        n_invalid = len(dataset) - n_valid

        if n_valid == 0:
            lo = hi = delta = float("nan")
            inv_rate = float("nan")
        else:
            d = np.array(deltas, dtype=float)
            delta = float(d.mean())
            lo, hi = cluster_bootstrap_ci(d, level=level, n_boot=n_boot, rng=rng)
            inv_rate = float(np.mean(invariances))

        effect = PerturbationEffect(
            relation=rel.name,
            kind=rel.kind,
            n_items=n_valid,
            n_invalid_items=n_invalid,
            invariance_rate=inv_rate,
            metric_delta=delta,
            ci_low=lo,
            ci_high=hi,
            ci_level=level,
            method=f"paired_cluster_bootstrap_studentized_B={n_boot}",
            attribute_pair=rel.attribute_pair,
            flagged=(n_valid > 0 and (lo > 0.0 or hi < 0.0)),
        )
        if rel.attribute_pair is not None and n_valid > 0:
            swap_indices.append(len(effects))
            swap_pvalues.append(_wilcoxon_p(np.array(deltas)))
        effects.append(effect)

    # Holm-adjust the fairness gaps across the attribute-swap family.
    if swap_pvalues:
        adjusted = holm_adjust(swap_pvalues)
        for idx, p_adj in zip(swap_indices, adjusted, strict=True):
            effects[idx] = effects[idx].model_copy(update={"p_adjusted": p_adj})

    provenance = build_provenance(
        goldens=[{"item_id": it.item_id, "question": it.question} for it in dataset],
        seed=seed,
    )
    return PerturbationResult(provenance=provenance, effects=effects, measurements=measurements)


def perturb(
    app: PerturbApp,
    dataset: list[PerturbItem],
    relations: list[MetamorphicRelation],
    **kwargs: Any,
) -> PerturbationResult:
    """Synchronous wrapper around :func:`a_perturb`."""
    return asyncio.run(a_perturb(app, dataset, relations, **kwargs))
