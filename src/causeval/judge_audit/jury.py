"""A jury of judges from different model families (SPEC 3.5).

Aggregating several judges reduces single-judge bias and, more importantly, *surfaces* it:
where judges disagree, the verdict is unreliable. We report an aggregate verdict per item plus
Krippendorff's alpha (inter-judge reliability) and the items with the most disagreement.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

Aggregation = Literal["mean", "majority"]


class JurorScore(BaseModel):
    juror: str  # model/family identifier
    score: float  # in [0, 1]


class ItemVerdict(BaseModel):
    item_id: str
    scores: dict[str, float]  # juror -> score
    aggregate: float
    disagreement: float  # std across jurors (spread)


class JuryResult(BaseModel):
    schema_version: str = "0.1"
    aggregation: Aggregation
    threshold: float
    jurors: list[str]
    verdicts: list[ItemVerdict]
    krippendorff_alpha: float  # inter-judge reliability (interval); 1=perfect, 0=chance
    high_disagreement_items: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        return (
            f"jury of {len(self.jurors)} ({self.aggregation}): "
            f"alpha={self.krippendorff_alpha:.3f}; "
            f"{len(self.high_disagreement_items)} high-disagreement item(s)"
        )


def krippendorff_alpha_interval(matrix: np.ndarray) -> float:
    """Krippendorff's alpha for interval data.

    ``matrix`` is ``(n_judges, n_items)``; NaN marks a missing rating. Alpha = 1 - Do/De, where
    Do is observed and De expected disagreement using squared differences over all rating pairs
    within (Do) and across (De) items. Returns 1.0 when there is no variation to disagree over.
    """
    m = np.asarray(matrix, dtype=float)
    # observed disagreement: mean squared diff over pairs of ratings within the same item.
    num_o = 0.0
    den_o = 0.0
    for col in m.T:
        vals = col[~np.isnan(col)]
        if vals.size < 2:
            continue
        diffs = np.subtract.outer(vals, vals) ** 2
        num_o += diffs.sum()
        den_o += vals.size * (vals.size - 1)
    if den_o == 0:
        return 1.0
    do = num_o / den_o

    # expected disagreement: mean squared diff over all pairs of ratings, regardless of item.
    all_vals = m[~np.isnan(m)]
    if all_vals.size < 2:
        return 1.0
    de_diffs = np.subtract.outer(all_vals, all_vals) ** 2
    de = de_diffs.sum() / (all_vals.size * (all_vals.size - 1))
    if de == 0:
        return 1.0
    return float(1.0 - do / de)


def convene_jury(
    scores_by_item: dict[str, dict[str, float]],
    *,
    aggregation: Aggregation = "mean",
    threshold: float = 0.5,
    disagreement_flag: float = 0.25,
) -> JuryResult:
    """Aggregate ``item_id -> {juror -> score}`` into per-item verdicts + reliability.

    ``mean`` averages juror scores; ``majority`` averages their pass indicators at
    ``threshold``. Items whose juror spread (std) exceeds ``disagreement_flag`` are surfaced.
    """
    jurors = sorted({j for s in scores_by_item.values() for j in s})
    verdicts: list[ItemVerdict] = []
    matrix = np.full((len(jurors), len(scores_by_item)), np.nan)
    j_index = {j: i for i, j in enumerate(jurors)}

    high: list[str] = []
    for col, (item_id, scores) in enumerate(scores_by_item.items()):
        vals = np.array(list(scores.values()), dtype=float)
        if aggregation == "mean":
            agg = float(vals.mean())
        else:
            agg = float(np.mean((vals >= threshold).astype(float)))
        spread = float(vals.std()) if vals.size > 1 else 0.0
        if spread > disagreement_flag:
            high.append(item_id)
        for j, v in scores.items():
            matrix[j_index[j], col] = v
        verdicts.append(
            ItemVerdict(item_id=item_id, scores=dict(scores), aggregate=agg, disagreement=spread)
        )

    alpha = krippendorff_alpha_interval(matrix)
    return JuryResult(
        aggregation=aggregation,
        threshold=threshold,
        jurors=jurors,
        verdicts=verdicts,
        krippendorff_alpha=alpha,
        high_disagreement_items=high,
    )
