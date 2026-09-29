"""Choose which items to send for human labeling (SPEC 3.5).

PPI validity requires the labeled subset to be a *random* sample of the dataset. This module
draws that sample with a recorded seed (so the randomness is auditable), optionally stratified
by judge score, and round-trips a labeling sheet to and from JSONL.

The recorded :class:`LabelingPlan` is what :func:`causeval.judge_audit.ppi.ppi_mean` checks
against: only items in the plan may carry human labels used by PPI.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field


class LabelingPlan(BaseModel):
    """A recorded random labeling sample: the selected ids, the seed, and the method."""

    schema_version: str = "0.1"
    selected_ids: list[str]
    seed: int
    method: str  # "simple_random" or "stratified_by_judge_score"
    n_total: int
    strata: dict[str, list[str]] = Field(default_factory=dict)  # stratum label -> ids (if used)


def simple_random_sample(item_ids: list[str], n: int, *, seed: int) -> LabelingPlan:
    """Draw ``n`` items uniformly at random without replacement, recording the seed."""
    if n > len(item_ids):
        raise ValueError(f"cannot sample {n} from {len(item_ids)} items")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(item_ids, dtype=object), size=n, replace=False)
    return LabelingPlan(
        selected_ids=sorted(map(str, chosen)),
        seed=seed,
        method="simple_random",
        n_total=len(item_ids),
    )


def stratified_sample(
    item_ids: list[str], judge_scores: list[float], n: int, *, seed: int, n_strata: int = 4
) -> LabelingPlan:
    """Stratify by judge-score quantile bins, then sample proportionally within each stratum.

    Reduces variance when the judge score predicts the human label. Still a valid random
    sample within strata (the seed is recorded), so PPI's random-sample check accepts it.
    """
    ids = np.asarray(item_ids, dtype=object)
    scores = np.asarray(judge_scores, dtype=float)
    if ids.size != scores.size:
        raise ValueError("item_ids and judge_scores must be the same length")
    if n > ids.size:
        raise ValueError(f"cannot sample {n} from {ids.size} items")
    rng = np.random.default_rng(seed)

    # quantile edges -> stratum index per item.
    edges = np.quantile(scores, np.linspace(0, 1, n_strata + 1))
    bins = np.clip(np.digitize(scores, edges[1:-1]), 0, n_strata - 1)

    selected: list[str] = []
    strata: dict[str, list[str]] = {}
    for s in range(n_strata):
        mask = bins == s
        stratum_ids = ids[mask]
        # proportional allocation, at least the rounded share.
        take = round(n * stratum_ids.size / ids.size)
        take = min(take, stratum_ids.size)
        picked = rng.choice(stratum_ids, size=take, replace=False) if take > 0 else []
        strata[f"q{s}"] = sorted(map(str, stratum_ids.tolist()))
        selected.extend(map(str, picked))

    # rounding can leave us a few short/over; top up or trim randomly from the remainder.
    remaining = sorted(set(map(str, ids.tolist())) - set(selected))
    while len(selected) < n and remaining:
        j = int(rng.integers(0, len(remaining)))
        selected.append(remaining.pop(j))
    selected = sorted(selected[:n])

    return LabelingPlan(
        selected_ids=selected,
        seed=seed,
        method="stratified_by_judge_score",
        n_total=ids.size,
        strata=strata,
    )


def export_labeling_sheet(
    plan: LabelingPlan, questions: dict[str, str], answers: dict[str, str], path: str | Path
) -> Path:
    """Write a JSONL labeling sheet (one row per selected item) for humans to fill in."""
    out = Path(path)
    lines = []
    for iid in plan.selected_ids:
        row = {
            "item_id": iid,
            "question": questions.get(iid, ""),
            "answer": answers.get(iid, ""),
            "human_label": None,  # to be filled in [0, 1]
        }
        lines.append(json.dumps(row))
    out.write_text("\n".join(lines) + "\n")
    return out


def import_labels(path: str | Path) -> dict[str, float]:
    """Read completed labels back from a JSONL sheet; skips rows with a null ``human_label``."""
    labels: dict[str, float] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("human_label") is None:
            continue
        labels[str(row["item_id"])] = float(row["human_label"])
    return labels
