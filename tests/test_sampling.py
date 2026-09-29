"""Unit tests for human-labeling sampling (SPEC 3.5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from causeval.judge_audit.sampling import (
    export_labeling_sheet,
    import_labels,
    simple_random_sample,
    stratified_sample,
)


def test_simple_random_sample_is_reproducible_and_sized() -> None:
    ids = [f"item{i}" for i in range(100)]
    a = simple_random_sample(ids, 20, seed=7)
    b = simple_random_sample(ids, 20, seed=7)
    assert a.selected_ids == b.selected_ids  # seed recorded -> reproducible
    assert len(a.selected_ids) == 20
    assert set(a.selected_ids).issubset(set(ids))
    assert a.method == "simple_random" and a.seed == 7


def test_simple_random_sample_rejects_oversample() -> None:
    with pytest.raises(ValueError, match="cannot sample"):
        simple_random_sample(["a", "b"], 5, seed=0)


def test_stratified_sample_covers_all_strata() -> None:
    ids = [f"item{i}" for i in range(200)]
    scores = [i / 200 for i in range(200)]  # spread across [0, 1)
    plan = stratified_sample(ids, scores, 40, seed=3, n_strata=4)
    assert len(plan.selected_ids) == 40
    assert plan.method == "stratified_by_judge_score"
    assert len(plan.strata) == 4
    assert set(plan.selected_ids).issubset(set(ids))


def test_labeling_sheet_roundtrip(tmp_path: Path) -> None:
    ids = [f"item{i}" for i in range(10)]
    plan = simple_random_sample(ids, 4, seed=1)
    sheet = export_labeling_sheet(
        plan,
        questions={i: f"q for {i}" for i in ids},
        answers={i: f"a for {i}" for i in ids},
        path=tmp_path / "sheet.jsonl",
    )
    # simulate a human filling in two labels, leaving two blank.
    lines = sheet.read_text().splitlines()
    import json

    filled = []
    for k, line in enumerate(lines):
        row = json.loads(line)
        if k < 2:
            row["human_label"] = 1.0
        filled.append(json.dumps(row))
    sheet.write_text("\n".join(filled) + "\n")

    labels = import_labels(sheet)
    assert len(labels) == 2
    assert all(v == 1.0 for v in labels.values())
    assert set(labels).issubset(set(plan.selected_ids))
