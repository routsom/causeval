"""Unit tests for the jury of judges (SPEC 3.5)."""

from __future__ import annotations

import numpy as np
import pytest

from causeval.judge_audit.jury import convene_jury, krippendorff_alpha_interval


def test_krippendorff_perfect_agreement() -> None:
    m = np.array([[0.2, 0.8, 0.5], [0.2, 0.8, 0.5]])
    assert krippendorff_alpha_interval(m) == pytest.approx(1.0)


def test_krippendorff_disagreement_lowers_alpha() -> None:
    agree = np.array([[0.1, 0.9], [0.15, 0.85]])
    disagree = np.array([[0.1, 0.9], [0.9, 0.1]])
    assert krippendorff_alpha_interval(agree) > krippendorff_alpha_interval(disagree)


def test_convene_jury_mean_and_disagreement() -> None:
    scores = {
        "item0": {"gpt": 0.9, "claude": 0.85, "llama": 0.88},  # agree
        "item1": {"gpt": 0.9, "claude": 0.1, "llama": 0.5},  # disagree
    }
    res = convene_jury(scores, aggregation="mean", disagreement_flag=0.25)
    by_id = {v.item_id: v for v in res.verdicts}
    assert by_id["item0"].aggregate == pytest.approx(np.mean([0.9, 0.85, 0.88]))
    assert "item1" in res.high_disagreement_items
    assert "item0" not in res.high_disagreement_items
    assert res.jurors == ["claude", "gpt", "llama"]


def test_convene_jury_majority() -> None:
    scores = {"item0": {"a": 0.9, "b": 0.9, "c": 0.1}}  # 2/3 pass at 0.5
    res = convene_jury(scores, aggregation="majority", threshold=0.5)
    assert res.verdicts[0].aggregate == pytest.approx(2 / 3)
