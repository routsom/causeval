"""B2 acceptance: Counterfactual Adherence separates grounded vs parametric (AUROC >= 0.9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from causeval.bench import rag_grounding as b2
from causeval.interventions.rag import ground


def test_roc_auc_basic() -> None:
    assert b2.roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert b2.roc_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == pytest.approx(0.0)
    # ties -> 0.5
    assert b2.roc_auc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == pytest.approx(0.5)


@pytest.mark.sim
def test_b2_ca_auroc_offline(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(b2, "RESULTS_DIR", tmp_path)
    payload, json_path, md_path = b2.run_offline(n=40, repeats=8, seed=0)
    assert payload["ca_auroc"] >= 0.9
    assert json_path.exists() and md_path.exists()
    # behaviour is recovered: grounded items classified grounded, parametric as parametric.
    assert payload["class_counts"].get("grounded", 0) == 20
    assert payload["class_counts"].get("parametric", 0) == 20


@pytest.mark.sim
def test_b2_ca_auroc_survives_noise() -> None:
    """With noisier apps (0.8/0.2 follow rates) CA still separates the two behaviours."""
    labeled = b2.build_labeled_dataset(60, seed=1)
    labels = {li.item.item_id: li.label for li in labeled}
    app = b2.BehaviorApp(labeled, follow_prob_grounded=0.8, follow_prob_parametric=0.2, seed=1)
    result = ground(app, [li.item for li in labeled], repeats=8, seed=1)
    scores = [it.counterfactual_adherence for it in result.items]
    ys = [labels[it.item_id] for it in result.items]
    assert b2.roc_auc(scores, ys) >= 0.9
