"""Offline coverage for the baseline variance benchmark analysis (no LLM)."""

from __future__ import annotations

from pathlib import Path

import pytest

from causeval.bench import baseline_variance as bv
from causeval.core.schemas import Measurement


def _measurements(
    scores_by_item: dict[str, list[float]], *, metric: str = "M"
) -> list[Measurement]:
    out = []
    for item_id, scores in scores_by_item.items():
        for r, s in enumerate(scores):
            out.append(Measurement(item_id=item_id, metric=metric, repeat_index=r, score=s))
    return out


def test_flip_fraction_zero_when_scores_stable() -> None:
    ms = _measurements({"a": [0.9, 0.9, 0.9], "b": [0.1, 0.1, 0.1]})
    (base,) = bv.analyze(ms, threshold=0.5)
    assert base.flip_fraction == 0.0
    assert base.mean_within_item_variance == pytest.approx(0.0, abs=1e-12)
    assert base.n_items == 2 and base.n_repeats == 3


def test_flip_fraction_detects_boundary_flips() -> None:
    # item "a" straddles the 0.5 threshold across repeats -> a flip.
    ms = _measurements({"a": [0.4, 0.6, 0.55], "b": [0.9, 0.95, 0.92]})
    (base,) = bv.analyze(ms, threshold=0.5)
    assert base.flip_fraction == 0.5  # 1 of 2 items flips
    assert base.mean_within_item_variance > 0


def test_failed_measurements_excluded_and_counted() -> None:
    ms = _measurements({"a": [0.9, 0.9]})
    ms.append(Measurement(item_id="a", metric="M", repeat_index=2, error="boom"))
    (base,) = bv.analyze(ms, threshold=0.5)
    assert base.n_failed == 1
    assert base.n_repeats == 2  # only the two good repeats counted


def test_run_offline_writes_reports(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bv, "RESULTS_DIR", tmp_path)
    json_path, md_path = bv.run_offline(repeats=4, n_items=6)
    assert json_path.exists() and md_path.exists()
    assert "Baseline variance" in md_path.read_text()
