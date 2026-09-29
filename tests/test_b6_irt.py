"""B6 acceptance: 2PL recovery >= 0.9 and pruning keeps ranking Kendall tau >= 0.9 (SPEC 6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from causeval.bench import irt_bench as b6


@pytest.mark.sim
def test_b6_parameter_recovery() -> None:
    rec = b6.recover(seed=0)
    assert rec["corr_difficulty"] >= 0.9
    assert rec["corr_discrimination"] >= 0.9
    assert rec["corr_ability"] >= 0.9


@pytest.mark.sim
def test_b6_pruning_preserves_ranking() -> None:
    for seed in range(3):
        res = b6.pruning_preserves_ranking(seed=seed)
        assert 0.3 <= res["keep_frac"] <= 0.5
        assert res["kendall_tau"] >= 0.9


def test_b6_report_written(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(b6, "RESULTS_DIR", tmp_path)
    payload, json_path, md_path = b6.run_offline(seed=0)
    assert json_path.exists() and md_path.exists()
    assert payload["recovery"]["corr_discrimination"] >= 0.9
    assert payload["pruning"]["kendall_tau"] >= 0.9
