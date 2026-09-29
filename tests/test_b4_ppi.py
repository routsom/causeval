"""B4 acceptance: PPI covers the true human mean; naive judge-mean does not (SPEC 6).

Accept: PPI CI covers theta at ~95% (offline: >=0.90 with 300 sims, lenient binomial bound);
the naive judge-mean CI covers poorly because of its systematic bias; PPI CI is narrower than
human-only at equal label count.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from causeval.bench import ppi_bench as b4


def test_b4_report_written(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(b4, "RESULTS_DIR", tmp_path)
    payload, json_path, md_path = b4.run_offline(n_total=800, n_labeled=100, seed=0)
    assert json_path.exists() and md_path.exists()
    assert payload["example_draw"]["ppi_ci_width"] < payload["example_draw"]["human_only_ci_width"]


@pytest.mark.sim
def test_b4_coverage_and_width() -> None:
    c = b4.coverage(n_total=1000, n_labeled=100, n_sims=300, seed=0)
    # PPI covers theta (conservative is fine); naive is biased so it misses badly.
    assert c["ppi_coverage"] >= 0.90
    assert c["naive_coverage"] <= 0.5
    # borrowing the judge's signal makes PPI strictly narrower at the same label count.
    assert c["ppi_mean_width"] < c["human_only_mean_width"]
