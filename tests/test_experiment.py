"""Experiment orchestration: repeats -> measurements -> estimates -> RunResult."""

from __future__ import annotations

import pytest

from causeval import Experiment, MetricSpec
from tests.fakes.fake_judge import FakeScoringMetric


def _dataset(n: int = 12) -> list[dict[str, str]]:
    return [
        {"item_id": f"i{i:02d}", "input": f"q{i}", "actual_output": f"{0.4 + 0.02 * i:.3f}"}
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_experiment_produces_estimate_with_ci() -> None:
    spec = MetricSpec(builder=lambda: FakeScoringMetric(), label="fake")
    exp = Experiment(dataset=_dataset(), metrics=[spec], repeats=3, seed=0)
    run = await exp.a_run()

    assert len(run.measurements) == 12 * 3
    (est,) = run.estimates
    assert est.metric == "fake"
    assert est.n_items == 12 and est.n_repeats == 3
    assert est.ci_low <= est.estimate <= est.ci_high
    assert est.method.startswith("cluster_bootstrap_")


def test_experiment_sync_wrapper_and_summary() -> None:
    spec = MetricSpec(builder=lambda: FakeScoringMetric(), label="fake")
    run = Experiment(dataset=_dataset(), metrics=[spec], repeats=2, seed=1).run()
    text = run.summary()
    assert "fake" in text
    assert "CI" in text  # never a bare score


def test_experiment_missing_output_raises() -> None:
    spec = MetricSpec(builder=lambda: FakeScoringMetric(), label="fake")
    exp = Experiment(dataset=[{"item_id": "x", "input": "q"}], metrics=[spec], repeats=1)
    with pytest.raises(ValueError, match="no actual_output"):
        exp.run()


def test_experiment_app_generates_output() -> None:
    async def app(_input: str) -> str:
        return "0.9"

    spec = MetricSpec(builder=lambda: FakeScoringMetric(), label="fake")
    exp = Experiment(dataset=[{"item_id": "x", "input": "q"}], metrics=[spec], repeats=4, app=app)
    run = exp.run()
    assert all(m.score == 0.9 for m in run.measurements)
