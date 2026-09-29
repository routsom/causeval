"""Core building blocks: schemas invariants, cost planner, provenance hashing."""

from __future__ import annotations

import pytest

from causeval.core.cost import ModelPricing, TokensPerCall, cost_per_call, plan
from causeval.core.errors import ConfigError
from causeval.core.provenance import build_provenance, dataset_hash
from causeval.core.schemas import Measurement, ScoreEstimate


def test_measurement_failed_property() -> None:
    ok = Measurement(item_id="i", metric="m", repeat_index=0, score=0.5)
    bad = Measurement(item_id="i", metric="m", repeat_index=0, error="boom")
    assert ok.failed is False
    assert bad.failed is True and bad.score is None


def test_score_estimate_is_json_serializable() -> None:
    est = ScoreEstimate(
        metric="m",
        estimate=0.5,
        ci_low=0.4,
        ci_high=0.6,
        method="cluster_bootstrap_percentile_B=2000",
        n_items=30,
        n_repeats=10,
    )
    round_tripped = ScoreEstimate.model_validate_json(est.model_dump_json())
    assert round_tripped == est


PRICES = {"m1": ModelPricing(input_usd_per_1m=1.0, output_usd_per_1m=2.0)}


def test_cost_per_call_math() -> None:
    # 1000 prompt tokens @ $1/1M + 500 completion @ $2/1M = 0.001 + 0.001 = 0.002
    cost = cost_per_call("m1", TokensPerCall(prompt=1000, completion=500), PRICES)
    assert cost == pytest.approx(0.002)


def test_cost_plan_total_calls_and_cost() -> None:
    p = plan(
        n_items=30,
        n_metrics=3,
        n_repeats=10,
        n_conditions=1,
        calls_per_metric=1,
        tokens_per_call=TokensPerCall(prompt=1000, completion=500),
        model="m1",
        price_table=PRICES,
    )
    assert p.total_calls == 30 * 3 * 10
    assert p.est_cost_usd == pytest.approx(900 * 0.002)


def test_cost_unpriced_model_raises() -> None:
    with pytest.raises(ConfigError):
        cost_per_call("unknown", TokensPerCall(1, 1), PRICES)


def test_dataset_hash_is_order_sensitive_and_stable() -> None:
    a = dataset_hash([{"q": "1"}, {"q": "2"}])
    b = dataset_hash([{"q": "1"}, {"q": "2"}])
    c = dataset_hash([{"q": "2"}, {"q": "1"}])
    assert a == b
    assert a != c


def test_build_provenance_populates_versions() -> None:
    prov = build_provenance(goldens=[{"q": "1"}], seed=7, judge_model="m1")
    assert prov.seed == 7
    assert prov.judge_model == "m1"
    assert prov.deepeval_version  # importlib metadata, not "unknown" since installed
    assert prov.dataset_hash
