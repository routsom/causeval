"""A failed judge call is recorded as an error, never substituted with a default score."""

from __future__ import annotations

import pytest

from causeval.adapters.deepeval_import import LLMTestCase
from causeval.adapters.judge_model import CachedJudgeModel
from causeval.adapters.metric_factory import MetricSpec, a_measure_once
from causeval.core.concurrency import RetryPolicy
from causeval.core.errors import JudgeCallError
from causeval.core.llm_cache import LLMCache
from tests.fakes.fake_judge import FailingMetric, SoftFailMetric, make_fake_generate_fn


def _case() -> LLMTestCase:
    return LLMTestCase(input="q", actual_output="0.9", expected_output="")


@pytest.mark.asyncio
async def test_raising_metric_yields_error_measurement() -> None:
    spec = MetricSpec(builder=lambda: FailingMetric(), label="failing")
    m = await a_measure_once(spec, _case(), item_id="i0")
    assert m.error is not None
    assert m.score is None
    assert m.failed is True
    assert m.passed is None


@pytest.mark.asyncio
async def test_soft_failure_becomes_error_measurement() -> None:
    spec = MetricSpec(builder=lambda: SoftFailMetric(), label="soft")
    m = await a_measure_once(spec, _case(), item_id="i0")
    assert m.error == "rate limited"
    assert m.score is None
    assert m.failed is True


@pytest.mark.asyncio
async def test_judge_model_raises_typed_error_after_retries() -> None:
    gen = make_fake_generate_fn(fail=True)
    fast = RetryPolicy(attempts=2, initial_wait=0.001, max_wait=0.01)
    model = CachedJudgeModel(
        gen,
        model_name="fake-1",
        cache=LLMCache(enabled=False),
        repeat_index=0,
        retry_policy=fast,
    )
    with pytest.raises(JudgeCallError):
        await model.a_generate("ping")
