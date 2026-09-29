"""Regression test for DeepEval issue #3356 (shared metric objects under concurrency).

The metric factory builds a fresh metric per measurement, so concurrent measurements of
different test cases never clobber each other's ``score``. We also show the negative
control: sharing a single instance across the same concurrent load *does* corrupt scores,
which is exactly what the factory prevents.
"""

from __future__ import annotations

import asyncio

import pytest

from causeval.adapters.deepeval_import import LLMTestCase
from causeval.adapters.metric_factory import MetricSpec, a_measure_once
from tests.fakes.fake_judge import FakeScoringMetric

N = 40


def _test_cases() -> list[LLMTestCase]:
    # actual_output encodes the expected score, unique per item.
    return [
        LLMTestCase(input=f"q{i}", actual_output=f"{i / N:.4f}", expected_output="")
        for i in range(N)
    ]


@pytest.mark.asyncio
async def test_factory_fresh_instance_keeps_scores_correct() -> None:
    spec = MetricSpec(builder=lambda: FakeScoringMetric(delay=0.02), label="fake")
    cases = _test_cases()

    measurements = await asyncio.gather(
        *(
            a_measure_once(spec, tc, item_id=f"item{i}", repeat_index=0)
            for i, tc in enumerate(cases)
        )
    )

    for i, m in enumerate(measurements):
        assert m.error is None
        assert m.score == pytest.approx(i / N, abs=1e-9), (
            f"item{i} got score {m.score}, expected {i / N}"
        )


@pytest.mark.asyncio
async def test_shared_instance_corrupts_scores_negative_control() -> None:
    # Demonstrate the hazard the factory avoids: one shared metric instance.
    shared = FakeScoringMetric(delay=0.02)
    cases = _test_cases()

    async def measure(tc: LLMTestCase) -> float:
        await shared.a_measure(tc)
        return float(shared.score)  # read after an await -> may reflect another task

    results = await asyncio.gather(*(measure(tc) for tc in cases))
    expected = [i / N for i in range(N)]
    # With a shared instance, some observed scores do not line up item-for-item.
    mismatches = sum(1 for r, e in zip(results, expected, strict=True) if abs(r - e) > 1e-9)
    assert mismatches > 0
