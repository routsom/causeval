"""A fake judge with configurable bias and noise, plus fake DeepEval metrics for tests.

None of this touches a network. ``FakeJudge`` models a judge whose score is
``clip(true_quality + bias + N(0, noise_sd))`` and is used both directly (statistical
tests) and to build a ``CachedJudgeModel`` generate function.

``FakeScoringMetric`` is a ``BaseMetric`` whose score depends only on its test case, with a
deliberate ``await`` between reading the test case and writing ``self.score``. Sharing one
instance across concurrent measurements would corrupt scores (DeepEval issue #3356); the
metric factory's fresh-instance-per-measurement rule prevents that, which the concurrency
test verifies.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import numpy as np

from causeval.adapters.deepeval_import import BaseMetric, LLMTestCase
from causeval.adapters.judge_model import LLMResponse


class FakeJudge:
    """Deterministic-with-seed judge: score = clip(true_quality + bias + noise)."""

    def __init__(self, *, bias: float = 0.0, noise_sd: float = 0.0, seed: int = 0) -> None:
        self.bias = bias
        self.noise_sd = noise_sd
        self._rng = np.random.default_rng(seed)

    def score(self, true_quality: float) -> float:
        noise = self._rng.normal(0.0, self.noise_sd) if self.noise_sd > 0 else 0.0
        return float(np.clip(true_quality + self.bias + noise, 0.0, 1.0))


def make_fake_generate_fn(
    *,
    responses: dict[str, str] | None = None,
    default: str = "ok",
    delay: float = 0.0,
    calls: list[str] | None = None,
    fail: bool = False,
):
    """Build an async generate function for ``CachedJudgeModel``.

    ``responses`` maps a prompt to its reply; ``calls`` (if given) records every prompt that
    actually reaches the function (i.e. cache misses), so tests can assert cache hits.
    """

    async def generate_fn(prompt: str, *, schema: Any | None = None) -> LLMResponse:
        if calls is not None:
            calls.append(prompt)
        if delay:
            await asyncio.sleep(delay)
        if fail:
            raise RuntimeError("fake provider outage")
        text = (responses or {}).get(prompt, default)
        return LLMResponse(text=text, prompt_tokens=len(prompt.split()), completion_tokens=1)

    return generate_fn


class FakeScoringMetric(BaseMetric):
    """A metric whose score is a function of its own test case, with an internal ``await``.

    ``actual_output`` is parsed as a float in [0, 1] and used as the score, so a test can
    assert each measurement's score matches the item it was given.
    """

    def __init__(self, *, delay: float = 0.01, threshold: float = 0.5) -> None:
        self.delay = delay
        self.threshold = threshold
        self.async_mode = True
        self.score = None
        self.reason = None
        self.success = None
        self.error = None
        self.evaluation_cost = None
        self.input_tokens = None
        self.output_tokens = None

    @property
    def __name__(self) -> str:
        return "FakeScoringMetric"

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        target = float(test_case.actual_output or 0.0)
        # Write state, then yield control before finalizing. A shared instance used by
        # another concurrent task will have `self.score` overwritten across this await --
        # exactly the #3356 hazard. A fresh instance per measurement is immune.
        self.score = target
        self.reason = f"score for {test_case.input}"
        self.error = None
        await asyncio.sleep(self.delay)
        self.success = self.score is not None and self.score >= self.threshold
        return float(self.score)

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return asyncio.run(self.a_measure(test_case))


class FailingMetric(BaseMetric):
    """A metric that always raises, to test failed-call handling."""

    def __init__(self, *, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.async_mode = True
        self.score = None
        self.reason = None
        self.success = None
        self.error = None
        self.evaluation_cost = None

    @property
    def __name__(self) -> str:
        return "FailingMetric"

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        raise RuntimeError("judge call blew up")

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        raise RuntimeError("judge call blew up")


class SoftFailMetric(BaseMetric):
    """A metric that records a soft failure in ``self.error`` without raising.

    Mirrors DeepEval metrics that set ``self.error`` and leave ``self.score`` None.
    """

    def __init__(self, *, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.async_mode = True
        self.score = None
        self.reason = None
        self.success = None
        self.error = None
        self.evaluation_cost = None

    @property
    def __name__(self) -> str:
        return "SoftFailMetric"

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        self.error = "rate limited"
        self.score = None
        return 0.0

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return asyncio.run(self.a_measure(test_case))


def as_json(obj: dict[str, Any]) -> str:
    """Helper: render a dict as a JSON string (for schema-returning fake judges)."""
    return json.dumps(obj)
