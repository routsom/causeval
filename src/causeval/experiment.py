"""The top-level :class:`Experiment` orchestrator.

Runs each metric over each dataset item ``R`` times, collects :class:`Measurement`s, and
produces a :class:`RunResult` whose per-metric :class:`ScoreEstimate`s carry cluster-
bootstrap CIs and variance components (never a bare score, CLAUDE.md rule 4).

The app (optional) generates ``actual_output`` from an item's ``input`` once per item; the
``R`` repeats then resample the metric measurement, isolating judge/app-measurement noise.
Interventions that vary the app input (RAG ablation, perturbations) arrive in later phases.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np

from causeval.adapters.deepeval_import import LLMTestCase
from causeval.adapters.metric_factory import MetricSpec, a_measure_once
from causeval.core.provenance import build_provenance
from causeval.core.schemas import Measurement, RunResult, ScoreEstimate
from causeval.stats.estimate import estimate

# An app maps an item's input to a generated output.
AppFn = Callable[[str], Awaitable[str]]


def _normalize_item(item: Any, index: int) -> dict[str, Any]:
    """Accept a dict or a DeepEval Golden-like object; return a plain field dict."""

    def get(key: str) -> Any:
        if isinstance(item, dict):
            return item.get(key)
        return getattr(item, key, None)

    item_id = get("item_id") or get("id") or f"item{index:03d}"
    return {
        "item_id": str(item_id),
        "input": get("input"),
        "expected_output": get("expected_output"),
        "actual_output": get("actual_output"),
        "retrieval_context": get("retrieval_context"),
        "context": get("context"),
    }


class Experiment:
    """Configure and run a repeated-sampling evaluation over a dataset.

    Example::

        exp = Experiment(dataset=goldens, metrics=[MetricSpec("FaithfulnessMetric")], repeats=5)
        run = await exp.a_run()
        print(run.summary())
    """

    def __init__(
        self,
        *,
        dataset: list[Any],
        metrics: list[MetricSpec],
        repeats: int = 5,
        seed: int = 0,
        app: AppFn | None = None,
        name: str = "run",
        judge_model: str | None = None,
    ) -> None:
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        if not metrics:
            raise ValueError("at least one metric is required")
        self.dataset = dataset
        self.metrics = metrics
        self.repeats = repeats
        self.seed = seed
        self.app = app
        self.name = name
        self.judge_model = judge_model

    async def _output_for(self, item: dict[str, Any]) -> str:
        if self.app is not None:
            return await self.app(item["input"])
        if item["actual_output"] is None:
            raise ValueError(
                f"item {item['item_id']!r} has no actual_output and no app was provided"
            )
        return str(item["actual_output"])

    async def a_run(self) -> RunResult:
        items = [_normalize_item(it, i) for i, it in enumerate(self.dataset)]
        outputs = await asyncio.gather(*(self._output_for(it) for it in items))

        test_cases: dict[str, LLMTestCase] = {}
        for it, out in zip(items, outputs, strict=True):
            test_cases[it["item_id"]] = LLMTestCase(
                input=it["input"],
                actual_output=out,
                expected_output=it["expected_output"],
                retrieval_context=it["retrieval_context"],
                context=it["context"],
            )

        tasks = [
            a_measure_once(spec, test_cases[it["item_id"]], item_id=it["item_id"], repeat_index=r)
            for it in items
            for spec in self.metrics
            for r in range(self.repeats)
        ]
        measurements: list[Measurement] = list(await asyncio.gather(*tasks))

        rng = np.random.default_rng(self.seed)
        estimates: list[ScoreEstimate] = []
        for spec in self.metrics:
            label = spec.result_label
            if any(m.metric == label and m.error is None for m in measurements):
                estimates.append(estimate(measurements, metric=label, rng=rng))

        provenance = build_provenance(
            goldens=[_normalize_item(it, i) for i, it in enumerate(self.dataset)],
            seed=self.seed,
            judge_model=self.judge_model,
        )
        return RunResult(
            name=self.name,
            provenance=provenance,
            measurements=measurements,
            estimates=estimates,
        )

    def run(self) -> RunResult:
        """Synchronous wrapper around :meth:`a_run`."""
        return asyncio.run(self.a_run())
