"""Build DeepEval metrics and measure a single test case.

Rule 3: a fresh metric object per measurement. Never share a metric instance across
concurrent calls (DeepEval issue #3356). :func:`build` returns a new instance every call.

Rule (errors): a failed judge call is captured into ``Measurement.error`` with a ``None``
score; it is never replaced by a default value.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from causeval.adapters.deepeval_import import (
    BaseMetric,
    GEval,
    LLMTestCase,
    resolve_metric_class,
)
from causeval.core.errors import ConfigError
from causeval.core.schemas import Measurement

# A user may pass a class-name spec, or a zero-arg callable that returns a fresh metric.
MetricBuilder = Callable[[], BaseMetric]


@dataclass(frozen=True)
class MetricSpec:
    """How to build a metric.

    Either give a DeepEval metric class ``name`` plus ``kwargs``, or a zero-arg
    ``builder`` callable that returns a fresh metric instance. Exactly one is required.
    ``label`` names the metric in results (defaults to ``name`` or the builder's name).
    """

    name: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    builder: MetricBuilder | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if (self.name is None) == (self.builder is None):
            raise ConfigError("MetricSpec needs exactly one of `name` or `builder`")

    @property
    def result_label(self) -> str:
        if self.label is not None:
            return self.label
        if self.name is not None:
            return self.name
        return getattr(self.builder, "__name__", "metric")


def build(spec: MetricSpec) -> BaseMetric:
    """Return a **new** metric instance for ``spec`` on every call (rule 3)."""
    if spec.builder is not None:
        return spec.builder()
    assert spec.name is not None  # guaranteed by MetricSpec.__post_init__
    cls = resolve_metric_class(spec.name)
    try:
        return cls(**spec.kwargs)
    except TypeError as exc:
        raise ConfigError(f"cannot build {spec.name} with {spec.kwargs!r}: {exc}") from exc


def _is_nondeterministic_geval(metric: BaseMetric) -> bool:
    """A GEval given only ``criteria`` regenerates its steps each run (nondeterministic)."""
    if not isinstance(metric, GEval):
        return False
    criteria = getattr(metric, "criteria", None)
    steps = getattr(metric, "evaluation_steps", None)
    return bool(criteria) and not steps


async def a_measure_once(
    spec: MetricSpec,
    test_case: LLMTestCase,
    *,
    item_id: str,
    condition: str = "base",
    repeat_index: int = 0,
) -> Measurement:
    """Build a fresh metric, measure ``test_case`` once, and return a :class:`Measurement`.

    Any exception from the measurement is caught and recorded in ``error`` with a ``None``
    score. A GEval built from bare ``criteria`` is flagged ``nondeterministic_steps=True``.
    """
    metric = build(spec)
    metadata: dict[str, object] = {}
    if _is_nondeterministic_geval(metric):
        metadata["nondeterministic_steps"] = True
        warnings.warn(
            f"GEval metric {spec.result_label!r} was given `criteria` but no "
            "`evaluation_steps`; it regenerates steps each run and is nondeterministic. "
            "Provide locked `evaluation_steps` for reproducible scores.",
            stacklevel=2,
        )

    start = time.perf_counter()
    try:
        await metric.a_measure(test_case)
    except Exception as exc:
        return Measurement(
            item_id=item_id,
            metric=spec.result_label,
            condition=condition,
            repeat_index=repeat_index,
            score=None,
            error=f"{type(exc).__name__}: {exc}",
            latency_s=time.perf_counter() - start,
            metadata=metadata,
        )
    latency = time.perf_counter() - start

    # DeepEval records a soft failure in `metric.error` without raising.
    if metric.error is not None:
        return Measurement(
            item_id=item_id,
            metric=spec.result_label,
            condition=condition,
            repeat_index=repeat_index,
            score=None,
            error=str(metric.error),
            latency_s=latency,
            metadata=metadata,
        )

    return Measurement(
        item_id=item_id,
        metric=spec.result_label,
        condition=condition,
        repeat_index=repeat_index,
        score=metric.score,
        passed=metric.success,
        reason=metric.reason,
        cost_usd=metric.evaluation_cost,
        latency_s=latency,
        metadata=metadata,
    )
