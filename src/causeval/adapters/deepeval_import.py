"""The one and only place causeval imports DeepEval (CLAUDE.md rule 2).

Before the first ``import deepeval`` this module disables DeepEval's telemetry and dotenv
loading, but only if the user has not already set those variables. No other module in
``src/causeval`` may import deepeval directly; a test enforces this.
"""

from __future__ import annotations

import os

# Must run before the first `import deepeval`. `setdefault` respects a user who set these.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
os.environ.setdefault("DEEPEVAL_DISABLE_DOTENV", "1")

from deepeval import metrics as _metrics_ns
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.base_model import DeepEvalModelData
from deepeval.test_case import LLMTestCase


def resolve_metric_class(name: str) -> type[BaseMetric]:
    """Look up a DeepEval metric class by name (e.g. ``"FaithfulnessMetric"``).

    Kept here so callers never import ``deepeval.metrics`` themselves.
    """
    cls = getattr(_metrics_ns, name, None)
    if not isinstance(cls, type) or not issubclass(cls, BaseMetric):
        available = sorted(n for n in dir(_metrics_ns) if n.endswith(("Metric", "Eval")))
        raise ValueError(f"unknown DeepEval metric {name!r}; available: {available}")
    return cls


__all__ = [
    "AnswerRelevancyMetric",
    "BaseMetric",
    "ContextualPrecisionMetric",
    "ContextualRecallMetric",
    "ContextualRelevancyMetric",
    "DeepEvalBaseLLM",
    "DeepEvalModelData",
    "FaithfulnessMetric",
    "GEval",
    "HallucinationMetric",
    "LLMTestCase",
    "resolve_metric_class",
]
