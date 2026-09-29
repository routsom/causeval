"""causeval: a statistically rigorous, causal evaluation layer for LLM apps.

Wraps DeepEval (the measurement instrument) and adds uncertainty, causality, and judge
validity. See ``SPEC.md`` for the design.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from causeval.core import (
    Comparison,
    Measurement,
    Provenance,
    RunResult,
    ScoreEstimate,
)

try:
    __version__ = version("causeval")
except PackageNotFoundError:  # pragma: no cover - not installed (editable dev without metadata)
    __version__ = "0.0.0"

__all__ = [
    "Comparison",
    "Measurement",
    "Provenance",
    "RunResult",
    "ScoreEstimate",
    "__version__",
]
