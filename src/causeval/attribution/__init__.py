"""Agent attribution: trace recording, deterministic replay, step-level blame (Phase 5+)."""

from causeval.attribution.attribute import (
    AttributionCase,
    AttributionResult,
    OracleSource,
    StepEffect,
    attribute_failure,
    decisive_step_histogram,
)
from causeval.attribution.cassette import (
    Cassette,
    RecordingBackend,
    ReplayBackend,
    verify_prefix,
)
from causeval.attribution.harness import AgentHarness, DictToolBackend, ToolBackend, ToolSpec
from causeval.attribution.trace import (
    Step,
    StepKind,
    Trace,
    canonical_args,
    from_deepeval_trace,
    from_otel_spans,
    from_records,
)

__all__ = [
    "AgentHarness",
    "AttributionCase",
    "AttributionResult",
    "Cassette",
    "DictToolBackend",
    "OracleSource",
    "RecordingBackend",
    "ReplayBackend",
    "Step",
    "StepEffect",
    "StepKind",
    "ToolBackend",
    "ToolSpec",
    "Trace",
    "attribute_failure",
    "canonical_args",
    "decisive_step_histogram",
    "from_deepeval_trace",
    "from_otel_spans",
    "from_records",
    "verify_prefix",
]
