"""Core: result schemas, LLM call cache, concurrency, cost planner, provenance, errors.

Dependency direction (CLAUDE.md): ``core`` is the base layer and imports from no other
causeval package.
"""

from causeval.core.errors import (
    CausevalError,
    ConfigError,
    InsufficientDataError,
    InterventionInvalidError,
    JudgeCallError,
    ReplayDivergenceError,
)
from causeval.core.schemas import (
    SCHEMA_VERSION,
    Comparison,
    Measurement,
    Provenance,
    RunResult,
    ScoreEstimate,
)

__all__ = [
    "SCHEMA_VERSION",
    "CausevalError",
    "Comparison",
    "ConfigError",
    "InsufficientDataError",
    "InterventionInvalidError",
    "JudgeCallError",
    "Measurement",
    "Provenance",
    "ReplayDivergenceError",
    "RunResult",
    "ScoreEstimate",
]
