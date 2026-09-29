"""Typed exceptions for causeval.

Never swallow a failed judge call and substitute a default score. A failed measurement
is recorded with its error and excluded from estimates explicitly.
"""

from __future__ import annotations


class CausevalError(Exception):
    """Base class for all causeval errors."""


class JudgeCallError(CausevalError):
    """An LLM judge/app call failed after retries.

    The offending :class:`~causeval.core.schemas.Measurement` records this error and
    carries a ``None`` score; it must never be replaced by a default value.
    """


class InterventionInvalidError(CausevalError):
    """A generated intervention (e.g. a counterfactual edit) failed its validity checks."""


class ReplayDivergenceError(CausevalError):
    """A replayed agent prefix did not reproduce the recorded steps."""


class InsufficientDataError(CausevalError):
    """Not enough data to compute a requested estimate (too few items, repeats, or systems)."""


class ConfigError(CausevalError):
    """User configuration is missing or invalid (e.g. no price table entry for a model)."""
