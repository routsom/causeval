"""Shared helpers for turning raw measurements into per-item arrays.

Failed measurements (``error is not None`` / ``score is None``) are excluded and counted;
they are never treated as a zero score (CLAUDE.md errors rule).
"""

from __future__ import annotations

from collections import defaultdict

from causeval.core.schemas import Measurement


def group_scores(
    measurements: list[Measurement],
    *,
    metric: str | None = None,
    condition: str | None = None,
) -> tuple[dict[str, list[float]], int]:
    """Return ``(item_id -> [scores across repeats], n_failed)``.

    If ``metric`` / ``condition`` are given, only matching measurements are used.
    """
    scores: dict[str, list[float]] = defaultdict(list)
    n_failed = 0
    for m in measurements:
        if metric is not None and m.metric != metric:
            continue
        if condition is not None and m.condition != condition:
            continue
        if m.error is not None or m.score is None:
            n_failed += 1
            continue
        scores[m.item_id].append(m.score)
    return dict(scores), n_failed


def group_pass(
    measurements: list[Measurement],
    *,
    metric: str | None = None,
    condition: str | None = None,
    threshold: float | None = None,
) -> dict[str, list[bool]]:
    """Return ``item_id -> [pass/fail across repeats]``.

    Uses ``Measurement.passed`` when set, else derives ``score >= threshold`` when a
    ``threshold`` is provided. Measurements with no verdict available are skipped.
    """
    passes: dict[str, list[bool]] = defaultdict(list)
    for m in measurements:
        if metric is not None and m.metric != metric:
            continue
        if condition is not None and m.condition != condition:
            continue
        if m.error is not None:
            continue
        if m.passed is not None:
            passes[m.item_id].append(bool(m.passed))
        elif threshold is not None and m.score is not None:
            passes[m.item_id].append(m.score >= threshold)
    return dict(passes)
