"""Unit tests for the bias probes (SPEC 3.5). B3 coverage lives in test_b3_bias_recovery.py."""

from __future__ import annotations

import numpy as np
import pytest

from causeval.judge_audit.bias_probes import (
    PairItem,
    ScoreItem,
    paired_score_effect,
    position_bias,
    verbose_padding,
)


class _PositionJudge:
    """Picks the first-presented answer with a fixed probability (pure position effect)."""

    def __init__(self, p_first: float, seed: int = 0) -> None:
        self.p_first = p_first
        self._rng = np.random.default_rng(seed)

    async def __call__(self, question: str, first: str, second: str) -> float:
        return 1.0 if self._rng.random() < self.p_first else 0.0


class _PadJudge:
    """Scores 0.5, plus a fixed bonus when the padding signature is present."""

    def __init__(self, bonus: float) -> None:
        self.bonus = bonus

    async def __call__(self, question: str, answer: str) -> float:
        return 0.5 + (self.bonus if "To restate the question" in answer else 0.0)


async def test_position_bias_recovers_injected_offset() -> None:
    pairs = [
        PairItem(f"p{i}", "q?", f"q={i / 300:.3f}|A", f"q={i / 400:.3f}|B") for i in range(300)
    ]
    est = await position_bias(pairs, _PositionJudge(0.65, seed=1), seed=1)
    assert est.ci_low <= 0.15 <= est.ci_high
    assert est.estimate == pytest.approx(0.15, abs=0.05)
    assert est.flagged


async def test_paired_score_effect_recovers_verbosity_bonus() -> None:
    items = [ScoreItem(f"i{i}", "q?", "q=0.5|body") for i in range(80)]
    est = await paired_score_effect(
        items, _PadJudge(0.08), verbose_padding, probe="verbosity", seed=2
    )
    assert est.estimate == pytest.approx(0.08)  # deterministic judge -> degenerate CI at 0.08
    assert est.ci_low <= est.estimate <= est.ci_high
    assert est.flagged


async def test_no_effect_is_not_flagged() -> None:
    items = [ScoreItem(f"i{i}", "q?", "q=0.5|body") for i in range(80)]
    # zero bonus -> effect is exactly 0, CI degenerate at 0, not flagged.
    est = await paired_score_effect(
        items, _PadJudge(0.0), verbose_padding, probe="verbosity", seed=3
    )
    assert est.estimate == pytest.approx(0.0)
    assert not est.flagged


async def test_effect_below_tolerance_not_flagged_even_if_ci_excludes_zero() -> None:
    items = [ScoreItem(f"i{i}", "q?", "q=0.5|body") for i in range(80)]
    est = await paired_score_effect(
        items, _PadJudge(0.02), verbose_padding, probe="verbosity", tolerance=0.05, seed=4
    )
    assert est.ci_excludes_zero  # a real, tiny effect
    assert not est.flagged  # but below the tolerance, so not actionable
