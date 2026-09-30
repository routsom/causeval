"""Unit tests for chain-of-thought faithfulness (SPEC 3.4.3)."""

from __future__ import annotations

from causeval.interventions.cot import CoTItem, cot_faithfulness


class FaithfulModel:
    """Answer is driven by the reasoning: it echoes the last (uncorrupted) step's tag.

    Each step is ``step<k>=<tag>``; the final answer is the tag of the last step. Truncating
    changes the answer; corrupting the last step changes it too.
    """

    def __init__(self, n_steps: int = 4) -> None:
        self.n_steps = n_steps

    async def reason(self, question: str) -> tuple[str, list[str]]:
        steps = [f"step{k}=t{k}" for k in range(self.n_steps)]
        return f"t{self.n_steps - 1}", steps

    async def answer_given(self, question: str, reasoning: list[str]) -> str:
        if not reasoning:
            return "no_answer"
        last = reasoning[-1]
        # answer = the tag after '=', unless the step was corrupted.
        if "=" not in last or last.startswith("Actually"):
            return "corrupted"
        return last.split("=")[1]


class UnfaithfulModel:
    """Answer is fixed regardless of the reasoning provided."""

    async def reason(self, question: str) -> tuple[str, list[str]]:
        return "42", [f"filler step {k}" for k in range(4)]

    async def answer_given(self, question: str, reasoning: list[str]) -> str:
        return "42"


def _dataset(n: int = 8) -> list[CoTItem]:
    return [CoTItem(item_id=f"q{i}", question=f"question {i}") for i in range(n)]


def test_faithful_model_has_high_aoc_and_mistake_sensitivity() -> None:
    res = cot_faithfulness(FaithfulModel(), _dataset(), seed=0)
    # answer changes as reasoning is truncated -> low agreement at small fractions -> high AOC.
    assert res.aoc > 0.2
    assert res.agreement_curve[0] < res.agreement_curve[-1]
    assert res.agreement_curve[-1] == 1.0  # full reasoning reproduces the full answer
    # corrupting the decisive step flips the answer often.
    assert res.mistake_sensitivity > 0.0


def test_unfaithful_model_has_low_aoc_and_zero_sensitivity() -> None:
    res = cot_faithfulness(UnfaithfulModel(), _dataset(), seed=0)
    assert res.aoc < 0.05  # agreement is 1.0 everywhere -> AOC ~ 0
    assert res.mistake_sensitivity == 0.0


def test_faithful_beats_unfaithful() -> None:
    ds = _dataset(10)
    faithful = cot_faithfulness(FaithfulModel(), ds, seed=1)
    unfaithful = cot_faithfulness(UnfaithfulModel(), ds, seed=1)
    assert faithful.aoc > unfaithful.aoc
    assert faithful.mistake_sensitivity > unfaithful.mistake_sensitivity


def test_summary_and_provenance() -> None:
    res = cot_faithfulness(FaithfulModel(), _dataset(3), seed=0)
    assert "CoT faithfulness" in res.summary()
    assert res.n_items == 3
    assert len(res.per_item_aoc) == 3
    assert res.provenance.seed == 0
