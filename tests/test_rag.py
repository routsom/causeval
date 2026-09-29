"""RAG grounding engine: outcomes, conditions, classification, and the full run."""

from __future__ import annotations

import pytest

from causeval.interventions.cf_gen import Counterfactual
from causeval.interventions.rag import (
    GroundingItem,
    classify_item,
    contains_outcome,
    follows_cf_outcome,
    ground,
)
from causeval.interventions.rag import _conditions_for as conditions_for

CF = Counterfactual(
    chunk_index=0, original_value="2019", cf_value="1994", edited_chunk="founded in 1994."
)


def test_contains_outcome() -> None:
    assert contains_outcome("It was founded in 2019.", "2019") == 1.0
    assert contains_outcome("No idea.", "2019") == 0.0


def test_follows_cf_follow_context_policy() -> None:
    assert follows_cf_outcome("Founded in 1994.", CF, "follow_context") == 1.0
    assert follows_cf_outcome("Founded in 2019.", CF, "follow_context") == 0.0


def test_follows_cf_flag_conflict_policy() -> None:
    both = "The context says 1994 but I believe it was 2019."
    assert follows_cf_outcome(both, CF, "flag_conflict") == 1.0
    assert follows_cf_outcome("Founded in 1994.", CF, "flag_conflict") == 0.0


def test_classify_item_rules() -> None:
    assert classify_item(ca=0.9, correct_full=1.0, correct_none=0.0, tau=0.5) == "grounded"
    assert classify_item(ca=0.1, correct_full=1.0, correct_none=1.0, tau=0.5) == "parametric"
    assert classify_item(ca=0.0, correct_full=0.0, correct_none=0.0, tau=0.5) == "confabulating"
    assert classify_item(ca=0.0, correct_full=1.0, correct_none=0.0, tau=0.5) == "mixed"
    assert classify_item(ca=None, correct_full=1.0, correct_none=1.0, tau=0.5) == "parametric"


def test_conditions_for_single_chunk_has_no_loo() -> None:
    item = GroundingItem(item_id="i", question="q", contexts=["c0"], cf=CF)
    conds = conditions_for(item)
    assert set(conds) == {"full", "none", "cf"}
    assert conds["none"] == []
    assert conds["cf"] == ["founded in 1994."]


def test_conditions_for_multi_chunk_has_loo() -> None:
    item = GroundingItem(item_id="i", question="q", contexts=["a", "b", "c"])
    conds = conditions_for(item)
    assert "loo_0" in conds and "loo_2" in conds
    assert conds["loo_1"] == ["a", "c"]


def test_conditions_skips_invalid_cf() -> None:
    bad_cf = Counterfactual.skipped(0, "2019", "not local")
    item = GroundingItem(item_id="i", question="q", contexts=["c0"], cf=bad_cf)
    assert "cf" not in conditions_for(item)


class _GroundedApp:
    """Answers from context; states whatever the (possibly edited) context says."""

    async def answer(self, question: str, contexts: list[str]) -> str:
        return " ".join(contexts) if contexts else "I don't know."


def test_ground_end_to_end_grounded_app() -> None:
    item = GroundingItem(
        item_id="i0",
        question="year?",
        contexts=["Founded in 2019."],
        expected_output="2019",
        cf=Counterfactual(
            chunk_index=0, original_value="2019", cf_value="1994", edited_chunk="Founded in 1994."
        ),
    )
    result = ground(_GroundedApp(), [item], repeats=3, seed=0)
    (item_res,) = result.items
    assert item_res.correct_full == 1.0  # states 2019 from context
    assert item_res.correct_none == 0.0  # no context -> "I don't know"
    assert item_res.counterfactual_adherence == 1.0  # follows the edited 1994
    assert item_res.classification == "grounded"
    # CR and CA estimates are present with CIs
    metrics = {e.metric for e in result.estimates}
    assert "context_reliance" in metrics and "counterfactual_adherence" in metrics


def test_ground_result_summary_and_provenance() -> None:
    item = GroundingItem(
        item_id="i0", question="q", contexts=["Founded in 2019."], expected_output="2019"
    )
    result = ground(_GroundedApp(), [item], repeats=2, seed=0)
    assert "RAG grounding" in result.summary()
    assert result.provenance.dataset_hash
    # No cf on this item -> counted as skipped, CA absent.
    assert result.n_cf_skipped == 1
    assert result.items[0].counterfactual_adherence is None


def test_ground_multichunk_reports_chunk_effects() -> None:
    # supporting chunk 0 contains the answer; chunk 1 is a distractor.
    item = GroundingItem(
        item_id="i0",
        question="year?",
        contexts=["Founded in 2019.", "Unrelated distractor sentence."],
        expected_output="2019",
    )
    result = ground(_GroundedApp(), [item], repeats=3, seed=0)
    chunk_metrics = {e.metric for e in result.estimates if e.metric.startswith("chunk_effect")}
    assert chunk_metrics == {"chunk_effect_loo_0", "chunk_effect_loo_1"}
    # dropping the supporting chunk 0 hurts correctness more than dropping the distractor
    drop0 = next(e for e in result.estimates if e.metric == "chunk_effect_loo_0")
    assert drop0.estimate > 0.0


@pytest.mark.parametrize("policy", ["follow_context", "flag_conflict"])
def test_ground_accepts_conflict_policy(policy: str) -> None:
    item = GroundingItem(
        item_id="i0",
        question="q",
        contexts=["Founded in 2019."],
        expected_output="2019",
        cf=Counterfactual(
            chunk_index=0, original_value="2019", cf_value="1994", edited_chunk="Founded in 1994."
        ),
    )
    result = ground(_GroundedApp(), [item], repeats=2, seed=0, conflict_policy=policy)
    assert result.conflict_policy == policy
