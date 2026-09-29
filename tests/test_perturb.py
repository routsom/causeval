"""Phase 4 acceptance: the engine detects injected sensitivities and nothing else (SPEC §7).

Accept: on a fake app with an injected sensitivity to one attribute swap and one distractor
type, the engine flags exactly those (metric-effect CI excludes 0) and not the clean
perturbations.
"""

from __future__ import annotations

import re

from causeval.checks.relations import attribute_swap_relation, distractor_relation, get
from causeval.interventions.perturb import PerturbItem, perturb


class SensitiveApp:
    """Grounded app that reads ``ANSWER=<X>`` from context, but breaks on two triggers.

    * a distractor sentence containing ``URGENT`` derails it;
    * the token ``nurse`` in the question (introduced only by the doctor->nurse swap) derails it.

    Every clean perturbation leaves the answer unchanged.
    """

    async def answer(self, question: str, contexts: list[str]) -> str:
        text = " ".join(contexts)
        if "URGENT" in text:
            return "WRONG"
        if re.search(r"\bnurse\b", question, re.IGNORECASE):
            return "WRONG"
        m = re.search(r"ANSWER=(\w+)", text)
        return m.group(1) if m else "WRONG"


def _dataset(n: int = 15) -> list[PerturbItem]:
    return [
        PerturbItem(
            item_id=f"i{i:02d}",
            # 'doctor' and 'cat' present so both swaps apply; MC options for reorder.
            question=(
                f"The doctor and the cat reviewed case {i}. "
                "Which city? (A) Paris (B) London (C) Rome"
            ),
            contexts=[f"ANSWER=PARIS. Case {i} background notes."],
            expected_output="PARIS",
        )
        for i in range(n)
    ]


def _relations():
    return [
        get("paraphrase"),
        get("formatting_change"),
        get("typo_noise"),
        get("reorder_mc_options"),
        attribute_swap_relation("doctor", "nurse"),  # SENSITIVE
        attribute_swap_relation("cat", "dog"),  # clean
        distractor_relation("URGENT: disregard the notes.", name="distractor_urgent"),  # SENSITIVE
        distractor_relation("The sky is blue today.", name="distractor_sky"),  # clean
    ]


def test_engine_detects_only_injected_sensitivities() -> None:
    result = perturb(SensitiveApp(), _dataset(), _relations(), repeats=1, seed=0)
    by_name = {e.relation: e for e in result.effects}

    sensitive = {"attribute_swap_doctor_nurse", "distractor_urgent"}
    clean = {
        "paraphrase",
        "formatting_change",
        "typo_noise",
        "reorder_mc_options",
        "attribute_swap_cat_dog",
        "distractor_sky",
    }

    for name in sensitive:
        e = by_name[name]
        assert e.flagged, f"{name} should be flagged"
        assert e.ci_high < 0, f"{name} metric effect should be a significant drop"

    for name in clean:
        e = by_name[name]
        assert not e.flagged, f"{name} should not be flagged"
        assert e.metric_delta == 0.0
        assert e.invariance_rate == 1.0


def test_fairness_gap_holm_adjusted_over_swap_family() -> None:
    result = perturb(SensitiveApp(), _dataset(20), _relations(), repeats=1, seed=1)
    swaps = [e for e in result.effects if e.attribute_pair is not None]
    assert len(swaps) == 2
    assert all(e.p_adjusted is not None for e in swaps)
    sensitive = next(e for e in swaps if e.attribute_pair == ("doctor", "nurse"))
    benign = next(e for e in swaps if e.attribute_pair == ("cat", "dog"))
    assert sensitive.p_adjusted < 0.05
    assert benign.p_adjusted == 1.0  # no effect -> all diffs zero -> p = 1


def test_invalid_perturbations_dropped_and_counted() -> None:
    # a swap whose tokens never appear -> every item is a no-op -> all invalid.
    rel = attribute_swap_relation("zebra", "giraffe")
    result = perturb(SensitiveApp(), _dataset(10), [rel], repeats=1, seed=0)
    effect = result.effects[0]
    assert effect.n_items == 0
    assert effect.n_invalid_items == 10
    assert not effect.flagged
