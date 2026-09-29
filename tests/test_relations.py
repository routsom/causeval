"""Unit tests for the metamorphic relation registry and built-in transforms (SPEC 3.4.2)."""

from __future__ import annotations

import numpy as np
import pytest

from causeval.checks.relations import (
    Example,
    attribute_swap_relation,
    distractor_relation,
    get,
    register,
)


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_builtins_registered() -> None:
    for name in ("paraphrase", "formatting_change", "typo_noise", "reorder_mc_options"):
        assert get(name).name == name


def test_get_unknown_raises() -> None:
    with pytest.raises(KeyError, match="unknown metamorphic relation"):
        get("does_not_exist")


def test_paraphrase_changes_question_only() -> None:
    rel = get("paraphrase")
    ex = Example(question="What is the capital?", contexts=["ctx"])
    out = rel.transform(ex, _rng())
    assert out.question != ex.question
    assert out.contexts == ex.contexts
    assert rel.validity(ex, out)  # not a no-op


def test_typo_noise_preserves_length_of_word_count() -> None:
    rel = get("typo_noise")
    ex = Example(question="According to the official record what happened", contexts=[])
    out = rel.transform(ex, np.random.default_rng(3))
    assert out.question != ex.question
    assert len(out.question.split()) == len(ex.question.split())


def test_reorder_requires_options() -> None:
    rel = get("reorder_mc_options")
    no_opts = Example(question="What city?", contexts=[])
    out = rel.transform(no_opts, _rng())
    assert not rel.validity(no_opts, out)  # no options -> no-op -> dropped

    with_opts = Example(question="Which city? (A) Paris (B) London (C) Rome", contexts=[])
    out2 = rel.transform(with_opts, _rng())
    assert rel.validity(with_opts, out2)
    # labels preserved, contents rearranged.
    assert "(A)" in out2.question and "(B)" in out2.question and "(C)" in out2.question


def test_attribute_swap_applies_and_is_noop_when_absent() -> None:
    rel = attribute_swap_relation("doctor", "nurse")
    present = Example(question="The doctor arrived", contexts=[])
    out = rel.transform(present, _rng())
    assert "nurse" in out.question and "doctor" not in out.question
    assert rel.validity(present, out)
    assert rel.attribute_pair == ("doctor", "nurse")

    absent = Example(question="The engineer arrived", contexts=[])
    out2 = rel.transform(absent, _rng())
    assert not rel.validity(absent, out2)  # token absent -> no-op


def test_attribute_swap_is_bidirectional() -> None:
    rel = attribute_swap_relation("he", "she", name="pronoun_swap")
    ex = Example(question="he told her", contexts=[])
    out = rel.transform(ex, _rng())
    assert out.question == "she told her"


def test_distractor_appends_to_first_context() -> None:
    rel = distractor_relation("The sky is blue.", name="d_sky")
    ex = Example(question="q", contexts=["ANSWER=PARIS."])
    out = rel.transform(ex, _rng())
    assert out.contexts[0].endswith("The sky is blue.")
    assert rel.validity(ex, out)

    no_ctx = Example(question="q", contexts=[])
    assert not rel.validity(no_ctx, rel.transform(no_ctx, _rng()))


def test_register_returns_relation() -> None:
    rel = distractor_relation("noise", name="d_noise")
    assert register(rel) is rel
    assert get("d_noise") is rel
