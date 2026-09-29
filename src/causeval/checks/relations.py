"""Metamorphic relation registry (SPEC 3.4.2, 3.7).

A metamorphic relation is a transformation of an input plus a claim about how the output must
respond: ``invariant`` (meaning unchanged) or ``directional`` (changes in a stated way). The
registry lives here and is shared with :mod:`causeval.interventions.perturb`, which runs the
relations against an app and measures the response.

A transform receives an :class:`Example` and an RNG and returns a perturbed :class:`Example`
(so repeats can redraw a different paraphrase, typo position, etc.). A transform that does not
apply returns the input unchanged; the relation's ``validity`` then reports it as a no-op so
the engine drops and counts it rather than measuring a non-perturbation.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

RelationKind = Literal["invariant", "directional"]


@dataclass(frozen=True)
class Example:
    """The perturbable part of an item: the question and any retrieval contexts."""

    question: str
    contexts: list[str] = field(default_factory=list)


TransformFn = Callable[[Example, np.random.Generator], Example]
# meaning-preserving / applicability check: (original, perturbed) -> is this a valid perturbation?
ValidityFn = Callable[[Example, Example], bool]


def _not_a_noop(original: Example, perturbed: Example) -> bool:
    """Default validity: the perturbation actually changed the input (else it does not apply)."""
    return (original.question, tuple(original.contexts)) != (
        perturbed.question,
        tuple(perturbed.contexts),
    )


@dataclass(frozen=True)
class MetamorphicRelation:
    name: str
    kind: RelationKind
    transform: TransformFn
    validity: ValidityFn = _not_a_noop
    # for directional relations: the expected sign of the metric change.
    direction: Literal["increase", "decrease"] | None = None
    # marks attribute-swap relations so the engine Holm-adjusts their fairness gaps together.
    attribute_pair: tuple[str, str] | None = None


# ---- built-in transforms ---------------------------------------------------------


def _paraphrase(ex: Example, _rng: np.random.Generator) -> Example:
    """Meaning-preserving reword of the question (deterministic; live runs use an LLM)."""
    return replace(ex, question=f"Could you tell me: {ex.question}")


def _formatting_change(ex: Example, _rng: np.random.Generator) -> Example:
    """Same content, different surface formatting."""
    return replace(ex, question=f"**Question:** {ex.question}")


def _typo_noise(ex: Example, rng: np.random.Generator) -> Example:
    """Swap two adjacent characters in one word (a small, meaning-preserving typo)."""
    words = ex.question.split()
    candidates = [i for i, w in enumerate(words) if len(w) >= 4]
    if not candidates:
        return ex
    i = int(rng.choice(candidates))
    w = list(words[i])
    j = int(rng.integers(0, len(w) - 1))
    w[j], w[j + 1] = w[j + 1], w[j]
    words[i] = "".join(w)
    return replace(ex, question=" ".join(words))


_MC_OPTION = re.compile(r"\(([A-D])\)\s*([^()]+?)(?=\s*\([A-D]\)|$)")


def _reorder_mc_options(ex: Example, rng: np.random.Generator) -> Example:
    """Reorder multiple-choice options (labels stay in place, contents rotate)."""
    matches = list(_MC_OPTION.finditer(ex.question))
    if len(matches) < 2:
        return ex  # no options -> no-op, dropped by validity
    labels = [m.group(1) for m in matches]
    contents = [m.group(2).strip() for m in matches]
    perm = rng.permutation(len(contents))
    if np.array_equal(perm, np.arange(len(contents))):
        perm = np.roll(perm, 1)  # force an actual reorder
    reordered = [contents[k] for k in perm]
    prefix = ex.question[: matches[0].start()]
    body = " ".join(f"({lab}) {c}" for lab, c in zip(labels, reordered, strict=True))
    return replace(ex, question=f"{prefix}{body}")


def _insert_distractor(sentence: str) -> TransformFn:
    """Append an irrelevant sentence to the first context (invariant: score should not move)."""

    def _t(ex: Example, _rng: np.random.Generator) -> Example:
        if not ex.contexts:
            return ex
        ctx = list(ex.contexts)
        ctx[0] = f"{ctx[0]} {sentence}"
        return replace(ex, contexts=ctx)

    return _t


def _attribute_swap(a: str, b: str) -> TransformFn:
    """Swap whole-word occurrences of ``a`` and ``b`` in the question (both directions)."""
    rx = re.compile(rf"\b({re.escape(a)}|{re.escape(b)})\b", re.IGNORECASE)

    def _t(ex: Example, _rng: np.random.Generator) -> Example:
        def _sub(m: re.Match[str]) -> str:
            return b if m.group(0).lower() == a.lower() else a

        return replace(ex, question=rx.sub(_sub, ex.question))

    return _t


# ---- registry --------------------------------------------------------------------

_REGISTRY: dict[str, MetamorphicRelation] = {}


def register(relation: MetamorphicRelation) -> MetamorphicRelation:
    """Add (or replace) a relation in the shared registry."""
    _REGISTRY[relation.name] = relation
    return relation


def get(name: str) -> MetamorphicRelation:
    if name not in _REGISTRY:
        raise KeyError(f"unknown metamorphic relation {name!r}")
    return _REGISTRY[name]


def all_relations() -> list[MetamorphicRelation]:
    return list(_REGISTRY.values())


def attribute_swap_relation(a: str, b: str, *, name: str | None = None) -> MetamorphicRelation:
    """Build (and register) an invariant attribute-swap relation for the pair ``(a, b)``."""
    rel = MetamorphicRelation(
        name=name or f"attribute_swap_{a}_{b}",
        kind="invariant",
        transform=_attribute_swap(a, b),
        attribute_pair=(a, b),
    )
    return register(rel)


def distractor_relation(sentence: str, *, name: str) -> MetamorphicRelation:
    """Build (and register) an invariant distractor-insertion relation."""
    return register(
        MetamorphicRelation(name=name, kind="invariant", transform=_insert_distractor(sentence))
    )


# built-in invariant relations, registered on import.
register(MetamorphicRelation("paraphrase", "invariant", _paraphrase))
register(MetamorphicRelation("formatting_change", "invariant", _formatting_change))
register(MetamorphicRelation("typo_noise", "invariant", _typo_noise))
register(MetamorphicRelation("reorder_mc_options", "invariant", _reorder_mc_options))
