"""Counterfactual context generation (SPEC 3.4.1).

A counterfactual edits one supporting chunk so a single key fact takes a plausible but false
value. If the app's answer then follows the edited value, the answer is grounded in the
context; if it keeps the original value, the answer is parametric.

Generation pipeline (LLM-backed parts are injectable so the engine is testable offline):
1. identify the supporting chunk (caller supplies its index, or the highest-``drop_k`` chunk);
2. extract the key-fact span answering the question;
3. generate a same-type replacement (number->number, date->date, name->name);
4. validity checks -- all must pass or the item is skipped with a recorded reason:
   * local edit: the character diff touches only the target span (+/- punctuation);
   * contradiction: the edited chunk contradicts the original on that fact;
   * naturalness: the edited chunk stays fluent.

Hand-written counterfactuals (``cf_overrides.jsonl``) are supported and preferred in
high-stakes domains.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from causeval.core.errors import InterventionInvalidError

# Injectable LLM-backed steps. Kept as plain callables so tests pass deterministic fakes.
SpanExtractor = Callable[[str, str], Awaitable[str]]  # (question, chunk) -> key-fact substring
Replacer = Callable[[str], Awaitable[str]]  # (original_value) -> same-type false value
ContradictionChecker = Callable[[str, str], bool]  # (original_chunk, edited_chunk) -> contradicts?
NaturalnessChecker = Callable[[str], bool]  # (edited_chunk) -> fluent?


@dataclass(frozen=True)
class Counterfactual:
    """A single-fact edit to one context chunk."""

    chunk_index: int
    original_value: str
    cf_value: str
    edited_chunk: str
    valid: bool = True
    reason_skipped: str | None = None

    @classmethod
    def skipped(cls, chunk_index: int, original_value: str, reason: str) -> Counterfactual:
        return cls(
            chunk_index=chunk_index,
            original_value=original_value,
            cf_value="",
            edited_chunk="",
            valid=False,
            reason_skipped=reason,
        )


def edit_is_local(original_chunk: str, edited_chunk: str, original_value: str) -> bool:
    """True if the edit only replaces the target value's span (text outside it is unchanged).

    Checks that the edited chunk keeps the exact prefix before, and suffix after, the first
    occurrence of ``original_value``. Robust to the old/new values sharing characters (a diff
    opcode approach splits e.g. ``2019 -> 1994`` on the shared ``19``).
    """
    if not original_value:
        return False
    idx = original_chunk.find(original_value)
    if idx == -1:
        return False
    prefix = original_chunk[:idx]
    suffix = original_chunk[idx + len(original_value) :]
    return (
        len(edited_chunk) >= len(prefix) + len(suffix)
        and edited_chunk.startswith(prefix)
        and edited_chunk.endswith(suffix)
    )


def default_contradiction_checker(original_chunk: str, edited_chunk: str) -> bool:
    """Deterministic fallback: the fact text actually changed between the two chunks."""
    return original_chunk != edited_chunk


def default_naturalness_checker(_edited_chunk: str) -> bool:
    """Deterministic fallback: assume fluent (live runs use a judge or perplexity bound)."""
    return True


def build_counterfactual(
    *,
    chunk_index: int,
    original_chunk: str,
    original_value: str,
    cf_value: str,
    contradiction_checker: ContradictionChecker = default_contradiction_checker,
    naturalness_checker: NaturalnessChecker = default_naturalness_checker,
) -> Counterfactual:
    """Apply the edit and run the three validity checks.

    Returns a valid :class:`Counterfactual`, or a skipped one carrying the failure reason.
    Never raises for a failed check -- an invalid intervention is dropped and counted.
    """
    if original_value == cf_value:
        return Counterfactual.skipped(chunk_index, original_value, "cf_value equals original")
    edited_chunk = original_chunk.replace(original_value, cf_value, 1)

    if not edit_is_local(original_chunk, edited_chunk, original_value):
        return Counterfactual.skipped(chunk_index, original_value, "edit is not local")
    if not contradiction_checker(original_chunk, edited_chunk):
        return Counterfactual.skipped(chunk_index, original_value, "edit does not contradict")
    if not naturalness_checker(edited_chunk):
        return Counterfactual.skipped(chunk_index, original_value, "edit is not natural")

    return Counterfactual(
        chunk_index=chunk_index,
        original_value=original_value,
        cf_value=cf_value,
        edited_chunk=edited_chunk,
    )


async def generate_counterfactual(
    *,
    question: str,
    chunks: list[str],
    chunk_index: int,
    extract_span: SpanExtractor,
    replace: Replacer,
    contradiction_checker: ContradictionChecker = default_contradiction_checker,
    naturalness_checker: NaturalnessChecker = default_naturalness_checker,
) -> Counterfactual:
    """LLM-backed generation: extract the key fact, replace it, then validate."""
    if not 0 <= chunk_index < len(chunks):
        raise InterventionInvalidError(f"chunk_index {chunk_index} out of range")
    chunk = chunks[chunk_index]
    original_value = (await extract_span(question, chunk)).strip()
    if not original_value or original_value not in chunk:
        return Counterfactual.skipped(
            chunk_index, original_value, "key-fact span not found in chunk"
        )
    cf_value = (await replace(original_value)).strip()
    return build_counterfactual(
        chunk_index=chunk_index,
        original_chunk=chunk,
        original_value=original_value,
        cf_value=cf_value,
        contradiction_checker=contradiction_checker,
        naturalness_checker=naturalness_checker,
    )


def load_cf_overrides(path: str | Path) -> dict[str, Counterfactual]:
    """Load hand-written counterfactuals from JSONL keyed by ``item_id``.

    Each line: ``{"item_id", "chunk_index", "original_value", "cf_value", "edited_chunk"}``.
    """
    overrides: dict[str, Counterfactual] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        overrides[str(d["item_id"])] = Counterfactual(
            chunk_index=int(d["chunk_index"]),
            original_value=str(d["original_value"]),
            cf_value=str(d["cf_value"]),
            edited_chunk=str(d["edited_chunk"]),
        )
    return overrides
