"""Counterfactual generation and its validity checks (no LLM)."""

from __future__ import annotations

from pathlib import Path

import pytest

from causeval.core.errors import InterventionInvalidError
from causeval.interventions.cf_gen import (
    Counterfactual,
    build_counterfactual,
    edit_is_local,
    generate_counterfactual,
    load_cf_overrides,
)

CHUNK = "Northwind Robotics' Autonomy team was founded in 2019."


def test_build_counterfactual_valid_local_edit() -> None:
    cf = build_counterfactual(
        chunk_index=0, original_chunk=CHUNK, original_value="2019", cf_value="1994"
    )
    assert cf.valid
    assert cf.edited_chunk == CHUNK.replace("2019", "1994")
    assert cf.reason_skipped is None


def test_build_counterfactual_rejects_noop_edit() -> None:
    cf = build_counterfactual(
        chunk_index=0, original_chunk=CHUNK, original_value="2019", cf_value="2019"
    )
    assert not cf.valid
    assert "equals original" in (cf.reason_skipped or "")


def test_build_counterfactual_rejects_when_no_contradiction() -> None:
    # A contradiction checker that always says "no" -> skipped.
    cf = build_counterfactual(
        chunk_index=0,
        original_chunk=CHUNK,
        original_value="2019",
        cf_value="1994",
        contradiction_checker=lambda _o, _e: False,
    )
    assert not cf.valid and "contradict" in (cf.reason_skipped or "")


def test_build_counterfactual_rejects_unnatural_edit() -> None:
    cf = build_counterfactual(
        chunk_index=0,
        original_chunk=CHUNK,
        original_value="2019",
        cf_value="1994",
        naturalness_checker=lambda _e: False,
    )
    assert not cf.valid and "natural" in (cf.reason_skipped or "")


def test_edit_is_local_detects_multi_region_edit() -> None:
    assert edit_is_local(CHUNK, CHUNK.replace("2019", "1994"), "2019")
    # a global rewrite touching more than the target span is not local
    scrambled = "Totally different sentence about 1994 and Autonomy."
    assert not edit_is_local(CHUNK, scrambled, "2019")


def test_edit_is_local_false_when_value_absent() -> None:
    assert not edit_is_local(CHUNK, CHUNK, "not-present")


async def test_generate_counterfactual_pipeline() -> None:
    async def extract(_q: str, _chunk: str) -> str:
        return "2019"

    async def replace(_v: str) -> str:
        return "1994"

    cf = await generate_counterfactual(
        question="When founded?",
        chunks=[CHUNK],
        chunk_index=0,
        extract_span=extract,
        replace=replace,
    )
    assert cf.valid and cf.cf_value == "1994"


async def test_generate_counterfactual_span_not_found_is_skipped() -> None:
    async def extract(_q: str, _chunk: str) -> str:
        return "missing-value"

    async def replace(_v: str) -> str:
        return "x"

    cf = await generate_counterfactual(
        question="q", chunks=[CHUNK], chunk_index=0, extract_span=extract, replace=replace
    )
    assert not cf.valid and "not found" in (cf.reason_skipped or "")


async def test_generate_counterfactual_bad_index_raises() -> None:
    async def extract(_q: str, _c: str) -> str:
        return "2019"

    async def replace(_v: str) -> str:
        return "1994"

    with pytest.raises(InterventionInvalidError):
        await generate_counterfactual(
            question="q", chunks=[CHUNK], chunk_index=5, extract_span=extract, replace=replace
        )


def test_load_cf_overrides(tmp_path: Path) -> None:
    path = tmp_path / "cf.jsonl"
    path.write_text(
        '{"item_id": "a", "chunk_index": 0, "original_value": "2019", '
        '"cf_value": "1994", "edited_chunk": "founded in 1994."}\n'
    )
    overrides = load_cf_overrides(path)
    assert isinstance(overrides["a"], Counterfactual)
    assert overrides["a"].cf_value == "1994"
