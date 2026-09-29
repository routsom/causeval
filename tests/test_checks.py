"""Unit tests for the deterministic checks (SPEC 3.7)."""

from __future__ import annotations

from causeval.checks.deterministic import (
    json_schema_check,
    normalized_equivalence,
    regex_check,
    verifier_check,
)

_SCHEMA = {
    "type": "object",
    "required": ["name", "score"],
    "properties": {"name": {"type": "string"}, "score": {"type": "number"}},
}


def test_json_schema_pass() -> None:
    res = json_schema_check('{"name": "a", "score": 0.5}', _SCHEMA)
    assert res.passed


def test_json_schema_missing_required() -> None:
    res = json_schema_check('{"name": "a"}', _SCHEMA)
    assert not res.passed and "missing required" in res.detail


def test_json_schema_wrong_type_and_bool_not_number() -> None:
    assert not json_schema_check('{"name": 1, "score": 0.5}', _SCHEMA).passed
    # bool must not satisfy "number"
    assert not json_schema_check('{"name": "a", "score": true}', _SCHEMA).passed


def test_json_schema_invalid_json() -> None:
    res = json_schema_check("not json", _SCHEMA)
    assert not res.passed and "invalid JSON" in res.detail


def test_json_schema_nested_array() -> None:
    schema = {"type": "array", "items": {"type": "integer"}}
    assert json_schema_check("[1, 2, 3]", schema).passed
    assert not json_schema_check("[1, 2, 3.5]", schema).passed


def test_regex_check() -> None:
    assert regex_check("answer: 42", r"\d+").passed
    assert not regex_check("no digits", r"\d+").passed
    assert regex_check("ABC", r"[A-Z]+", fullmatch=True).passed
    assert not regex_check("ABc", r"[A-Z]+", fullmatch=True).passed


def test_verifier_check_catches_exceptions() -> None:
    assert verifier_check("x", lambda s: len(s) == 1).passed

    def boom(_: str) -> bool:
        raise RuntimeError("bad verifier")

    res = verifier_check("x", boom)
    assert not res.passed and "verifier raised" in res.detail


def test_normalized_equivalence() -> None:
    assert normalized_equivalence("The Cat.", "the cat")
    assert not normalized_equivalence("cat", "dog")
