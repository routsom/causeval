"""Unit tests for the agent trace schema (SPEC 3.6)."""

from __future__ import annotations

from causeval.attribution.trace import Trace, canonical_args, from_records


def test_from_records_indexes_steps() -> None:
    tr = from_records(
        [
            {"kind": "tool_call", "tool": "lookup", "args": {"key": "a"}, "output": 5},
            {"kind": "llm_call", "output": "done"},
        ],
        final_output="done",
    )
    assert [s.index for s in tr.steps] == [0, 1]
    assert tr.steps[0].tool == "lookup"
    assert tr.steps[1].kind == "llm_call"
    assert tr.final == "done"


def test_final_falls_back_to_last_step() -> None:
    tr = from_records([{"kind": "tool_call", "output": 42}])
    assert tr.final == 42
    assert Trace().final is None


def test_prefix() -> None:
    tr = from_records([{"output": i} for i in range(5)])
    assert [s.output for s in tr.prefix(3)] == [0, 1, 2]
    assert tr.prefix(0) == []


def test_canonical_args_is_order_independent() -> None:
    assert canonical_args({"a": 1, "b": 2}) == canonical_args({"b": 2, "a": 1})
    assert canonical_args({"a": 1}) != canonical_args({"a": 2})
