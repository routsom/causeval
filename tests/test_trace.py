"""Unit tests for the agent trace schema (SPEC 3.6)."""

from __future__ import annotations

import pytest

from causeval.attribution.trace import Trace, canonical_args, from_otel_spans, from_records


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


def test_from_otel_spans_maps_kinds_and_orders_by_time() -> None:
    spans = [
        {
            "name": "search",
            "startTimeUnixNano": 2000,
            "endTimeUnixNano": 3000,
            "attributes": {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "search"},
        },
        {
            "name": "chat",
            "startTimeUnixNano": 1000,
            "endTimeUnixNano": 1500,
            "attributes": {"gen_ai.operation.name": "chat", "gen_ai.completion": "hi"},
        },
        {
            "name": "vecdb",
            "startTimeUnixNano": 4000,
            "endTimeUnixNano": 4200,
            "attributes": {"gen_ai.operation.name": "retrieval"},
        },
    ]
    tr = from_otel_spans(spans)
    # ordered by start time: chat (llm), search (tool), retrieval.
    assert [s.kind for s in tr.steps] == ["llm_call", "tool_call", "retrieval"]
    assert tr.steps[1].tool == "search"
    assert tr.steps[0].output == "hi"
    assert tr.steps[0].latency_s == pytest.approx(500 / 1e9)


def test_from_otel_spans_wraps_nondict_input() -> None:
    spans = [
        {
            "name": "chat",
            "attributes": {"gen_ai.operation.name": "chat", "input": "a prompt string"},
        }
    ]
    tr = from_otel_spans(spans)
    assert tr.steps[0].args == {"input": "a prompt string"}
