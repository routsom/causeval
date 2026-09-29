"""Unit tests for record/replay cassettes and the novel-input policy (SPEC 3.6)."""

from __future__ import annotations

import pytest

from causeval.attribution.cassette import (
    Cassette,
    RecordingBackend,
    ReplayBackend,
    verify_prefix,
)
from causeval.attribution.harness import ToolSpec
from causeval.attribution.trace import Step
from causeval.core.errors import ReplayDivergenceError


def _specs(**flags):
    calls = {"n": 0}

    def add(a, b):
        calls["n"] += 1
        return a + b

    spec = ToolSpec(name="add", fn=add, **flags)
    return {"add": spec}, calls


def test_record_then_replay_serves_recorded_output() -> None:
    specs, calls = _specs()
    rec = RecordingBackend(specs)
    assert rec.call("add", a=1, b=2) == 3
    assert calls["n"] == 1

    replay = ReplayBackend(specs, rec.cassette)
    assert replay.call("add", a=1, b=2) == 3
    assert calls["n"] == 1  # served from the cassette, tool not called again
    assert replay.fidelity == "recorded"


def test_novel_input_safe_live_is_called_live() -> None:
    specs, _calls = _specs(safe_live=True)
    rec = RecordingBackend(specs)
    rec.call("add", a=1, b=2)
    replay = ReplayBackend(specs, rec.cassette)
    assert replay.call("add", a=10, b=20) == 30  # novel -> live
    assert replay.fidelity == "live"


def test_novel_input_side_effecting_blocked_without_permission() -> None:
    specs, _unused = _specs(side_effecting=True)
    rec = RecordingBackend(specs)
    rec.call("add", a=1, b=2)
    replay = ReplayBackend(specs, rec.cassette)
    with pytest.raises(ReplayDivergenceError, match="side-effecting"):
        replay.call("add", a=5, b=5)


def test_novel_input_falls_back_to_simulator() -> None:
    specs, _calls = _specs()  # neither safe_live nor side_effecting
    rec = RecordingBackend(specs)
    rec.call("add", a=1, b=2)
    replay = ReplayBackend(specs, rec.cassette, simulator=lambda tool, args: 999)
    assert replay.call("add", a=7, b=7) == 999
    assert replay.fidelity == "simulated"


def test_novel_input_no_policy_raises() -> None:
    specs, _ = _specs()
    rec = RecordingBackend(specs)
    rec.call("add", a=1, b=2)
    replay = ReplayBackend(specs, rec.cassette)
    with pytest.raises(ReplayDivergenceError, match="mark it safe_live"):
        replay.call("add", a=3, b=4)


def test_verify_prefix_detects_divergence() -> None:
    recorded = [Step(index=0, kind="tool_call", tool="add", args={"a": 1}, output=3)]
    same = [Step(index=0, kind="tool_call", tool="add", args={"a": 1}, output=3)]
    verify_prefix(recorded, same)  # no raise

    diff_output = [Step(index=0, kind="tool_call", tool="add", args={"a": 1}, output=99)]
    with pytest.raises(ReplayDivergenceError, match="output diverged"):
        verify_prefix(recorded, diff_output)

    with pytest.raises(ReplayDivergenceError, match="fewer than"):
        verify_prefix(recorded, [])


def test_cassette_to_dict_roundtrip() -> None:
    c = Cassette()
    c.record("add", {"a": 1, "b": 2}, 3)
    assert c.has("add", {"b": 2, "a": 1})  # order-independent
    restored = Cassette(c.to_dict())
    assert restored.get("add", {"a": 1, "b": 2}) == 3
