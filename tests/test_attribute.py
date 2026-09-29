"""Attribution engine unit tests + B5 acceptance (SPEC 3.6, 6)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from causeval.attribution.attribute import attribute_failure, decisive_step_histogram
from causeval.bench import agent_attribution as b5


def test_attribute_finds_injected_fault_single_task() -> None:
    faulty, oracle, _, _, fault_step = b5.build_task(seed=3)
    failed, oracle_trace = asyncio.run(b5._record_traces(faulty, oracle))
    result = asyncio.run(
        attribute_failure(
            faulty, "task", failed, oracle_trace, b5.toy_tools(), tau=0.2, r_refine=30
        )
    )
    assert result.decisive_step == fault_step
    # the decisive step's effect CI clears tau; earlier steps do not.
    decisive = result.effects[result.decisive_step]
    assert decisive.ci_low > 0.2
    for e in result.effects[:fault_step]:
        assert e.ci_low <= 0.2


def test_no_decisive_step_when_nothing_clears_tau() -> None:
    # a clean run (no fault): the oracle equals the natural trajectory, so no effect clears tau.
    _, oracle, _, _, _ = b5.build_task(seed=5)
    clean = oracle  # oracle agent has no fault
    trace, oracle_trace = asyncio.run(b5._record_traces(clean, oracle))
    result = asyncio.run(
        attribute_failure(clean, "task", trace, oracle_trace, b5.toy_tools(), tau=0.2, r_refine=30)
    )
    assert result.decisive_step is None


def test_histogram_keys_by_tool() -> None:
    results = [asyncio.run(b5.run_one(seed))[1] for seed in (0, 1)]
    hist = decisive_step_histogram(results)
    assert sum(hist.values()) == 2
    assert all("tool_call" in k or k == "none" for k in hist)


@pytest.mark.sim
def test_b5_decisive_step_accuracy() -> None:
    cov = asyncio.run(b5.coverage(n_tasks=40, seed=0))
    assert cov["accuracy"] >= 0.9


def test_b5_report_written(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(b5, "RESULTS_DIR", tmp_path)
    payload, json_path, md_path = b5.run_offline(n_tasks=12, seed=0)
    assert json_path.exists() and md_path.exists()
    assert payload["accuracy"] >= 0.9
