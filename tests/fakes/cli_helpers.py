"""Importable helpers for the offline CLI `run` test (module:function config paths)."""

from __future__ import annotations

from causeval.adapters.deepeval_import import BaseMetric
from tests.fakes.fake_judge import FakeScoringMetric


async def constant_app(_input: str) -> str:
    """An app that always returns the same score-encoding output."""
    return "0.7"


def build_fake_metric() -> BaseMetric:
    """Zero-arg factory returning a fresh fake metric (for MetricSpec builder)."""
    return FakeScoringMetric()


class GroundedApp:
    """A RAGApp that answers strictly from the (possibly edited) context."""

    async def answer(self, question: str, contexts: list[str]) -> str:
        return " ".join(contexts) if contexts else "I don't know."


def build_grounded_app() -> GroundedApp:
    """Zero-arg factory returning a grounded RAGApp (for the `ground` CLI test)."""
    return GroundedApp()


def build_attribution_case():
    """Zero-arg factory returning an AttributionCase (for the `attribute` CLI test)."""
    import asyncio

    from causeval.attribution.attribute import AttributionCase
    from causeval.bench import agent_attribution as b5

    faulty, oracle, _, _, _ = b5.build_task(seed=2)
    failed, oracle_trace = asyncio.run(b5._record_traces(faulty, oracle))
    return AttributionCase(
        harness=faulty,
        task="task",
        failed_trace=failed,
        oracle_trace=oracle_trace,
        tools=b5.toy_tools(),
        tau=0.2,
        r_refine=30,
    )
