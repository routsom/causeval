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
