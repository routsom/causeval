"""Offline fakes. Offline tests must never call a real LLM (CLAUDE.md rule 7)."""

from tests.fakes.fake_agent import FakeAgentHarness, ToolBackend
from tests.fakes.fake_judge import FakeJudge, FakeScoringMetric, make_fake_generate_fn
from tests.fakes.fake_rag import FakeRAGApp

__all__ = [
    "FakeAgentHarness",
    "FakeJudge",
    "FakeRAGApp",
    "FakeScoringMetric",
    "ToolBackend",
    "make_fake_generate_fn",
]
