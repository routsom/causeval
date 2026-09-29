"""A fake RAG app implementing the ``RAGApp`` protocol (SPEC §3.4.1).

Three behaviours, to exercise the Phase 2 grounding metrics later:
  * ``grounded``   — answers strictly from the supplied contexts.
  * ``parametric`` — ignores contexts, answers from its internal memory.
  * ``mixed``      — uses context when present, else falls back to memory.
"""

from __future__ import annotations

from typing import Literal

Mode = Literal["grounded", "parametric", "mixed"]


class FakeRAGApp:
    """Deterministic RAG app. ``memory`` maps a question to its parametric answer."""

    def __init__(
        self,
        *,
        mode: Mode = "grounded",
        memory: dict[str, str] | None = None,
        no_context_reply: str = "I don't know.",
    ) -> None:
        self.mode = mode
        self.memory = memory or {}
        self.no_context_reply = no_context_reply

    async def answer(self, question: str, contexts: list[str]) -> str:
        if self.mode == "parametric":
            return self.memory.get(question, self.no_context_reply)
        if contexts:
            # Grounded/mixed with context: echo the first context as the evidence used.
            return contexts[0]
        if self.mode == "mixed":
            return self.memory.get(question, self.no_context_reply)
        return self.no_context_reply
