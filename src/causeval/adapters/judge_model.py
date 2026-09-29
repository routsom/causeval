"""Adapt a causeval judge client into a DeepEval ``DeepEvalBaseLLM``.

Any user-supplied async generate function can be passed to DeepEval metrics through
:class:`CachedJudgeModel`. Every call is content-addressed in ``core.llm_cache`` (rule:
all LLM calls go through the cache). Repeated samples use distinct ``repeat_index`` values
so each repeat is a genuinely independent call, not a cache hit.

A fresh :class:`CachedJudgeModel` should be built per measurement (carrying that
measurement's ``repeat_index``); do not mutate ``repeat_index`` on a shared instance under
concurrency.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from causeval.adapters.deepeval_import import DeepEvalBaseLLM
from causeval.core.concurrency import RetryPolicy, call_with_retry
from causeval.core.errors import JudgeCallError
from causeval.core.llm_cache import CachedResponse, CacheKey, LLMCache


@dataclass(frozen=True)
class LLMResponse:
    """Return type of a user generate function; text plus optional accounting."""

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None


# A user generate function: (prompt, *, schema=None) -> LLMResponse | str
GenerateFn = Callable[..., Awaitable["LLMResponse | str"]]


class CachedJudgeModel(DeepEvalBaseLLM):  # type: ignore[misc]  # base is Any (deepeval untyped)
    """A ``DeepEvalBaseLLM`` that caches and retries a user-supplied generate function.

    ``a_generate`` returns a plain string; DeepEval parses it against any requested schema
    via its own ``trimAndLoadJson``, so the user function need only return text (optionally
    valid JSON when a schema is expected).
    """

    def __init__(
        self,
        generate_fn: GenerateFn,
        *,
        model_name: str,
        provider: str = "custom",
        cache: LLMCache | None = None,
        params: dict[str, Any] | None = None,
        seed: int = 0,
        repeat_index: int = 0,
        purpose: str = "judge",
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._generate_fn = generate_fn
        self._model_name = model_name
        self._provider = provider
        self._cache = cache if cache is not None else LLMCache(enabled=False)
        self._params = params or {}
        self._seed = seed
        self._repeat_index = repeat_index
        self._purpose = purpose
        self._retry_policy = retry_policy
        # Tracks accounting for the most recent generate (DeepEval doesn't read it for
        # non-native models, but callers can inspect it).
        self.last_response: LLMResponse | None = None
        super().__init__(model=model_name)

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        # No client object to load; the generate function is the client.
        return None

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return self._model_name

    def _cache_key(self, prompt: str, schema: Any | None) -> CacheKey:
        params = dict(self._params)
        if schema is not None:
            params["_schema"] = getattr(schema, "__name__", str(schema))
        return CacheKey(
            provider=self._provider,
            model=self._model_name,
            params=params,
            messages=[{"role": "user", "content": prompt}],
            seed=self._seed,
            repeat_index=self._repeat_index,
            purpose=self._purpose,
        )

    async def a_generate(
        self, prompt: str, schema: Any | None = None, *args: Any, **kwargs: Any
    ) -> str:
        key = self._cache_key(prompt, schema)
        cached = self._cache.get(key)
        if cached is not None:
            self.last_response = LLMResponse(
                text=cached.text,
                prompt_tokens=cached.prompt_tokens,
                completion_tokens=cached.completion_tokens,
                cost_usd=cached.cost_usd,
            )
            return cached.text

        async def _call() -> LLMResponse:
            raw = await self._generate_fn(prompt, schema=schema)
            return raw if isinstance(raw, LLMResponse) else LLMResponse(text=str(raw))

        try:
            resp = await call_with_retry(_call, policy=self._retry_policy)
        except Exception as exc:
            raise JudgeCallError(
                f"judge {self._model_name!r} failed after retries: {type(exc).__name__}: {exc}"
            ) from exc

        self.last_response = resp
        self._cache.put(
            key,
            CachedResponse(
                text=resp.text,
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
                cost_usd=resp.cost_usd,
            ),
        )
        return resp.text

    def generate(self, prompt: str, schema: Any | None = None, *args: Any, **kwargs: Any) -> str:
        """Synchronous wrapper around :meth:`a_generate`.

        Raises if called from within a running event loop; use ``a_measure`` / async paths
        under concurrency instead.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.a_generate(prompt, schema))
        raise JudgeCallError(
            "CachedJudgeModel.generate() called inside a running event loop; "
            "use the async API (a_measure / a_generate)"
        )
