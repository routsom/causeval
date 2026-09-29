"""LLM cache: hits avoid re-calling the provider; repeat_index / mode behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest

from causeval.adapters.judge_model import CachedJudgeModel
from causeval.core.llm_cache import CachedResponse, CacheKey, LLMCache
from tests.fakes.fake_judge import make_fake_generate_fn


def _key(repeat_index: int = 0, prompt: str = "hello") -> CacheKey:
    return CacheKey(
        provider="fake",
        model="fake-1",
        params={"temperature": 0.0},
        messages=[{"role": "user", "content": prompt}],
        seed=0,
        repeat_index=repeat_index,
        purpose="judge",
    )


def test_cache_roundtrip(tmp_path: Path) -> None:
    cache = LLMCache(tmp_path / "cache.db")
    key = _key()
    assert cache.get(key) is None
    cache.put(key, CachedResponse(text="world", prompt_tokens=1, completion_tokens=1))
    got = cache.get(key)
    assert got is not None and got.text == "world"
    cache.close()


def test_repeat_index_is_a_distinct_key(tmp_path: Path) -> None:
    cache = LLMCache(tmp_path / "cache.db")
    cache.put(_key(repeat_index=0), CachedResponse(text="r0"))
    assert cache.get(_key(repeat_index=0)).text == "r0"  # type: ignore[union-attr]
    assert cache.get(_key(repeat_index=1)) is None  # different repeat -> miss
    cache.close()


def test_disabled_cache_always_misses(tmp_path: Path) -> None:
    cache = LLMCache(tmp_path / "cache.db", enabled=False)
    cache.put(_key(), CachedResponse(text="x"))
    assert cache.get(_key()) is None
    cache.close()


def test_readonly_cache_does_not_write(tmp_path: Path) -> None:
    path = tmp_path / "cache.db"
    LLMCache(path).close()  # create the db file/schema
    ro = LLMCache(path, readonly=True)
    ro.put(_key(), CachedResponse(text="nope"))
    assert ro.get(_key()) is None
    ro.close()


@pytest.mark.asyncio
async def test_judge_model_second_call_is_a_cache_hit(tmp_path: Path) -> None:
    calls: list[str] = []
    gen = make_fake_generate_fn(responses={"ping": "pong"}, calls=calls)
    cache = LLMCache(tmp_path / "cache.db")

    model = CachedJudgeModel(gen, model_name="fake-1", provider="fake", cache=cache, repeat_index=0)
    first = await model.a_generate("ping")
    second = await model.a_generate("ping")

    assert first == "pong" and second == "pong"
    assert calls == ["ping"], "second call should be served from cache, not re-invoked"
    cache.close()


@pytest.mark.asyncio
async def test_judge_model_distinct_repeat_index_calls_again(tmp_path: Path) -> None:
    calls: list[str] = []
    gen = make_fake_generate_fn(responses={"ping": "pong"}, calls=calls)
    cache = LLMCache(tmp_path / "cache.db")

    m0 = CachedJudgeModel(gen, model_name="fake-1", cache=cache, repeat_index=0)
    m1 = CachedJudgeModel(gen, model_name="fake-1", cache=cache, repeat_index=1)
    await m0.a_generate("ping")
    await m1.a_generate("ping")

    assert calls == ["ping", "ping"], "distinct repeat_index must be a distinct call"
    cache.close()
