"""Concurrency and retry primitives for LLM calls.

A single global :class:`asyncio.Semaphore` bounds in-flight calls (default 8). Retries use
exponential backoff with jitter. When retries are exhausted the exception propagates; the
caller records it as ``Measurement.error`` and never substitutes a default score
(CLAUDE.md rule on errors).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

T = TypeVar("T")

DEFAULT_CONCURRENCY = 8

_semaphore: asyncio.Semaphore | None = None
_semaphore_limit = DEFAULT_CONCURRENCY


def configure_concurrency(limit: int) -> None:
    """Set the global concurrency limit. Resets the shared semaphore."""
    global _semaphore, _semaphore_limit
    if limit < 1:
        raise ValueError("concurrency limit must be >= 1")
    _semaphore_limit = limit
    _semaphore = None  # rebuilt lazily on the running loop


def get_semaphore() -> asyncio.Semaphore:
    """Return the shared semaphore, building it lazily on the current event loop."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_semaphore_limit)
    return _semaphore


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 4
    initial_wait: float = 0.5
    max_wait: float = 30.0
    retry_on: tuple[type[BaseException], ...] = (Exception,)


async def call_with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy | None = None,
    use_semaphore: bool = True,
) -> T:
    """Await ``fn()`` under the global semaphore, retrying with jittered backoff.

    Raises the last underlying exception (unwrapped from tenacity's ``RetryError``) when
    all attempts fail.
    """
    policy = policy or RetryPolicy()
    retrying = AsyncRetrying(
        stop=stop_after_attempt(policy.attempts),
        wait=wait_random_exponential(multiplier=policy.initial_wait, max=policy.max_wait),
        retry=retry_if_exception_type(policy.retry_on),
        reraise=True,
    )

    async def _guarded() -> T:
        if use_semaphore:
            async with get_semaphore():
                return await fn()
        return await fn()

    try:
        async for attempt in retrying:
            with attempt:
                return await _guarded()
    except RetryError as exc:  # pragma: no cover - reraise=True unwraps, kept for safety
        raise (exc.last_attempt.exception() or exc) from exc
    raise AssertionError("unreachable: AsyncRetrying yielded no attempts")
