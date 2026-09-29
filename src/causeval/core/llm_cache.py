"""Content-addressed SQLite cache for LLM calls.

Key = sha256 of (provider, model, sorted params, messages, seed, repeat_index, purpose).
Repeated samples use a distinct ``repeat_index`` on purpose, so each repeat is a real,
independent LLM call rather than a cache hit (CLAUDE.md conventions).

The cache is process-safe via a lock around a single connection opened with
``check_same_thread=False``; SQLite's own file locking covers multiple processes.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_CACHE_PATH = Path(".causeval") / "cache.db"


@dataclass(frozen=True)
class CacheKey:
    provider: str
    model: str
    params: dict[str, Any]
    messages: list[dict[str, Any]]
    seed: int
    repeat_index: int
    purpose: str  # e.g. "judge", "app", "cf_gen"

    def digest(self) -> str:
        payload = {
            "provider": self.provider,
            "model": self.model,
            "params": self.params,
            "messages": self.messages,
            "seed": self.seed,
            "repeat_index": self.repeat_index,
            "purpose": self.purpose,
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedResponse:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    created_at: str | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
    key TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    purpose TEXT NOT NULL,
    response_text TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost_usd REAL,
    created_at TEXT NOT NULL
);
"""


class LLMCache:
    """SQLite-backed LLM response cache.

    Modes:
      * normal (default): read and write.
      * ``readonly=True``: serve hits, never write (``--cache-readonly``).
      * ``enabled=False``: bypass entirely, always miss, never write (``--no-cache``).
    """

    def __init__(
        self,
        path: str | Path = DEFAULT_CACHE_PATH,
        *,
        enabled: bool = True,
        readonly: bool = False,
    ) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self.readonly = readonly
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def get(self, key: CacheKey) -> CachedResponse | None:
        if not self.enabled or self._conn is None:
            return None
        digest = key.digest()
        with self._lock:
            row = self._conn.execute(
                "SELECT response_text, prompt_tokens, completion_tokens, cost_usd, created_at "
                "FROM llm_cache WHERE key = ?",
                (digest,),
            ).fetchone()
        if row is None:
            return None
        return CachedResponse(
            text=row[0],
            prompt_tokens=row[1],
            completion_tokens=row[2],
            cost_usd=row[3],
            created_at=row[4],
        )

    def put(self, key: CacheKey, response: CachedResponse) -> None:
        if not self.enabled or self.readonly or self._conn is None:
            return
        digest = key.digest()
        created_at = response.created_at or datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache "
                "(key, provider, model, purpose, response_text, prompt_tokens, "
                "completion_tokens, cost_usd, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    digest,
                    key.provider,
                    key.model,
                    key.purpose,
                    response.text,
                    response.prompt_tokens,
                    response.completion_tokens,
                    response.cost_usd,
                    created_at,
                ),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> LLMCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
