"""Build :class:`Provenance` records.

Version numbers come from installed package metadata (``importlib.metadata``), which does
*not* import deepeval and so respects the import guard (CLAUDE.md rule 2).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from causeval.core.schemas import Provenance


def package_version(name: str, default: str = "unknown") -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return default


def _canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace, stable separators."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def dataset_hash(goldens: Any) -> str:
    """sha256 of the canonical JSON of the goldens.

    ``goldens`` may be anything JSON-serializable; pydantic models are dumped first.
    """
    if hasattr(goldens, "model_dump"):
        payload: Any = goldens.model_dump(mode="json")
    elif isinstance(goldens, (list, tuple)):
        payload = [g.model_dump(mode="json") if hasattr(g, "model_dump") else g for g in goldens]
    else:
        payload = goldens
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def git_sha(short: bool = False) -> str | None:
    """Current git SHA, or ``None`` when not in a git repo / git unavailable."""
    args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def build_provenance(
    *,
    goldens: Any,
    seed: int,
    judge_model: str | None = None,
    judge_params: dict[str, object] | None = None,
    prompt_hashes: dict[str, str] | None = None,
) -> Provenance:
    return Provenance(
        causeval_version=package_version("causeval"),
        deepeval_version=package_version("deepeval"),
        judge_model=judge_model,
        judge_params=judge_params or {},
        prompt_hashes=prompt_hashes or {},
        dataset_hash=dataset_hash(goldens),
        git_sha=git_sha(),
        seed=seed,
    )
