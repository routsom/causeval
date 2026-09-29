"""Enforce CLAUDE.md rule 2: deepeval is imported only in adapters/deepeval_import.py,
and the telemetry / dotenv opt-outs are set before the first import.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "causeval"
GUARD = SRC / "adapters" / "deepeval_import.py"

_DEEPEVAL_IMPORT = re.compile(r"^\s*(?:import\s+deepeval|from\s+deepeval)\b", re.MULTILINE)


def test_only_guard_imports_deepeval() -> None:
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        if path == GUARD:
            continue
        text = path.read_text(encoding="utf-8")
        if _DEEPEVAL_IMPORT.search(text):
            offenders.append(str(path.relative_to(SRC)))
    assert not offenders, (
        f"these modules import deepeval directly; route through "
        f"adapters.deepeval_import instead: {offenders}"
    )


def test_guard_sets_optouts_before_import() -> None:
    # Importing the guard must leave both opt-outs set.
    import causeval.adapters.deepeval_import  # noqa: F401

    assert os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] == "1"
    assert os.environ["DEEPEVAL_DISABLE_DOTENV"] == "1"


def test_guard_uses_setdefault_not_overwrite() -> None:
    # The guard must use setdefault so a user's explicit value is respected.
    text = GUARD.read_text(encoding="utf-8")
    assert 'setdefault("DEEPEVAL_TELEMETRY_OPT_OUT"' in text
    assert 'setdefault("DEEPEVAL_DISABLE_DOTENV"' in text
    # The setdefault calls must appear before any deepeval import in the file.
    first_import = _DEEPEVAL_IMPORT.search(text)
    assert first_import is not None
    assert text.index("setdefault") < first_import.start()
