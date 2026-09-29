"""Deterministic, cheap checks that run before any judge (SPEC 3.7).

These never call an LLM (except the optional NLI extra) and are the first line of evaluation:
if an output fails a schema or a regex, no judge time is spent on it. Every check returns a
:class:`CheckResult` rather than raising, so a batch of checks can be reported together.

  * :func:`json_schema_check` -- a minimal, dependency-free JSON Schema subset (type / required
    / properties / items), enough for structured-output validation;
  * :func:`regex_check` -- the output matches a pattern;
  * :func:`verifier_check` -- run a user-provided Python predicate;
  * :func:`normalized_equivalence` -- deterministic text-equivalence used as the default
    invariance check for perturbations; live runs inject a bidirectional NLI or judge instead;
  * :func:`nli_equivalence` -- optional local NLI entailment (extra ``causeval[nli]``).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
}


def _validate(value: Any, schema: dict[str, Any], path: str) -> str | None:
    """Return an error string, or ``None`` if ``value`` satisfies ``schema`` at ``path``."""
    expected = schema.get("type")
    if expected is not None:
        py = _JSON_TYPES.get(expected)
        if py is None:
            return f"{path}: unknown schema type {expected!r}"
        # bool is a subclass of int; reject it where a number/integer is required.
        if expected in ("number", "integer") and isinstance(value, bool):
            return f"{path}: expected {expected}, got boolean"
        if not isinstance(value, py):
            return f"{path}: expected {expected}, got {type(value).__name__}"
    if expected == "object":
        for key in schema.get("required", []):
            if key not in value:
                return f"{path}: missing required key {key!r}"
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                err = _validate(value[key], sub, f"{path}.{key}")
                if err:
                    return err
    if expected == "array" and "items" in schema:
        for i, elem in enumerate(value):
            err = _validate(elem, schema["items"], f"{path}[{i}]")
            if err:
                return err
    return None


def json_schema_check(
    output: str, schema: dict[str, Any], *, name: str = "json_schema"
) -> CheckResult:
    """Parse ``output`` as JSON and validate it against a minimal JSON Schema subset."""
    import json

    try:
        value = json.loads(output)
    except (json.JSONDecodeError, TypeError) as exc:
        return CheckResult(name=name, passed=False, detail=f"invalid JSON: {exc}")
    err = _validate(value, schema, "$")
    return CheckResult(name=name, passed=err is None, detail=err or "ok")


def regex_check(
    output: str, pattern: str, *, name: str = "regex", fullmatch: bool = False
) -> CheckResult:
    """Check that ``output`` matches ``pattern`` (search by default, or full match)."""
    rx = re.compile(pattern, re.DOTALL)
    hit = rx.fullmatch(output) if fullmatch else rx.search(output)
    return CheckResult(name=name, passed=hit is not None, detail=pattern)


def verifier_check(
    output: str, predicate: Callable[[str], bool], *, name: str = "verifier"
) -> CheckResult:
    """Run a user predicate; a raised exception is reported as a failed check, never propagated."""
    try:
        ok = bool(predicate(output))
    except Exception as exc:  # a verifier bug is a failed check, not a crash
        return CheckResult(name=name, passed=False, detail=f"verifier raised: {exc}")
    return CheckResult(name=name, passed=ok, detail="ok" if ok else "predicate returned False")


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace -- for deterministic equivalence."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def normalized_equivalence(a: str, b: str) -> bool:
    """Deterministic equivalence: equal after normalization. The default invariance check."""
    return normalize_text(a) == normalize_text(b)


def nli_equivalence(a: str, b: str, *, threshold: float = 0.5) -> bool:  # pragma: no cover
    """Bidirectional NLI entailment via the optional ``causeval[nli]`` extra.

    Two texts are equivalent when each entails the other above ``threshold``. Requires
    ``transformers`` + ``torch``; raises ``ImportError`` with install guidance otherwise.
    """
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "nli_equivalence needs the 'nli' extra: pip install 'causeval[nli]'"
        ) from exc
    clf = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base")

    def entails(premise: str, hypothesis: str) -> float:
        out = clf({"text": premise, "text_pair": hypothesis}, top_k=None)
        scores = {d["label"].lower(): d["score"] for d in out}
        return float(scores.get("entailment", 0.0))

    return entails(a, b) >= threshold and entails(b, a) >= threshold
