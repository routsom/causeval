"""Deterministic checks: schema, NLI, metamorphic relations (Phase 4+)."""

from causeval.checks.deterministic import (
    CheckResult,
    json_schema_check,
    nli_equivalence,
    normalize_text,
    normalized_equivalence,
    regex_check,
    verifier_check,
)
from causeval.checks.relations import (
    Example,
    MetamorphicRelation,
    RelationKind,
    all_relations,
    attribute_swap_relation,
    distractor_relation,
    get,
    register,
)

__all__ = [
    "CheckResult",
    "Example",
    "MetamorphicRelation",
    "RelationKind",
    "all_relations",
    "attribute_swap_relation",
    "distractor_relation",
    "get",
    "json_schema_check",
    "nli_equivalence",
    "normalize_text",
    "normalized_equivalence",
    "regex_check",
    "register",
    "verifier_check",
]
