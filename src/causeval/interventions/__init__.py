"""Interventions: RAG context ablation + counterfactual context, perturbations, CoT (Phase 2+)."""

from causeval.interventions.cf_gen import (
    Counterfactual,
    build_counterfactual,
    generate_counterfactual,
    load_cf_overrides,
)
from causeval.interventions.rag import (
    GroundingItem,
    GroundingResult,
    ItemGrounding,
    RAGApp,
    a_ground,
    classify_item,
    contains_outcome,
    follows_cf_outcome,
    ground,
)

__all__ = [
    "Counterfactual",
    "GroundingItem",
    "GroundingResult",
    "ItemGrounding",
    "RAGApp",
    "a_ground",
    "build_counterfactual",
    "classify_item",
    "contains_outcome",
    "follows_cf_outcome",
    "generate_counterfactual",
    "ground",
    "load_cf_overrides",
]
