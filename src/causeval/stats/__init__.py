"""Statistics: repeated sampling, clustered bootstrap, paired tests, gates, IRT (Phase 1+).

Dependency rule (CLAUDE.md): ``stats`` imports only from ``core``/``adapters`` and never
from ``interventions``, ``judge_audit``, or ``attribution``.
"""

from causeval.stats.adaptive import allocate_repeats, repeats_per_item
from causeval.stats.compare import compare, compare_pass_rates
from causeval.stats.estimate import cluster_bootstrap_ci, estimate, variance_components
from causeval.stats.gate import GateReport, gate, gate_verdict, holm_adjust
from causeval.stats.irt import IRTModel, fisher_information, fit_2pl, prune, rank_systems
from causeval.stats.planner import (
    PlanOption,
    VarianceForPlanning,
    estimate_variance_for_planning,
    plan_power,
)

__all__ = [
    "GateReport",
    "IRTModel",
    "PlanOption",
    "VarianceForPlanning",
    "allocate_repeats",
    "cluster_bootstrap_ci",
    "compare",
    "compare_pass_rates",
    "estimate",
    "estimate_variance_for_planning",
    "fisher_information",
    "fit_2pl",
    "gate",
    "gate_verdict",
    "holm_adjust",
    "plan_power",
    "prune",
    "rank_systems",
    "repeats_per_item",
    "variance_components",
]
