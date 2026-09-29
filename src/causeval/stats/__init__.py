"""Statistics: repeated sampling, clustered bootstrap, paired tests, gates, IRT (Phase 1+).

Dependency rule (CLAUDE.md): ``stats`` imports only from ``core``/``adapters`` and never
from ``interventions``, ``judge_audit``, or ``attribution``.
"""

from causeval.stats.compare import compare, compare_pass_rates
from causeval.stats.estimate import cluster_bootstrap_ci, estimate, variance_components
from causeval.stats.gate import GateReport, gate, gate_verdict, holm_adjust
from causeval.stats.planner import (
    PlanOption,
    VarianceForPlanning,
    estimate_variance_for_planning,
    plan_power,
)

__all__ = [
    "GateReport",
    "PlanOption",
    "VarianceForPlanning",
    "cluster_bootstrap_ci",
    "compare",
    "compare_pass_rates",
    "estimate",
    "estimate_variance_for_planning",
    "gate",
    "gate_verdict",
    "holm_adjust",
    "plan_power",
    "variance_components",
]
