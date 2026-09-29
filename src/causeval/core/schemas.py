"""Public result schemas.

All public results are pydantic v2 models, JSON-serializable, and carry a
``schema_version``. Per CLAUDE.md rule 4, no public result reports a bare score: an
estimate always carries ``n_items``, ``n_repeats``, a confidence interval, the method
used, and provenance.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "0.1"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Provenance(BaseModel):
    """Everything needed to reproduce and trust a run."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    causeval_version: str
    deepeval_version: str
    judge_model: str | None = None
    judge_params: dict[str, object] = Field(default_factory=dict)
    # metric name -> hash of its prompt templates, where available
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    dataset_hash: str
    git_sha: str | None = None
    seed: int
    created_at: datetime = Field(default_factory=_utcnow)


class Measurement(BaseModel):
    """One metric applied to one item on one repeat under one condition.

    ``score`` is ``None`` exactly when ``error`` is set: a failed judge call is recorded,
    never replaced by a default score.
    """

    schema_version: str = SCHEMA_VERSION
    item_id: str
    metric: str
    # e.g. "base", "ctx_none", "loo_2", "cf", "perturb:paraphrase"
    condition: str = "base"
    repeat_index: int
    score: float | None = None
    passed: bool | None = None
    reason: str | None = None
    # set when the judge/app call failed; score is then None
    error: str | None = None
    cost_usd: float | None = None
    latency_s: float | None = None
    # free-form per-measurement flags, e.g. {"nondeterministic_steps": True}
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.error is not None


class ScoreEstimate(BaseModel):
    """A metric's estimate under one condition, with uncertainty.

    Estimand (default): ``mu = E_item[ E_repeat[ y ] ]``, the expected score of a randomly
    drawn item under a random judge/app sample. Estimator: ``mean_i(mean_r y_ir)``.
    """

    schema_version: str = SCHEMA_VERSION
    metric: str
    condition: str = "base"
    estimate: float
    ci_low: float
    ci_high: float
    ci_level: float = 0.95
    method: str  # e.g. "cluster_bootstrap_percentile_B=2000"
    n_items: int
    n_repeats: int
    n_failed: int = 0  # measurements excluded because of errors
    var_between: float = 0.0  # between-item variance of item means
    var_within: float = 0.0  # mean within-item (judge/app noise) variance
    icc: float = 0.0  # var_between / (var_between + var_within)
    flaky_items: list[str] = Field(default_factory=list)  # 0.2 < pass rate < 0.8
    small_sample: bool = False
    t_ci_low: float | None = None
    t_ci_high: float | None = None


class Comparison(BaseModel):
    """A paired candidate-vs-baseline comparison for one metric."""

    schema_version: str = SCHEMA_VERSION
    metric: str
    delta: float  # candidate - baseline
    ci_low: float
    ci_high: float
    ci_level: float = 0.95
    p_value: float | None = None
    p_adjusted: float | None = None  # Holm across metrics in the same comparison
    method: str
    verdict: Literal["pass", "regression", "inconclusive"]
    margin: float
    n_items: int
    n_dropped: int = 0  # items dropped when allow_partial intersected the id sets


class RunResult(BaseModel):
    """The canonical artifact of a run. Stored as JSON; Markdown is rendered from it."""

    schema_version: str = SCHEMA_VERSION
    name: str = "run"
    provenance: Provenance
    measurements: list[Measurement] = Field(default_factory=list)
    estimates: list[ScoreEstimate] = Field(default_factory=list)
