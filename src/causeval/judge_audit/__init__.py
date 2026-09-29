"""Judge audit: bias audits, human calibration, PPI estimates, conformal abstention (Phase 3+)."""

from causeval.judge_audit.bias_probes import (
    BiasEstimate,
    PairItem,
    ScoreItem,
    authorship_label,
    markdown_formatting,
    paired_score_effect,
    position_bias,
    verbose_padding,
)
from causeval.judge_audit.calibration import (
    CalibrationReport,
    IsotonicMap,
    calibrate,
    cohen_kappa,
    expected_calibration_error,
)
from causeval.judge_audit.conformal import (
    ConformalThreshold,
    SelectiveResult,
    SelectiveVerdict,
    agreement_confidence,
    apply_threshold,
    calibrate_threshold,
    clopper_pearson_upper,
)
from causeval.judge_audit.jury import (
    ItemVerdict,
    JuryResult,
    convene_jury,
    krippendorff_alpha_interval,
)
from causeval.judge_audit.ppi import PPIEstimate, optimal_lambda, ppi_mean, ppi_mean_ci
from causeval.judge_audit.sampling import (
    LabelingPlan,
    export_labeling_sheet,
    import_labels,
    simple_random_sample,
    stratified_sample,
)

__all__ = [
    "BiasEstimate",
    "CalibrationReport",
    "ConformalThreshold",
    "IsotonicMap",
    "ItemVerdict",
    "JuryResult",
    "LabelingPlan",
    "PPIEstimate",
    "PairItem",
    "ScoreItem",
    "SelectiveResult",
    "SelectiveVerdict",
    "agreement_confidence",
    "apply_threshold",
    "authorship_label",
    "calibrate",
    "calibrate_threshold",
    "clopper_pearson_upper",
    "cohen_kappa",
    "convene_jury",
    "expected_calibration_error",
    "export_labeling_sheet",
    "import_labels",
    "krippendorff_alpha_interval",
    "markdown_formatting",
    "optimal_lambda",
    "paired_score_effect",
    "position_bias",
    "ppi_mean",
    "ppi_mean_ci",
    "simple_random_sample",
    "stratified_sample",
    "verbose_padding",
]
