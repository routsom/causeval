"""causeval command-line interface.

Phase 1 commands:
  * ``run``     -- run an experiment from a config; write a RunResult JSON.
  * ``compare`` -- paired candidate-vs-baseline comparison from two RunResult JSONs.
  * ``gate``    -- like ``compare`` but set the process exit code (0 pass, 1 regression,
                   2 inconclusive).
  * ``plan``    -- sample-size / power planning from a pilot pair.

The intervention/audit/attribute commands arrive in later phases (SPEC section 4).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from causeval import __version__
from causeval.adapters.metric_factory import MetricSpec
from causeval.core.schemas import Comparison, RunResult
from causeval.experiment import Experiment
from causeval.interventions.cf_gen import Counterfactual, load_cf_overrides
from causeval.interventions.rag import GroundingItem, ground
from causeval.stats.compare import compare
from causeval.stats.gate import gate
from causeval.stats.planner import estimate_variance_for_planning, plan_power

# ---- run persistence ------------------------------------------------------------


def load_run(path: str | Path) -> RunResult:
    return RunResult.model_validate_json(Path(path).read_text())


def save_run(run: RunResult, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(run.model_dump_json(indent=2))


# ---- config loading for `run` ---------------------------------------------------


def _load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"config {path!s} must be a mapping")
    return raw


def _load_dataset(spec: Any, base: Path) -> list[dict[str, Any]]:
    """Dataset is a path to a JSONL file (one item per line) or an inline list."""
    if isinstance(spec, list):
        return spec
    dataset_path = base / spec if not Path(spec).is_absolute() else Path(spec)
    items = []
    for line in dataset_path.read_text().splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def _resolve_callable(dotted: str) -> Any:
    module_name, _, attr = dotted.partition(":")
    if not attr:
        raise ValueError(f"expected 'module:function', got {dotted!r}")
    return getattr(importlib.import_module(module_name), attr)


def _build_metric_specs(entries: list[dict[str, Any]]) -> list[MetricSpec]:
    """Build specs from config entries.

    Each entry is either ``{name: <DeepEvalMetricClass>, kwargs: {...}}`` or
    ``{builder: "module:function"}`` where the function is a zero-arg callable returning a
    fresh metric instance (useful for custom or offline metrics).
    """
    specs = []
    for e in entries:
        if e.get("builder"):
            specs.append(MetricSpec(builder=_resolve_callable(e["builder"]), label=e.get("label")))
        else:
            specs.append(
                MetricSpec(name=e["name"], kwargs=e.get("kwargs", {}), label=e.get("label"))
            )
    return specs


def _counterfactual_from(d: dict[str, Any]) -> Counterfactual:
    original = str(d["original_value"])
    cf_value = str(d["cf_value"])
    chunk_index = int(d.get("chunk_index", 0))
    edited = d.get("edited_chunk")
    return Counterfactual(
        chunk_index=chunk_index,
        original_value=original,
        cf_value=cf_value,
        edited_chunk=str(edited) if edited is not None else "",
    )


def _load_grounding_dataset(
    items: list[dict[str, Any]], overrides: dict[str, Counterfactual]
) -> list[GroundingItem]:
    dataset: list[GroundingItem] = []
    for it in items:
        item_id = str(it["item_id"])
        cf = overrides.get(item_id)
        if cf is None and it.get("cf"):
            cf = _counterfactual_from(it["cf"])
        dataset.append(
            GroundingItem(
                item_id=item_id,
                question=str(it["question"]),
                contexts=list(it["contexts"]),
                expected_output=it.get("expected_output"),
                cf=cf,
            )
        )
    return dataset


def _resolve_app(dotted: str) -> Any:
    """Resolve a RAGApp: a dotted path to an app object, or a zero-arg factory for one."""
    obj = _resolve_callable(dotted)
    if hasattr(obj, "answer"):
        return obj
    return obj()  # a factory returning a RAGApp


# ---- command handlers -----------------------------------------------------------


def cmd_ground(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    base = Path(args.config).resolve().parent
    overrides: dict[str, Counterfactual] = {}
    if config.get("cf_overrides"):
        cf_path = base / config["cf_overrides"]
        overrides = load_cf_overrides(cf_path)
    dataset = _load_grounding_dataset(_load_dataset(config["dataset"], base), overrides)
    app = _resolve_app(config["app"])

    faith_spec = None
    if config.get("faithfulness"):
        f = config["faithfulness"]
        faith_spec = MetricSpec(name=f["name"], kwargs=f.get("kwargs", {}))

    result = ground(
        app,
        dataset,
        repeats=int(config.get("repeats", 5)),
        seed=int(config.get("seed", 0)),
        tau=float(config.get("tau", 0.5)),
        conflict_policy=config.get("conflict_policy", "follow_context"),
        faithfulness_spec=faith_spec,
        level=float(config.get("level", 0.95)),
        n_boot=int(config.get("n_boot", 2000)),
    )

    name = config.get("name", "grounding")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.out) / f"{stamp}_{name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.model_dump_json(indent=2))
    print(result.summary())
    print(f"\nwrote {out_path}")
    return 0


def cmd_audit_judge(args: argparse.Namespace) -> int:
    """Calibrate a judge against human labels, and (if unlabeled scores are given) run PPI.

    Config keys:
      * ``labeled``   -- JSONL of ``{item_id, judge_score, human_label}`` (required);
      * ``unlabeled`` -- JSONL of ``{item_id, judge_score}`` (optional; enables PPI);
      * ``threshold`` -- pass threshold for calibration (default 0.5);
      * ``seed``      -- RNG seed for the ECE/PPI bootstrap (default 0);
      * ``i_know_the_labels_are_random`` -- bypass PPI's random-sample guard (default false).
    """
    from causeval.judge_audit.calibration import calibrate
    from causeval.judge_audit.ppi import ppi_mean

    config = _load_config(args.config)
    base = Path(args.config).resolve().parent
    seed = int(config.get("seed", 0))
    threshold = float(config.get("threshold", 0.5))

    labeled = _load_dataset(config["labeled"], base)
    labeled_ids = [str(r["item_id"]) for r in labeled]
    judge_scores = [float(r["judge_score"]) for r in labeled]
    human_labels = [float(r["human_label"]) for r in labeled]

    report = calibrate(
        judge_scores,
        human_labels,
        threshold=threshold,
        rng=np.random.default_rng(seed),
    )
    output: dict[str, Any] = {"calibration": report.model_dump(mode="json")}
    print(report.summary())

    if config.get("unlabeled"):
        unlabeled = _load_dataset(config["unlabeled"], base)
        f_unlabeled = [float(r["judge_score"]) for r in unlabeled]
        plan_ids = labeled_ids + [str(r["item_id"]) for r in unlabeled]
        ppi = ppi_mean(
            human_labels,
            judge_scores,
            f_unlabeled,
            labeled_ids=labeled_ids,
            sampling_plan_ids=plan_ids,
            i_know_the_labels_are_random=bool(config.get("i_know_the_labels_are_random", False)),
            level=float(config.get("level", 0.95)),
        )
        output["ppi"] = ppi.model_dump(mode="json")
        print(
            f"PPI human-mean: {ppi.estimate:.3f} "
            f"[{int(ppi.ci_level * 100)}% CI {ppi.ci_low:.3f}, {ppi.ci_high:.3f}] "
            f"(lambda={ppi.lam:.3f}, effective_n={ppi.effective_n:.1f} vs {ppi.n_labeled} labels)"
        )

    name = config.get("name", "judge_audit")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.out) / f"{stamp}_{name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nwrote {out_path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    base = Path(args.config).resolve().parent
    dataset = _load_dataset(config["dataset"], base)
    metrics = _build_metric_specs(config["metrics"])
    app = _resolve_callable(config["app"]) if config.get("app") else None
    name = config.get("name", "run")

    exp = Experiment(
        dataset=dataset,
        metrics=metrics,
        repeats=int(config.get("repeats", 5)),
        seed=int(config.get("seed", 0)),
        app=app,
        name=name,
        judge_model=config.get("judge_model"),
    )
    run = exp.run()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out)
    out_path = out_dir / f"{stamp}_{name}.json"
    save_run(run, out_path)
    print(run.summary())
    print(f"\nwrote {out_path}")
    return 0


def _render_comparisons(comparisons: list[Comparison]) -> str:
    lines = ["metric                         delta      CI (level)                verdict"]
    for c in comparisons:
        lines.append(
            f"{c.metric:<30.30} {c.delta:+.3f}   "
            f"[{c.ci_low:+.3f}, {c.ci_high:+.3f}] @{int(c.ci_level * 100)}%   "
            f"{c.verdict}" + (f"  (dropped {c.n_dropped})" if c.n_dropped else "")
        )
    return "\n".join(lines)


def _run_comparison(args: argparse.Namespace) -> list[Comparison]:
    baseline = load_run(args.baseline)
    candidate = load_run(args.candidate)
    rng = np.random.default_rng(args.seed)
    return compare(
        baseline.measurements,
        candidate.measurements,
        metrics=args.metrics,
        margin=args.margin,
        level=args.level,
        n_boot=args.n_boot,
        rng=rng,
        allow_partial=args.allow_partial,
        family_correction=args.family_correction,
    )


def cmd_compare(args: argparse.Namespace) -> int:
    comparisons = _run_comparison(args)
    print(_render_comparisons(comparisons))
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    comparisons = _run_comparison(args)
    report = gate(comparisons)
    print(_render_comparisons(comparisons))
    print(f"\noverall: {report.overall}  (exit {report.exit_code})")
    return report.exit_code


def cmd_plan(args: argparse.Namespace) -> int:
    baseline = load_run(args.pilot_baseline)
    candidate = load_run(args.pilot_candidate)
    variance = estimate_variance_for_planning(
        baseline.measurements, candidate.measurements, metric=args.metric
    )
    r_options = tuple(int(x) for x in args.repeats_options.split(","))
    options = plan_power(
        delta=args.detect,
        variance=variance,
        power=args.power,
        level=args.level,
        r_options=r_options,
        cost_per_call_usd=args.cost_per_call,
    )
    print(f"Plan to detect delta={args.detect} at power={args.power}, level={args.level}")
    print(f"metric={args.metric}  (variance: {variance})")
    print("   n     R    exp. CI half-width   calls     est. cost")
    for o in options:
        cost = "n/a" if o.est_cost_usd is None else f"${o.est_cost_usd:.4f}"
        print(
            f"  {o.n_items:>4}  {o.n_repeats:>4}   {o.expected_ci_halfwidth:>16.4f}   "
            f"{o.total_calls:>6}   {cost}"
        )
    return 0


# ---- parser ---------------------------------------------------------------------


def _add_compare_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--baseline", required=True, help="baseline RunResult JSON")
    p.add_argument("--candidate", required=True, help="candidate RunResult JSON")
    p.add_argument("--margin", type=float, required=True, help="non-inferiority tolerance")
    p.add_argument("--metrics", nargs="*", default=None, help="restrict to these metrics")
    p.add_argument("--level", type=float, default=0.95)
    p.add_argument("--n-boot", type=int, default=2000, dest="n_boot")
    p.add_argument("--allow-partial", action="store_true", dest="allow_partial")
    p.add_argument(
        "--family-correction",
        choices=["holm", "none"],
        default="holm",
        dest="family_correction",
    )
    p.add_argument("--seed", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="causeval", description=__doc__)
    parser.add_argument("--version", action="version", version=f"causeval {__version__}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("version", help="print the causeval version")

    p_run = sub.add_parser("run", help="run an experiment from a config")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--out", default="runs")
    p_run.set_defaults(func=cmd_run)

    p_ground = sub.add_parser("ground", help="run RAG causal grounding from a config")
    p_ground.add_argument("--config", required=True)
    p_ground.add_argument("--out", default="runs")
    p_ground.set_defaults(func=cmd_ground)

    p_audit = sub.add_parser("audit-judge", help="calibrate a judge vs human labels (+ PPI)")
    p_audit.add_argument("--config", required=True)
    p_audit.add_argument("--out", default="runs")
    p_audit.set_defaults(func=cmd_audit_judge)

    p_cmp = sub.add_parser("compare", help="compare two runs")
    _add_compare_args(p_cmp)
    p_cmp.set_defaults(func=cmd_compare)

    p_gate = sub.add_parser("gate", help="compare two runs and set the exit code")
    _add_compare_args(p_gate)
    p_gate.set_defaults(func=cmd_gate)

    p_plan = sub.add_parser("plan", help="sample-size / power planning from a pilot pair")
    p_plan.add_argument("--pilot-baseline", required=True, dest="pilot_baseline")
    p_plan.add_argument("--pilot-candidate", required=True, dest="pilot_candidate")
    p_plan.add_argument("--metric", required=True)
    p_plan.add_argument("--detect", type=float, required=True, help="effect size to detect")
    p_plan.add_argument("--power", type=float, default=0.8)
    p_plan.add_argument("--level", type=float, default=0.95)
    p_plan.add_argument("--repeats-options", default="1,3,5,10", dest="repeats_options")
    p_plan.add_argument("--cost-per-call", type=float, default=None, dest="cost_per_call")
    p_plan.set_defaults(func=cmd_plan)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in (None, "version"):
        print(f"causeval {__version__}")
        return 0
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
