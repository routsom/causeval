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


# ---- command handlers -----------------------------------------------------------


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
