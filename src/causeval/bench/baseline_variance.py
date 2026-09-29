"""Baseline variance benchmark (SPEC §5 Phase 0 deliverable).

Runs a fixed dataset through several DeepEval metrics with ``R`` repeats each and reports,
per metric, the within-item variance of the score and the fraction of items whose pass/fail
verdict flips across repeats. This quantifies the problem causeval solves: a single DeepEval
sample is a noisy draw, and CI thresholds flip on that noise.

Two entry points:
  * :func:`run_live`     — real judge calls (needs API keys); the committed deliverable.
  * :func:`run_offline`  — a seeded synthetic judge; no network, used to smoke-test the
    analysis and report rendering (and runs in CI via a test).

Rule 6: model names are never hardcoded in logic; the live run takes ``--model`` from the
CLI. Rule 4: the report carries a CI on every mean, not a bare score.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from causeval.adapters.metric_factory import MetricSpec, a_measure_once
from causeval.core.schemas import Measurement

RESULTS_DIR = Path("bench/results")

# Metrics probed in the baseline. Class names are resolved by the metric factory; the judge
# model is injected per run (never hardcoded here).
DEFAULT_METRICS = ["AnswerRelevancyMetric", "FaithfulnessMetric", "ContextualRelevancyMetric"]


@dataclass(frozen=True)
class Item:
    item_id: str
    input: str
    actual_output: str
    retrieval_context: list[str]


def build_dataset(n: int = 30) -> list[Item]:
    """A small fictional-knowledge-base QA set, generated so no literal is huge.

    The facts are invented (a fictional company "Northwind Robotics") so a model cannot
    answer them from memory; the answer is grounded in ``retrieval_context``.
    """
    rng = np.random.default_rng(0)
    departments = ["Autonomy", "Perception", "Fleet Ops", "Safety", "Hardware"]
    items: list[Item] = []
    for i in range(n):
        dept = departments[i % len(departments)]
        year = 2018 + (i % 7)
        headcount = int(rng.integers(12, 240))
        fact = (
            f"Northwind Robotics' {dept} team was founded in {year} "
            f"and now has {headcount} engineers."
        )
        question = f"When was Northwind Robotics' {dept} team founded and how large is it?"
        answer = f"The {dept} team was founded in {year} and has {headcount} engineers."
        items.append(
            Item(
                item_id=f"item{i:02d}",
                input=question,
                actual_output=answer,
                retrieval_context=[fact],
            )
        )
    return items


@dataclass(frozen=True)
class MetricBaseline:
    """Per-metric baseline summary. Carries a CI on the mean (rule 4)."""

    metric: str
    n_items: int
    n_repeats: int
    n_failed: int
    mean_score: float
    ci_low: float
    ci_high: float
    ci_method: str
    mean_within_item_variance: float
    flip_fraction: float  # fraction of items whose pass/fail flips across repeats
    n_flaky_items: int  # 0.2 < pass-rate < 0.8


def _bootstrap_mean_ci(
    item_means: np.ndarray, *, level: float = 0.95, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of per-item means (Phase-0 stand-in).

    The full clustered-bootstrap estimator lives in ``stats`` (Phase 1); this keeps the
    baseline honest (no bare mean) without importing unbuilt code.
    """
    if item_means.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = item_means.size
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = float(item_means[idx].mean())
    lo = float(np.quantile(boots, (1 - level) / 2))
    hi = float(np.quantile(boots, 1 - (1 - level) / 2))
    return (lo, hi)


def analyze(
    measurements: list[Measurement],
    *,
    threshold: float,
    flaky_low: float = 0.2,
    flaky_high: float = 0.8,
) -> list[MetricBaseline]:
    """Compute per-metric within-item variance and pass/fail flip fraction.

    A measurement with an ``error`` is excluded and counted in ``n_failed``. Pass/fail is
    derived from ``score >= threshold`` when ``passed`` is not set, so the analysis works
    for synthetic measurements too.
    """
    by_metric: dict[str, list[Measurement]] = defaultdict(list)
    for m in measurements:
        by_metric[m.metric].append(m)

    baselines: list[MetricBaseline] = []
    for metric, ms in sorted(by_metric.items()):
        # group by item
        scores_by_item: dict[str, list[float]] = defaultdict(list)
        passes_by_item: dict[str, list[bool]] = defaultdict(list)
        n_failed = 0
        for m in ms:
            if m.error is not None or m.score is None:
                n_failed += 1
                continue
            scores_by_item[m.item_id].append(m.score)
            passed = m.passed if m.passed is not None else (m.score >= threshold)
            passes_by_item[m.item_id].append(bool(passed))

        item_means = np.array([float(np.mean(v)) for v in scores_by_item.values()])
        within_vars = [float(np.var(v, ddof=1)) for v in scores_by_item.values() if len(v) > 1]
        mean_within = float(np.mean(within_vars)) if within_vars else 0.0

        n_repeats = max((len(v) for v in scores_by_item.values()), default=0)
        # flip = an item shows both a pass and a fail across its repeats
        flips = sum(1 for v in passes_by_item.values() if len(set(v)) > 1)
        n_items = len(scores_by_item)
        flip_fraction = flips / n_items if n_items else 0.0
        n_flaky = sum(
            1 for v in passes_by_item.values() if flaky_low < (sum(v) / len(v)) < flaky_high
        )

        mean_score = float(item_means.mean()) if item_means.size else float("nan")
        lo, hi = _bootstrap_mean_ci(item_means)
        baselines.append(
            MetricBaseline(
                metric=metric,
                n_items=n_items,
                n_repeats=n_repeats,
                n_failed=n_failed,
                mean_score=mean_score,
                ci_low=lo,
                ci_high=hi,
                ci_method="cluster_bootstrap_percentile_B=2000",
                mean_within_item_variance=mean_within,
                flip_fraction=flip_fraction,
                n_flaky_items=n_flaky,
            )
        )
    return baselines


def render_markdown(baselines: list[MetricBaseline], *, title: str, meta: dict[str, str]) -> str:
    lines = [f"# {title}", ""]
    for k, v in meta.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    header = (
        "| metric | n | R | mean [95% CI] | mean within-item var | "
        "pass/fail flip frac | flaky items | failed |"
    )
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|")
    for b in baselines:
        lines.append(
            f"| {b.metric} | {b.n_items} | {b.n_repeats} | "
            f"{b.mean_score:.3f} [{b.ci_low:.3f}, {b.ci_high:.3f}] | "
            f"{b.mean_within_item_variance:.4f} | {b.flip_fraction:.2f} | "
            f"{b.n_flaky_items} | {b.n_failed} |"
        )
    lines.append("")
    lines.append(
        "> A nonzero flip fraction means a single DeepEval sample would pass or fail the "
        "same item depending on luck. This is the motivation for repeated sampling and CIs."
    )
    lines.append("")
    return "\n".join(lines)


def write_report(
    baselines: list[MetricBaseline], *, name: str, meta: dict[str, str]
) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"{name}.json"
    md_path = RESULTS_DIR / f"{name}.md"
    payload = {"meta": meta, "baselines": [asdict(b) for b in baselines]}
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(render_markdown(baselines, title=meta.get("title", name), meta=meta))
    return json_path, md_path


async def _measure_all(
    items: list[Item], specs: list[MetricSpec], *, repeats: int, threshold: float
) -> list[Measurement]:
    from causeval.adapters.deepeval_import import LLMTestCase

    tasks = []
    for it in items:
        tc = LLMTestCase(
            input=it.input,
            actual_output=it.actual_output,
            retrieval_context=it.retrieval_context,
        )
        for spec in specs:
            for r in range(repeats):
                tasks.append(a_measure_once(spec, tc, item_id=it.item_id, repeat_index=r))
    return list(await asyncio.gather(*tasks))


def run_live(
    *, model: str, repeats: int = 10, n_items: int = 30, threshold: float = 0.7
) -> tuple[Path, Path]:
    """Live baseline: real judge calls with ``model``. Needs provider API keys."""
    items = build_dataset(n_items)
    specs = [
        MetricSpec(name=name, kwargs={"model": model, "threshold": threshold})
        for name in DEFAULT_METRICS
    ]
    measurements = asyncio.run(_measure_all(items, specs, repeats=repeats, threshold=threshold))
    baselines = analyze(measurements, threshold=threshold)
    meta = {
        "title": "Baseline variance (live)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "judge_model": model,
        "repeats": str(repeats),
        "threshold": str(threshold),
    }
    return write_report(baselines, name="baseline_live", meta=meta)


def _synthetic_measurements(
    items: list[Item], *, repeats: int, threshold: float, seed: int
) -> list[Measurement]:
    """A seeded synthetic judge near the threshold, to exercise the analysis offline."""
    rng = np.random.default_rng(seed)
    out: list[Measurement] = []
    for metric in DEFAULT_METRICS:
        for it in items:
            true_q = float(rng.uniform(0.55, 0.85))  # near the 0.7 threshold on purpose
            for r in range(repeats):
                score = float(np.clip(true_q + rng.normal(0, 0.12), 0, 1))
                out.append(
                    Measurement(
                        item_id=it.item_id,
                        metric=metric,
                        repeat_index=r,
                        score=score,
                        passed=score >= threshold,
                    )
                )
    return out


def run_offline(
    *, repeats: int = 10, n_items: int = 30, threshold: float = 0.7, seed: int = 0
) -> tuple[Path, Path]:
    """Offline smoke: synthetic judge, no network. Proves the analysis + report render."""
    items = build_dataset(n_items)
    measurements = _synthetic_measurements(items, repeats=repeats, threshold=threshold, seed=seed)
    baselines = analyze(measurements, threshold=threshold)
    meta = {
        "title": "Baseline variance (offline synthetic smoke)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "judge_model": "synthetic (seeded)",
        "repeats": str(repeats),
        "threshold": str(threshold),
        "note": "Synthetic data; not a real judge. Use run_live for the committed deliverable.",
    }
    return write_report(baselines, name="baseline_offline_smoke", meta=meta)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="causeval.bench", description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="run the seeded synthetic smoke instead of live judge calls",
    )
    parser.add_argument("--model", help="judge model for the live run (e.g. gpt-4o-mini)")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--n-items", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args(argv)

    if args.offline:
        json_path, md_path = run_offline(
            repeats=args.repeats, n_items=args.n_items, threshold=args.threshold
        )
    else:
        if not args.model:
            parser.error("--model is required for a live run (or pass --offline)")
        json_path, md_path = run_live(
            model=args.model,
            repeats=args.repeats,
            n_items=args.n_items,
            threshold=args.threshold,
        )
    print(f"wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
