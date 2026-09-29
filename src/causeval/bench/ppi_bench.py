"""B4 -- prediction-powered inference benchmark (SPEC 6).

Claim: PPI's CI covers the true human mean at ~95% while borrowing the judge's precision,
whereas naively averaging the (biased) judge scores has poor coverage.

Setup: a fixed finite population of ``N`` items with known human labels ``Y`` (so the estimand
``theta = mean(Y)`` is known exactly). A synthetic judge predicts ``f = clip(Y + bias + noise)``
with a *systematic* positive bias. Each simulation labels a fresh random subset of ``n`` items
(the labeling is the only randomness) and compares three CIs for ``theta``:

  * **PPI** (uses labeled Y+f and unlabeled f) -- expected to cover ~95% and be narrow;
  * **human-only** (mean of the ``n`` labels) -- covers ~95% but wide;
  * **naive judge mean** (mean of all judge scores) -- tight but biased, so covers poorly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sps

from causeval.judge_audit.ppi import PPIEstimate, ppi_mean_ci

RESULTS_DIR = Path("bench/results")

JUDGE_BIAS = 0.1


@dataclass(frozen=True)
class Population:
    y: np.ndarray  # human labels for every item
    f: np.ndarray  # judge score for every item

    @property
    def theta(self) -> float:
        return float(self.y.mean())


def build_population(n_total: int = 1000, *, bias: float = JUDGE_BIAS, seed: int = 0) -> Population:
    """Fixed population: Y ~ U(0,1) rounded to a quality; f = clip(Y + bias + noise)."""
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.0, 1.0, size=n_total)
    f = np.clip(y + bias + rng.normal(0.0, 0.15, size=n_total), 0.0, 1.0)
    return Population(y=y, f=f)


def naive_judge_ci(f_all: np.ndarray, *, level: float = 0.95) -> tuple[float, float]:
    """Mean of judge scores over all items with a normal CI -- tight but biased."""
    mean = float(f_all.mean())
    se = float(f_all.std(ddof=1) / np.sqrt(f_all.size))
    z = float(sps.norm.ppf(1 - (1 - level) / 2))
    return (mean - z * se, mean + z * se)


def one_draw(
    pop: Population, n_labeled: int, *, rng: np.random.Generator, level: float = 0.95
) -> tuple[PPIEstimate, tuple[float, float]]:
    """Label a random subset of ``n_labeled`` items; return (ppi, naive_ci)."""
    idx = rng.permutation(pop.y.size)
    lab, unlab = idx[:n_labeled], idx[n_labeled:]
    ppi = ppi_mean_ci(pop.y[lab], pop.f[lab], pop.f[unlab], level=level)
    naive = naive_judge_ci(pop.f, level=level)
    return ppi, naive


def run_offline(
    *, n_total: int = 1000, n_labeled: int = 100, seed: int = 0
) -> tuple[dict[str, Any], Path, Path]:
    """Single-draw B4 report plus a small coverage summary."""
    pop = build_population(n_total, seed=seed)
    rng = np.random.default_rng(seed)
    ppi, naive = one_draw(pop, n_labeled, rng=rng)

    # small coverage summary so the committed report shows the headline numbers.
    cov = coverage(n_total=n_total, n_labeled=n_labeled, n_sims=200, seed=seed)

    payload = {
        "meta": {
            "title": "B4 prediction-powered inference (offline synthetic)",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_total": n_total,
            "n_labeled": n_labeled,
            "judge_bias": JUDGE_BIAS,
            "theta_true": pop.theta,
            "note": "Synthetic judge with systematic bias; not a real LLM.",
        },
        "example_draw": {
            "ppi_estimate": ppi.estimate,
            "ppi_ci": [ppi.ci_low, ppi.ci_high],
            "ppi_ci_width": ppi.ci_width,
            "human_only_ci": [ppi.human_only_ci_low, ppi.human_only_ci_high],
            "human_only_ci_width": ppi.human_only_ci_width,
            "naive_judge_ci": list(naive),
            "lambda": ppi.lam,
            "effective_n": ppi.effective_n,
        },
        "coverage": cov,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "b4_ppi_offline.json"
    md_path = RESULTS_DIR / "b4_ppi_offline.md"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(_render_markdown(payload))
    return payload, json_path, md_path


def coverage(
    *, n_total: int = 1000, n_labeled: int = 100, n_sims: int = 200, seed: int = 0
) -> dict[str, float]:
    """Coverage of theta and mean CI widths for PPI, human-only, and naive, over sims."""
    pop = build_population(n_total, seed=seed)
    theta = pop.theta
    rng = np.random.default_rng(seed + 1)
    ppi_hits = ho_hits = naive_hits = 0
    ppi_w: list[float] = []
    ho_w: list[float] = []
    for _ in range(n_sims):
        ppi, naive = one_draw(pop, n_labeled, rng=rng)
        ppi_hits += int(ppi.ci_low <= theta <= ppi.ci_high)
        ho_hits += int(ppi.human_only_ci_low <= theta <= ppi.human_only_ci_high)
        naive_hits += int(naive[0] <= theta <= naive[1])
        ppi_w.append(ppi.ci_width)
        ho_w.append(ppi.human_only_ci_width)
    return {
        "ppi_coverage": ppi_hits / n_sims,
        "human_only_coverage": ho_hits / n_sims,
        "naive_coverage": naive_hits / n_sims,
        "ppi_mean_width": float(np.mean(ppi_w)),
        "human_only_mean_width": float(np.mean(ho_w)),
        "n_sims": n_sims,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    m = payload["meta"]
    c = payload["coverage"]
    lines = [f"# {m['title']}", ""]
    for k, v in m.items():
        if k != "title":
            lines.append(f"- **{k}**: {v}")
    lines += [
        "",
        f"**theta (true human mean) = {m['theta_true']:.4f}**",
        "",
        "| estimator | coverage of theta | mean 95% CI width |",
        "|---|---|---|",
        f"| PPI | {c['ppi_coverage']:.3f} | {c['ppi_mean_width']:.4f} |",
        f"| human-only | {c['human_only_coverage']:.3f} | {c['human_only_mean_width']:.4f} |",
        f"| naive judge mean | {c['naive_coverage']:.3f} | (biased) |",
        "",
        "> B4 target: PPI covers theta ~95% and is narrower than human-only at equal label "
        "count; the naive judge mean covers poorly because of its systematic bias.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="causeval.bench.ppi_bench", description=__doc__)
    parser.add_argument("--n-total", type=int, default=1000)
    parser.add_argument("--n-labeled", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    payload, json_path, md_path = run_offline(
        n_total=args.n_total, n_labeled=args.n_labeled, seed=args.seed
    )
    c = payload["coverage"]
    print(f"PPI coverage={c['ppi_coverage']:.3f}  naive coverage={c['naive_coverage']:.3f}")
    print(f"PPI width={c['ppi_mean_width']:.4f}  human-only width={c['human_only_mean_width']:.4f}")
    print(f"wrote {json_path} and {md_path}")
    return 0


__all__ = ["PPIEstimate", "Population", "build_population", "coverage", "one_draw", "run_offline"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
