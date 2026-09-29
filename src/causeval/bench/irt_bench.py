"""B6 -- IRT parameter recovery and item pruning (SPEC 6).

Two claims:
  * on simulated 2PL data the native fit recovers item difficulty/discrimination and system
    ability with correlation >= 0.9;
  * pruning to 30-50% of items (by Fisher information over the systems' abilities) keeps the
    system ranking with Kendall tau >= 0.9.

The simulation draws true ``(a, b, theta)`` and samples a binary ``item x system`` matrix;
the pruning check ranks systems by mean score on the full vs pruned item set.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit
from scipy.stats import kendalltau, pearsonr

from causeval.stats.irt import IRTModel, fit_2pl, prune, rank_systems

RESULTS_DIR = Path("bench/results")


def simulate_2pl(
    n_items: int, n_systems: int, *, seed: int = 0, ability_spread: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw true (a, b, theta) and sample a binary item x system response matrix.

    With ``ability_spread`` set, abilities are evenly spaced over ``[-spread, spread]`` (a set
    of well-separated systems of interest, for the pruning check); otherwise they are drawn
    ``N(0, 1)`` and standardised (for parameter-recovery, which wants a representative spread).
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.6, 2.2, size=n_items)  # discrimination
    b = rng.normal(0.0, 1.0, size=n_items)  # difficulty
    if ability_spread is not None:
        theta = np.linspace(-ability_spread, ability_spread, n_systems)
    else:
        theta = rng.normal(0.0, 1.0, size=n_systems)
        theta = (theta - theta.mean()) / theta.std()  # matches the fit's standardised metric
    z = a[:, None] * (theta[None, :] - b[:, None])
    p = expit(z)
    x = (rng.uniform(size=p.shape) < p).astype(float)
    return x, a, b, theta


def _signed_corr(true: np.ndarray, est: np.ndarray) -> float:
    """Pearson correlation; abs value since IRT is identified only up to a reflection."""
    if np.std(est) < 1e-9 or np.std(true) < 1e-9:
        return 0.0
    return abs(float(pearsonr(true, est)[0]))


def recover(n_items: int = 80, n_systems: int = 500, *, seed: int = 0) -> dict[str, Any]:
    """Fit simulated data and report parameter-recovery correlations.

    Discrimination is the data-hungriest parameter (each item's ``a`` is estimated from
    ``n_systems`` observations), so recovery uses many simulated systems; the SPEC's "5 systems"
    floor is the identifiability minimum, exercised by the pruning check instead.
    """
    x, a, b, theta = simulate_2pl(n_items, n_systems, seed=seed)
    model = fit_2pl(x)
    est_theta = np.asarray(model.ability)
    # resolve the sign flip using ability (the shared axis), then apply it to a/b comparisons.
    flip = np.sign(float(pearsonr(theta, est_theta)[0])) or 1.0
    return {
        "corr_difficulty": _signed_corr(b, flip * np.asarray(model.difficulty)),
        "corr_discrimination": _signed_corr(a, np.asarray(model.discrimination)),
        "corr_ability": _signed_corr(theta, est_theta),
        "converged": model.converged,
        "n_iter": model.n_iter,
    }


def pruning_preserves_ranking(
    n_items: int = 200,
    n_systems: int = 10,
    *,
    keep_frac: float = 0.5,
    seed: int = 0,
    ability_spread: float = 2.5,
) -> dict[str, Any]:
    """Prune to ``keep_frac`` of items by Fisher info; compare system rankings via Kendall tau.

    Uses a handful of well-separated systems (the realistic "systems of interest") and reports
    tau against both the full-set ranking and the true-ability ranking.
    """
    x, _, _, theta = simulate_2pl(n_items, n_systems, seed=seed, ability_spread=ability_spread)
    model = fit_2pl(x)
    kept = prune(model, round(keep_frac * n_items))

    pruned_ranks = rank_systems(x, kept)
    tau_full = float(kendalltau(rank_systems(x), pruned_ranks).statistic)
    # true-ability ranking: higher theta is better (rank 0 = best).
    order = np.argsort(-theta, kind="mergesort")
    true_ranks = np.empty(n_systems, dtype=int)
    true_ranks[order] = np.arange(n_systems)
    tau_true = float(kendalltau(true_ranks, pruned_ranks).statistic)
    return {
        "keep_frac": keep_frac,
        "n_kept": len(kept),
        "kendall_tau": tau_full,
        "kendall_tau_vs_true": tau_true,
    }


def run_offline(*, seed: int = 0) -> tuple[dict[str, Any], Path, Path]:
    rec = recover(seed=seed)
    prune_res = pruning_preserves_ranking(seed=seed)
    payload = {
        "meta": {
            "title": "B6 IRT recovery and pruning (offline synthetic)",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "recovery_systems": 500,
            "pruning_systems": 10,
            "note": "Simulated 2PL data; native joint-MAP fit (L-BFGS).",
        },
        "recovery": rec,
        "pruning": prune_res,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "b6_irt_offline.json"
    md_path = RESULTS_DIR / "b6_irt_offline.md"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(_render_markdown(payload))
    return payload, json_path, md_path


def _render_markdown(payload: dict[str, Any]) -> str:
    m, r, p = payload["meta"], payload["recovery"], payload["pruning"]
    lines = [f"# {m['title']}", ""]
    for k, v in m.items():
        if k != "title":
            lines.append(f"- **{k}**: {v}")
    lines += [
        "",
        "**Parameter recovery (correlation with truth):**",
        f"- difficulty: {r['corr_difficulty']:.3f}",
        f"- discrimination: {r['corr_discrimination']:.3f}",
        f"- ability: {r['corr_ability']:.3f}",
        "",
        f"**Pruning to {int(p['keep_frac'] * 100)}% ({p['n_kept']} items): "
        f"ranking Kendall tau = {p['kendall_tau']:.3f} "
        f"(vs true ability = {p['kendall_tau_vs_true']:.3f})**",
        "",
        "> B6 targets: recovery correlation >= 0.9; pruned-set ranking Kendall tau >= 0.9.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="causeval.bench.irt_bench", description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    payload, json_path, md_path = run_offline(seed=args.seed)
    r, p = payload["recovery"], payload["pruning"]
    print(
        f"recovery: b={r['corr_difficulty']:.3f} a={r['corr_discrimination']:.3f} "
        f"theta={r['corr_ability']:.3f}; prune tau={p['kendall_tau']:.3f}"
    )
    print(f"wrote {json_path} and {md_path}")
    return 0


__all__ = [
    "IRTModel",
    "pruning_preserves_ranking",
    "recover",
    "run_offline",
    "simulate_2pl",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
