"""B3 -- judge bias recovery benchmark (SPEC 6).

Claim: the bias probes recover a known injected bias within their CI. We inject a position
bias of 0.15 (a pairwise judge over-picks the first-presented answer) and a verbosity bias of
+0.08 (a pointwise judge rewards padded answers), and check that:

  * the position and verbosity probes recover the injected values inside their CIs, with ~95%
    coverage across many simulations (the coverage loop lives in the B3 test);
  * probes for non-injected biases (formatting, authorship) report ~0 and are not flagged.

The judges are synthetic: answer strings encode a latent quality that a *fair* judge would
use, and each fake judge adds only its injected bias on top. Not a real LLM.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

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

RESULTS_DIR = Path("bench/results")

POSITION_BIAS = 0.15
VERBOSITY_BIAS = 0.08
_PAD_SIGNATURE = "To restate the question"


def _quality(answer: str) -> float:
    """Parse the latent quality a fair judge would read from an encoded answer string."""
    return float(answer.split("q=")[1].split("|")[0])


class FakePairwiseJudge:
    """Picks the first-presented answer with prob ``0.5 + content + position_bias``.

    ``content`` favors the truly-better answer; over pairs where either side is better with
    equal chance it averages out, leaving ``P(choose first) - 0.5 = position_bias``.
    """

    def __init__(
        self, *, position_bias: float, content_strength: float = 0.2, seed: int = 0
    ) -> None:
        self.position_bias = position_bias
        self.content_strength = content_strength
        self._rng = np.random.default_rng(seed)

    async def __call__(self, question: str, first: str, second: str) -> float:
        content = self.content_strength * np.sign(_quality(first) - _quality(second))
        p_first = float(np.clip(0.5 + content + self.position_bias, 0.0, 1.0))
        return 1.0 if self._rng.random() < p_first else 0.0


class FakePointwiseJudge:
    """Scores ``clip(quality + verbosity_bias*[padded] + noise)``; ignores other transforms."""

    def __init__(self, *, verbosity_bias: float, noise_sd: float = 0.05, seed: int = 0) -> None:
        self.verbosity_bias = verbosity_bias
        self.noise_sd = noise_sd
        self._rng = np.random.default_rng(seed)

    async def __call__(self, question: str, answer: str) -> float:
        quality = _quality(answer)
        pad = self.verbosity_bias if _PAD_SIGNATURE in answer else 0.0
        noise = self._rng.normal(0.0, self.noise_sd) if self.noise_sd > 0 else 0.0
        return float(np.clip(quality + pad + noise, 0.0, 1.0))


def build_pairs(n: int, *, seed: int) -> list[PairItem]:
    rng = np.random.default_rng(seed)
    pairs: list[PairItem] = []
    for i in range(n):
        qa, qb = rng.uniform(0.2, 0.8, size=2)
        pairs.append(
            PairItem(
                item_id=f"pair{i:03d}",
                question=f"Which answer is better for prompt {i}?",
                answer_a=f"q={qa:.3f}|A",
                answer_b=f"q={qb:.3f}|B",
            )
        )
    return pairs


def build_score_items(n: int, *, seed: int) -> list[ScoreItem]:
    rng = np.random.default_rng(seed)
    items: list[ScoreItem] = []
    for i in range(n):
        q = rng.uniform(0.2, 0.8)
        items.append(
            ScoreItem(item_id=f"item{i:03d}", question=f"Prompt {i}?", answer=f"q={q:.3f}|body")
        )
    return items


async def run_probes(
    *, n_pairs: int = 60, n_items: int = 60, seed: int = 0
) -> dict[str, BiasEstimate]:
    """Run all four probes once against freshly-seeded biased judges.

    The dataset, the two judges, and each probe's bootstrap draw all get *independent* seed
    streams (spawned from one root ``SeedSequence``). Reusing one seed across the
    data-generating process and the judge's noise correlates the judge's draws with the item
    qualities they are scored against and inflates the recovered bias -- see PROGRESS.md.
    """
    ds_pairs, ds_items, s_pair, s_score, s_pos, s_verb, s_fmt, s_auth = (
        int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(8)
    )
    pairs = build_pairs(n_pairs, seed=ds_pairs)
    items = build_score_items(n_items, seed=ds_items)
    pair_judge = FakePairwiseJudge(position_bias=POSITION_BIAS, seed=s_pair)
    score_judge = FakePointwiseJudge(verbosity_bias=VERBOSITY_BIAS, seed=s_score)

    position = await position_bias(pairs, pair_judge, seed=s_pos)
    verbosity = await paired_score_effect(
        items, score_judge, verbose_padding, probe="verbosity", seed=s_verb
    )
    formatting = await paired_score_effect(
        items, score_judge, markdown_formatting, probe="formatting", seed=s_fmt
    )
    authorship = await paired_score_effect(
        items, score_judge, authorship_label("gpt-4o"), probe="authorship", seed=s_auth
    )
    return {
        "position": position,
        "verbosity": verbosity,
        "formatting": formatting,
        "authorship": authorship,
    }


def run_offline(
    *, n_pairs: int = 80, n_items: int = 80, seed: int = 0
) -> tuple[dict[str, Any], Path, Path]:
    """Single-dataset B3 report: recovered vs injected bias for each probe."""
    import asyncio

    probes = asyncio.run(run_probes(n_pairs=n_pairs, n_items=n_items, seed=seed))
    truth = {
        "position": POSITION_BIAS,
        "verbosity": VERBOSITY_BIAS,
        "formatting": 0.0,
        "authorship": 0.0,
    }
    payload = {
        "meta": {
            "title": "B3 judge bias recovery (offline synthetic)",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_pairs": n_pairs,
            "n_items": n_items,
            "injected": {"position": POSITION_BIAS, "verbosity": VERBOSITY_BIAS},
            "note": "Synthetic biased judges; not a real LLM.",
        },
        "probes": {
            name: {
                "true": truth[name],
                "estimate": e.estimate,
                "ci_low": e.ci_low,
                "ci_high": e.ci_high,
                "flagged": e.flagged,
                "in_ci": e.ci_low <= truth[name] <= e.ci_high,
                "extra": e.extra,
            }
            for name, e in probes.items()
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "b3_judge_bias_offline.json"
    md_path = RESULTS_DIR / "b3_judge_bias_offline.md"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(_render_markdown(payload))
    return payload, json_path, md_path


def _render_markdown(payload: dict[str, Any]) -> str:
    m = payload["meta"]
    lines = [f"# {m['title']}", ""]
    for k, v in m.items():
        if k != "title":
            lines.append(f"- **{k}**: {v}")
    lines += [
        "",
        "| probe | injected | estimate | 95% CI | flagged | true in CI |",
        "|---|---|---|---|---|---|",
    ]
    for name, p in payload["probes"].items():
        lines.append(
            f"| {name} | {p['true']:+.3f} | {p['estimate']:+.3f} | "
            f"[{p['ci_low']:+.3f}, {p['ci_high']:+.3f}] | {p['flagged']} | {p['in_ci']} |"
        )
    lines += [
        "",
        "> B3 target: recover injected biases inside the CI (~95% coverage over sims); "
        "non-injected ~0.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="causeval.bench.judge_bias", description=__doc__)
    parser.add_argument("--n-pairs", type=int, default=80)
    parser.add_argument("--n-items", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    payload, json_path, md_path = run_offline(
        n_pairs=args.n_pairs, n_items=args.n_items, seed=args.seed
    )
    for name, p in payload["probes"].items():
        print(f"{name:12s} true={p['true']:+.3f} est={p['estimate']:+.3f} in_ci={p['in_ci']}")
    print(f"wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
