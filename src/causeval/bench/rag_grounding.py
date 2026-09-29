"""B2 -- RAG grounding benchmark (SPEC 6).

Claim: Counterfactual Adherence (CA) separates grounded items from parametric items, where
DeepEval Faithfulness cannot. The offline version uses a behavior-driven fake app with a
known per-item ground-truth label and checks that CA separates the two with AUROC >= 0.9.

The live version (real judge, run manually) additionally reports DeepEval Faithfulness AUROC
on the same split; the expected -- to be verified, not assumed -- result is that Faithfulness
cannot separate the world-knowledge set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from causeval.interventions.cf_gen import Counterfactual
from causeval.interventions.rag import GroundingItem, GroundingResult, ground

RESULTS_DIR = Path("bench/results")


def roc_auc(scores: list[float], labels: list[int]) -> float:
    """AUROC via the Mann-Whitney U statistic (ties handled by average ranks)."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)
    sum_pos = float(ranks[y == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ---- fictional knowledge base -----------------------------------------------------

_DEPARTMENTS = ["Autonomy", "Perception", "Fleet Ops", "Safety", "Hardware", "Simulation"]


@dataclass(frozen=True)
class LabeledItem:
    item: GroundingItem
    label: int  # 1 = grounded behaviour, 0 = parametric behaviour


def build_labeled_dataset(n: int = 40, *, seed: int = 0) -> list[LabeledItem]:
    """Fictional-KB QA items; half assigned grounded behaviour, half parametric."""
    out: list[LabeledItem] = []
    for i in range(n):
        dept = _DEPARTMENTS[i % len(_DEPARTMENTS)]
        # A distinct team per item so questions are unique (real questions are).
        team = f"{dept} Group {i:02d}"
        true_year = str(2015 + (i % 8))
        cf_year = str(1990 + (i % 8))  # plausible but false, same type
        chunk = f"Northwind Robotics' {team} was founded in {true_year}."
        edited = chunk.replace(true_year, cf_year, 1)
        item = GroundingItem(
            item_id=f"item{i:02d}",
            question=f"In what year was Northwind Robotics' {team} founded?",
            contexts=[chunk],
            expected_output=true_year,
            cf=Counterfactual(
                chunk_index=0, original_value=true_year, cf_value=cf_year, edited_chunk=edited
            ),
        )
        label = int(i % 2 == 0)  # even items grounded, odd items parametric
        out.append(LabeledItem(item=item, label=label))
    return out


class BehaviorApp:
    """Fake RAG app whose per-item behaviour is grounded or parametric, with slip noise.

    * grounded item  (follow_prob high, no memory): usually answers from context, so under
      the counterfactual it states the false value.
    * parametric item (follow_prob low, knows the true value): usually answers from memory,
      so it ignores the counterfactual and keeps the true value.
    """

    def __init__(
        self,
        labeled: list[LabeledItem],
        *,
        follow_prob_grounded: float = 0.9,
        follow_prob_parametric: float = 0.1,
        seed: int = 0,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._follow: dict[str, float] = {}
        self._memory: dict[str, str | None] = {}
        self._question_to_id: dict[str, str] = {}
        for li in labeled:
            gid = li.item.item_id
            self._follow[gid] = follow_prob_grounded if li.label == 1 else follow_prob_parametric
            self._memory[gid] = None if li.label == 1 else li.item.expected_output
            self._question_to_id[li.item.question] = gid

    async def answer(self, question: str, contexts: list[str]) -> str:
        gid = self._question_to_id[question]
        use_context = bool(contexts) and (self._rng.random() < self._follow[gid])
        if use_context:
            return " ".join(contexts)
        memory = self._memory[gid]
        return memory if memory is not None else "I don't know."


# ---- runners ----------------------------------------------------------------------


def _ca_auroc(result: GroundingResult, labels: dict[str, int]) -> float:
    scores: list[float] = []
    ys: list[int] = []
    for it in result.items:
        if it.counterfactual_adherence is not None:
            scores.append(it.counterfactual_adherence)
            ys.append(labels[it.item_id])
    return roc_auc(scores, ys)


def run_offline(
    *, n: int = 40, repeats: int = 8, seed: int = 0
) -> tuple[dict[str, Any], Path, Path]:
    """Offline B2: CA AUROC separating grounded vs parametric behaviour."""
    labeled = build_labeled_dataset(n, seed=seed)
    labels = {li.item.item_id: li.label for li in labeled}
    app = BehaviorApp(labeled, seed=seed)
    result = ground(app, [li.item for li in labeled], repeats=repeats, seed=seed)

    ca_auroc = _ca_auroc(result, labels)
    payload = {
        "meta": {
            "title": "B2 RAG grounding (offline synthetic)",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_items": n,
            "repeats": repeats,
            "note": "Synthetic behaviour-driven app; not a real judge.",
        },
        "ca_auroc": ca_auroc,
        "class_counts": result.class_counts,
        "estimates": [e.model_dump(mode="json") for e in result.estimates],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "b2_rag_grounding_offline.json"
    md_path = RESULTS_DIR / "b2_rag_grounding_offline.md"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(_render_markdown(payload, result))
    return payload, json_path, md_path


def _render_markdown(payload: dict[str, Any], result: GroundingResult) -> str:
    m = payload["meta"]
    lines = [f"# {m['title']}", ""]
    for k, v in m.items():
        if k != "title":
            lines.append(f"- **{k}**: {v}")
    lines.append("")
    auroc = payload["ca_auroc"]
    lines.append(f"**Counterfactual Adherence AUROC (grounded vs parametric): {auroc:.3f}**")
    lines.append("")
    lines.append(result.summary())
    lines.append("")
    lines.append(
        "> CA separates items that causally use the context from items answering "
        "parametrically. B2 target: AUROC >= 0.9."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="causeval.bench.rag_grounding", description=__doc__)
    parser.add_argument("--n-items", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    payload, json_path, md_path = run_offline(n=args.n_items, repeats=args.repeats, seed=args.seed)
    print(f"CA AUROC = {payload['ca_auroc']:.3f}  (target >= 0.9)")
    print(f"wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
