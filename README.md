<div align="center">

# causeval

**Turn LLM evaluation scores into defensible evidence.**

A statistically rigorous, *causal* evaluation layer for LLM apps, built on top of
[DeepEval](https://github.com/confident-ai/deepeval).

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](./LICENSE)
[![Built on DeepEval](https://img.shields.io/badge/built%20on-DeepEval%20%E2%89%A54.2-orange)](https://github.com/confident-ai/deepeval)
[![Status](https://img.shields.io/badge/status-alpha-yellow)](./PROGRESS.md)

[![GitHub followers](https://img.shields.io/github/followers/routsom?label=Follow%20%40routsom&style=social)](https://github.com/routsom)
[![GitHub stars](https://img.shields.io/github/stars/routsom/causeval?style=social)](https://github.com/routsom/causeval)

**Created and maintained by [@routsom](https://github.com/routsom).**

</div>

---

DeepEval **measures**. It gives you a score. But a single score can't tell you whether it's
*real* (or just noise), *why* your app produced an output, or whether the LLM judge that
produced the score can be *trusted*.

causeval wraps DeepEval's metrics - they stay the measurement instrument - and adds the three
things a score alone can't give you:

- 📊 **Uncertainty** - *Is the score real?* Repeated sampling, variance decomposition,
  clustered-bootstrap confidence intervals, paired comparisons, and statistically valid gates.
- 🔬 **Causality** - *What caused this output or failure?* RAG context ablation and
  counterfactual context, input perturbations with metamorphic relations, and agent
  step-level counterfactual replay.
- ⚖️ **Judge validity** - *Can we trust the judge?* Bias audits, calibration against human
  labels, prediction-powered inference (PPI), and conformal abstention.

> **The one rule that drives everything:** causeval *never reports a bare score*. Every public
> result carries an item count, a repeat count, a confidence interval, the method used, and
> full provenance (versions, seed, dataset hash, git SHA).

## Table of contents

- [Why causeval?](#why-causeval)
- [Install](#install)
- [Quickstart](#quickstart)
- [What it adds, feature by feature](#what-it-adds-feature-by-feature)
- [Does it actually work? (validation benchmarks)](#does-it-actually-work-validation-benchmarks)
- [CLI](#cli)
- [Design principles](#design-principles)
- [Project status & roadmap](#project-status--roadmap)
- [Author](#author)
- [License](#license)

## Why causeval?

DeepEval is an excellent measurement library. causeval is not a competitor - it is the layer
that turns DeepEval's measurements into decisions you can defend in a code review, a launch
meeting, or a paper. Here is precisely what it adds:

| Gap when you use metrics alone | causeval's answer |
|---|---|
| Single-sample scores, threshold pass/fail, **flaky results run to run** | Repeated sampling, clustered-bootstrap CIs, three-valued gates (pass / regression / inconclusive) |
| No paired tests, **no multiple-comparison control** | Paired bootstrap of Δ, Wilcoxon, Holm-adjusted CIs across metrics |
| Faithfulness checks *entailment*, not **dependence on the context** | Context Reliance + Counterfactual Adherence: does the answer actually *change* when you change the evidence? |
| The **judge is never itself evaluated** | Position/verbosity/formatting bias probes, human calibration, PPI, conformal abstention |
| No robustness or **fairness causal tests** | Perturbation engine with metamorphic relations + counterfactual fairness gaps |
| Agent failures give you **no step-level blame** | Counterfactual replay: which step, if fixed, would have saved the run? |
| **Wasteful test sets** - every item costs the same | 2PL IRT difficulty/discrimination + information-based pruning |
| Shared metric objects corrupt scores under concurrency ([issue #3356](https://github.com/confident-ai/deepeval/issues/3356)) | A fresh metric instance per measurement, enforced by a factory + a test |
| Telemetry on by default; dotenv loaded at import | An import guard sets the opt-outs *before* the first `import deepeval` |

## Install

causeval targets Python 3.10-3.12.

```bash
# from source (recommended while in alpha)
git clone https://github.com/routsom/causeval.git
cd causeval
uv sync                      # core install
uv sync --all-extras         # + optional backends (see below)
```

Optional extras, each pulling in a heavier dependency only where you need it:

| Extra | Enables | Pulls in |
|---|---|---|
| `causeval[nli]` | NLI-based equivalence checks for perturbation invariance | `transformers`, `torch` |
| `causeval[ppi]` | numerical cross-check of the PPI estimator | `ppi-python` |
| `causeval[irt]` | cross-check of the 2PL IRT fit | `girth` |
| `causeval[observational]` | reference for observational causal estimation | `dowhy`, `econml` |

> The core estimators (bootstrap CIs, PPI, IRT, AIPW) are implemented natively on
> numpy/scipy - the extras are for optional cross-checks and NLI, not for the math.

## Quickstart

### 1. Get a score you can trust (uncertainty)

The core object is `Experiment`: run each metric over each item `R` times and get back a
`RunResult` whose estimates carry CIs and variance components.

```python
from causeval import Experiment, MetricSpec

exp = Experiment(
    dataset=goldens,                                   # DeepEval Goldens or plain dicts
    metrics=[MetricSpec("FaithfulnessMetric", {"threshold": 0.7})],
    repeats=5,                                          # resample the judge 5x per item
    seed=0,
)
run = exp.run()          # or: await exp.a_run()
print(run.summary())
```

```text
Run 'run'  (seed=0)
judge=gpt-4o-mini  causeval=0.0.1
  Faithfulness [base]: 0.812 [95% CI 0.771, 0.849] (n=50, R=5, icc=0.34, flaky=6)
      method=cluster_bootstrap_studentized_B=2000
```

You immediately learn what a bare score hides: the CI, that ~34% of the variance is between
items (the rest is judge noise), and that **6 items are flaky** (pass rate between 0.2 and 0.8).

### 2. Decide if B is really better than A (paired gate)

```python
from causeval.stats import compare, gate

comparisons = compare(                              # Holm-adjusted across metrics
    baseline_run.measurements,
    candidate_run.measurements,
    margin=0.02,
)
report = gate(comparisons)
print(report.overall)     # "pass" | "regression" | "inconclusive"
raise SystemExit(report.exit_code)                  # 0 / 1 / 2 for CI
```

The gate is **three-valued on purpose**: "inconclusive" (the CI straddles the margin) is a
first-class outcome, so you never ship a regression that hid behind noise, and never block a
release on a difference you didn't have the power to detect. Plan that power up front with
`causeval.stats.plan_power`.

### 3. Ask whether the answer actually used the retrieved context (causality)

DeepEval's Faithfulness tells you the answer is *entailed* by the context. It cannot tell you
the model would have said the same thing *without* it. causeval intervenes on the context and
watches the answer move:

```python
from causeval.interventions import ground, GroundingItem

result = ground(my_rag_app, dataset, repeats=8)
print(result.summary())
```

```text
RAG grounding (tau=0.5, policy=follow_context)
  context_reliance:          +0.463 [95% CI +0.311, +0.621] (n=40, R=8)
  counterfactual_adherence:  +0.506 [95% CI +0.372, +0.643] (n=40, R=8)
  classes: grounded=20, parametric=20
```

**Counterfactual Adherence** edits one supporting fact to a plausible-but-false value and
checks whether the answer follows the edit. Items are labelled *grounded* / *parametric* /
*confabulating* / *mixed*, and any answer that passes Faithfulness while being parametric is
flagged **false-faithful**.

### 4. Audit the judge before you believe it (judge validity)

```python
from causeval.judge_audit import calibrate, ppi_mean_ci

report = calibrate(judge_scores, human_labels)   # Spearman, kappa, isotonic map, ECE + CI
effect = ppi_mean_ci(human_labels, judge_on_labeled, judge_on_unlabeled)  # human mean, debiased
```

PPI gives you the **human-quality mean** with a CI that is unbiased (unlike averaging the
judge) yet far tighter than using your few human labels alone.

## What it adds, feature by feature

| Module | What you get | Key estimand |
|---|---|---|
| `stats/` | Repeated-sampling estimates, cluster-bootstrap CIs (studentized / percentile / BCa), variance decomposition (ICC), paired Δ + Wilcoxon + McNemar, Holm, three-valued gates, power planning | `μ = E_item[E_repeat[y]]` with valid uncertainty |
| `interventions/rag.py` | Context Reliance, Counterfactual Adherence, per-chunk leave-one-out effects, grounded/parametric classification, false-faithful flag | `P(answer follows a counterfactual edit)` |
| `interventions/perturb.py` | Metamorphic perturbations (paraphrase, reorder, distractor, typo, attribute swap) with invariance rate, paired metric effect, Holm-adjusted fairness gaps | `Δ metric under a meaning-preserving change` |
| `interventions/cot.py` | Chain-of-thought faithfulness: early-answering curve + area-over-curve, mistake-insertion sensitivity | `does the stated reasoning drive the answer?` |
| `judge_audit/` | Bias probes (position/verbosity/formatting/authorship), calibration (isotonic + ECE), sampling plans, PPI/PPI++, conformal abstention (Learn-then-Test), jury + Krippendorff's α | `bias effect`, `E[human score]`, `error rate ≤ α w.p. ≥ 1-δ` |
| `attribution/` | Agent trace schema, record/replay cassettes, step-level counterfactual attribution, decisive-step detection | `P(success \| do(step_k = oracle)) - P(success \| natural)` |
| `stats/irt.py` | 2PL item-response theory, Fisher-information pruning, system ranking | item difficulty / discrimination, ability |
| `stats/observational.py` | Doubly-robust AIPW effect from production logs, with DoWhy-style refuters | ATE of a version on a metric |
| `checks/` | Deterministic pre-judge checks (JSON schema, regex, verifier callables, NLI) + the metamorphic-relation registry | fail fast before spending judge tokens |
| `core/` | Pydantic result schemas, content-hashed LLM cache (SQLite), cost planner, provenance | reproducibility & cost control |

## Does it actually work? (validation benchmarks)

Every statistical claim has a benchmark with a *known ground truth*, run in CI on synthetic
data (fakes, no network), with a live variant for real models. These are the committed offline
numbers ([`bench/results/`](./bench/results)):

| Benchmark | Claim | Target | Result |
|---|---|---|---|
| **B2** RAG grounding | Counterfactual Adherence separates grounded from parametric answers | AUROC ≥ 0.9 | **1.000** |
| **B3** Judge bias recovery | Probes recover an injected position (0.15) + verbosity (0.08) bias inside their CIs; non-injected ≈ 0 | ~95% coverage | **recovered 0.156 / 0.070; formatting & authorship not flagged** |
| **B4** PPI | PPI covers the true human mean; the naive judge-mean does not | ~95% coverage, narrower than human-only | **PPI 0.99 @ width 0.061; naive 0.00; human-only 0.98 @ width 0.112** |
| **B5** Agent attribution | The decisive step equals the injected fault step | ≥ 90% of tasks | **100%** |
| **B6** IRT | 2PL parameter recovery; pruning to 50% preserves system ranking | corr ≥ 0.9; Kendall τ ≥ 0.9 | **b=0.99, a=0.95, θ=0.98; τ=1.00** |

*(B1 - CI coverage, false-regression rate, and power - is verified by simulation tests under
`tests/`.)*

### Live-model benchmarks

The numbers above are the **offline** benchmarks: synthetic fakes with a known ground truth,
run in CI on every commit so the *statistics* are provably correct without spending a token.
Each benchmark also has a **live** variant that swaps the fake for a real judge/model, so you
can publish the same claims against, say, `gpt-4o-mini` or `claude-haiku`.

> ⏳ **Status: not yet published.** Live tables need provider API keys and a chosen default
> judge model (tracked as an open question in [`PROGRESS.md`](./PROGRESS.md)). The table below
> is the template these results will fill; run it yourself with the commands underneath.

| Benchmark | Live claim to verify | Judge model | Result |
|---|---|---|---|
| **Baseline variance** | DeepEval single-sample scores are flaky; repeats + CIs quantify it | _tbd_ | _pending_ |
| **B2** RAG grounding | Counterfactual Adherence separates grounded vs parametric answers (AUROC ≥ 0.9); Faithfulness cannot | _tbd_ | _pending_ |
| **B3** Judge bias | Real judges show measurable position/verbosity bias with CIs | _tbd_ | _pending_ |
| **B4** PPI | PPI covers the human mean and beats the naive judge-mean at equal labels | _tbd_ | _pending_ |
| **B5** Agent attribution | Decisive step = injected fault step in ≥ 90% of tasks with a real model | _tbd_ | _pending_ |
| **B6** IRT | On ≥ 5 real systems, pruning 30-50% of items keeps ranking Kendall τ ≥ 0.9 | _tbd_ | _pending_ |

Reproduce a live run (needs an API key for the provider you name; nothing is hardcoded):

```bash
export OPENAI_API_KEY=...                # or your provider's key
uv sync

# baseline flakiness with a real judge
uv run python -m causeval.bench.baseline_variance --model gpt-4o-mini

# the full live test suite (opt-in; excluded from the default offline run)
uv run pytest -m live
```

Live results are written under [`bench/results/`](./bench/results) alongside the offline ones;
open a PR with your table and we'll add it here.

## CLI

Every command writes a JSON result and prints a human-readable summary; none of them will ever
print a bare number.

```bash
causeval run          --config eval.yaml   --out runs/   # repeated-sampling run
causeval compare      --baseline a.json --candidate b.json --margin 0.02
causeval gate         --baseline a.json --candidate b.json --margin 0.02  # exit 0/1/2
causeval plan         --pilot-baseline a.json --pilot-candidate b.json --metric Faithfulness --detect 0.03
causeval ground       --config rag.yaml    --out runs/   # RAG causal grounding
causeval audit-judge  --config judge.yaml  --out runs/   # calibration + PPI
causeval attribute    --config agent.yaml  --out runs/   # agent step-level blame
```

`causeval gate` sets the process exit code (`0` pass, `1` regression, `2` inconclusive), so it
drops straight into CI as a release gate.

## Design principles

These are enforced by tests, not just documented (see [`CLAUDE.md`](./CLAUDE.md)):

1. **Wrap DeepEval, never fork it.** Depend on `deepeval>=4.2,<5`; never patch its internals.
2. **All DeepEval imports go through one guarded module** that sets telemetry/dotenv opt-outs
   before the first import. A test enforces that no other module imports it directly.
3. **A fresh metric object per measurement** - never share an instance across concurrent calls
   (DeepEval issue #3356).
4. **Never report a bare score.** Every result carries `n_items`, `n_repeats`, a CI, the
   method, and provenance - including CLI output.
5. **No telemetry, no network, no side effects at import time.**
6. **No hardcoded model prices or names in logic** - they come from user config.
7. **Offline tests never call real LLMs** - they use configurable fakes; live tests are opt-in.
8. **Every statistical method has a simulation test** proving its coverage or error rate on
   data with a known answer.

## Project status & roadmap

**Alpha.** The full estimator suite (Phases 0-6) and the causal extensions of Phase 7 are
implemented, with 180+ offline tests and strict typing. `causeval` is a working name.

- ✅ Phase 0-1: scaffold, core schemas, adapters, statistics (CIs, gates, planning)
- ✅ Phase 2: RAG causal grounding
- ✅ Phase 3: judge audit (bias, calibration, PPI, conformal, jury)
- ✅ Phase 4: perturbations + deterministic checks
- ✅ Phase 5: agent step-level attribution
- ✅ Phase 6: IRT pruning + adaptive sampling
- ✅ Phase 7 (partial): CoT faithfulness, observational AIPW, OpenTelemetry trace import
- ⏳ Planned: framework harness adapters (LangGraph / OpenAI Agents / Pydantic AI), an HTML
  report, a pytest plugin, and published live-model benchmark tables

See [`SPEC.md`](./SPEC.md) for the full design and [`PROGRESS.md`](./PROGRESS.md) for the
decision log.

### Development

```bash
uv sync --all-extras
uv run pytest -m "not live"                       # fast offline suite (must always pass)
uv run pytest -m live                             # real LLM calls; needs API keys
uv run ruff check . && uv run ruff format --check .
uv run mypy src/causeval                          # strict
uv run python -m causeval.bench.rag_grounding     # regenerate a benchmark report
```

## Author

**causeval is created, designed, and maintained by [@routsom](https://github.com/routsom).**

If this project is useful to you, please ⭐ [star the repo](https://github.com/routsom/causeval)
and [follow @routsom](https://github.com/routsom) for more work on rigorous LLM evaluation.
Issues, ideas, and pull requests are welcome.

## License

[Apache 2.0](./LICENSE), matching DeepEval. Portions of DeepEval, where copied, retain their
Apache 2.0 headers and are recorded in [`NOTICE`](./NOTICE).

causeval is an independent project and is not affiliated with or endorsed by Confident AI, the
maintainers of DeepEval.
