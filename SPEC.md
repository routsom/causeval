# causeval — Design Specification

Status: draft v0.1 (2026-09-28). Owner: project author. Implementer: Claude Code.

## 0. Purpose and scope

causeval turns LLM evaluation scores into defensible evidence. It wraps DeepEval metrics
(which remain the measurement instrument) and adds:

1. **Uncertainty**: repeated sampling, variance decomposition, confidence intervals,
   paired comparisons, statistically valid CI gates, sample-size planning.
2. **Causality**: interventions that test *why* an output happened. RAG context ablation
   and counterfactual context, input perturbations, agent step-level counterfactual replay.
3. **Judge validity**: bias audits, calibration against human labels, prediction-powered
   inference (PPI), conformal abstention.

### Non-goals

- Replacing DeepEval's metrics, tracing, synthesizer, red-teaming or benchmarks.
- A hosted platform or UI (JSON + Markdown reports only until Phase 7).
- White-box mechanistic interpretability (activation patching etc.). Out of scope.
- Training or fine-tuning anything.

## 1. DeepEval gaps this addresses

| Gap in DeepEval (v4.2.x) | causeval answer | Phase |
|---|---|---|
| Single-sample scores, threshold pass/fail, flaky CI | Repeats, clustered bootstrap CIs, three-valued gates | 1 |
| No paired tests or multiple-comparison control | Paired bootstrap, Wilcoxon, Holm correction | 1 |
| Faithfulness checks entailment, not dependence on context | Context reliance + counterfactual adherence | 2 |
| Judge never evaluated | Bias probes, human calibration, PPI, conformal abstention | 3 |
| No robustness/fairness causal tests | Perturbation engine with metamorphic relations | 4 |
| Agent failures lack step-level blame | Counterfactual replay attribution | 5 |
| Wasteful test sets | IRT difficulty/discrimination, pruning | 6 |
| Shared metric objects under concurrency (issue #3356) | Fresh metric per measurement via factory | 0 |
| Telemetry on by default, dotenv loaded at import | Import guard sets opt-outs before first import | 0 |
| G-Eval regenerates steps when given only `criteria` | Adapter warns and records; recommends locked `evaluation_steps` | 0 |

## 2. Core data model (`core/schemas.py`)

All public results are pydantic v2 models, JSON-serializable, with `schema_version`.

```python
class Provenance(BaseModel):
    causeval_version: str
    deepeval_version: str
    judge_model: str | None
    judge_params: dict  # temperature, max_tokens, etc.
    prompt_hashes: dict[str, str]  # metric name -> hash of its prompt templates, where available
    dataset_hash: str
    git_sha: str | None
    seed: int
    created_at: datetime


class Measurement(BaseModel):  # one metric on one item on one repeat
    item_id: str
    metric: str
    condition: str = "base"  # e.g. "base", "ctx_none", "loo_2", "cf", "perturb:paraphrase"
    repeat_index: int
    score: float | None
    passed: bool | None
    reason: str | None
    error: str | None  # set when the judge call failed; score is then None
    cost_usd: float | None
    latency_s: float | None


class ScoreEstimate(BaseModel):
    metric: str
    condition: str
    estimate: float  # mean over items of per-item mean score
    ci_low: float
    ci_high: float
    ci_level: float
    method: str  # e.g. "cluster_bootstrap_percentile_B=2000"
    n_items: int
    n_repeats: int
    n_failed: int  # measurements excluded because of errors
    var_between: float  # between-item variance of item means
    var_within: float  # mean within-item (judge/app noise) variance
    icc: float  # var_between / (var_between + var_within)
    flaky_items: list[str]  # 0.2 < pass rate < 0.8 across repeats


class Comparison(BaseModel):
    metric: str
    delta: float  # candidate - baseline
    ci_low: float
    ci_high: float
    p_value: float | None
    p_adjusted: float | None  # Holm across metrics in the same comparison
    method: str
    verdict: Literal["pass", "regression", "inconclusive"]
    margin: float


class RunResult(BaseModel):
    provenance: Provenance
    measurements: list[Measurement]
    estimates: list[ScoreEstimate]
```

Store runs as `runs/<timestamp>_<name>.json`. JSON is the canonical artifact; Markdown
summaries are rendered from it.

## 3. Module specifications

### 3.1 `core/`

- `llm_cache.py`: SQLite content-addressed cache. Key = sha256 of (provider, model, sorted
  params, messages, seed, repeat_index, purpose). Stores response text, token counts, cost,
  timestamp. `--no-cache` and `--cache-readonly` modes. Thread/async safe.
- `concurrency.py`: global `asyncio.Semaphore`, per-provider rate limiting, tenacity retries
  with jitter. Failed calls after retries produce `Measurement.error`, never a default score.
- `cost.py`: planner. `plan(n_items, n_metrics, n_repeats, n_conditions, calls_per_metric,
  tokens_per_call, price_table) -> CostPlan`. Price table is user config (YAML), never
  hardcoded. CLI prints the plan and asks for confirmation above a configured budget.
- `provenance.py`: builds `Provenance`; dataset hash is sha256 of canonical JSON of goldens.
- `errors.py`: `CausevalError`, `JudgeCallError`, `InterventionInvalidError`,
  `ReplayDivergenceError`, `InsufficientDataError`.

### 3.2 `adapters/`

- `deepeval_import.py`: the only place that imports deepeval. Before import, set
  `os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")` and
  `os.environ.setdefault("DEEPEVAL_DISABLE_DOTENV", "1")`. Re-export the needed symbols.
  Test: grep `src/` for `import deepeval` / `from deepeval` outside this module -> fail.
- `metric_factory.py`: `MetricSpec(name: str, kwargs: dict)` or a user zero-arg callable.
  `build(spec) -> BaseMetric` returns a **new instance every call**. `a_measure_once(spec,
  test_case) -> Measurement` builds, measures with `a_measure`, extracts score/reason/success,
  catches exceptions into `error`.
  - If the spec is GEval with `criteria` but no `evaluation_steps`, emit a warning and record
    `nondeterministic_steps=True` in the measurement metadata.
- `judge_model.py`: adapter so any causeval judge client can be passed to DeepEval metrics as
  a `DeepEvalBaseLLM` subclass (implement `load_model`, `generate`, `a_generate`,
  `get_model_name`; check `generate_with_schema` support in the installed version). All calls
  route through `core.llm_cache`.
- Concurrency regression test: reproduce the #3356 pattern (many concurrent measurements of
  different test cases with an intentionally slow fake judge) and assert every measurement's
  score matches its own test case.

### 3.3 `stats/`

Notation: items `i = 1..n`, repeats `r = 1..R`, score `y_ir`. Item mean `ȳ_i`.

**Estimand (default):** `μ = E_item[ E_repeat[ y ] ]`, the expected score of a randomly drawn
item under a random judge/app sample. Estimator: `μ̂ = mean_i(ȳ_i)`.

- `estimate.py`
  - `estimate(measurements, level=0.95, B=2000, rng) -> ScoreEstimate`.
  - CI: cluster bootstrap. Resample items with replacement, keep each item's repeats together.
    Percentile interval by default; BCa optional. If `n < 20` also report a t-interval and
    flag `small_sample=True`.
  - Variance decomposition by one-way random-effects ANOVA (method of moments, truncate
    negative components at 0). Report `var_between`, `var_within`, `icc`.
  - Per-item pass rate `p_i`; flaky if `0.2 < p_i < 0.8` (configurable).
- `compare.py`
  - `compare(baseline, candidate, metrics, margin, level, rng) -> list[Comparison]`.
  - Requires matched item ids; error on mismatch unless `allow_partial=True` (then intersect
    and report dropped count).
  - Primary: paired cluster bootstrap of `Δ = mean_i(ȳ_i^cand − ȳ_i^base)`.
    Secondary p-value: Wilcoxon signed-rank on item-level differences.
    Binary pass/fail: paired bootstrap of per-item pass rates; McNemar on majority-vote pass.
  - Holm-Bonferroni across the metrics in one comparison call.
- `gate.py` — three-valued, non-inferiority style:
  - `regression` if `ci_high < -margin` (confidently worse than tolerance)
  - `pass` if `ci_low > -margin` (confidently not worse than tolerance)
  - otherwise `inconclusive`. CLI exit codes: 0 pass, 1 regression, 2 inconclusive.
  - Gate uses Holm-adjusted confidence levels when several metrics gate together.
- `planner.py`
  - From a pilot run, estimate variance components and compute the `(n, R)` needed to detect
    a delta `δ` with power `1−β`:
    `Var(Δ̂) ≈ (σ²_diff_between + σ²_within_A/R + σ²_within_B/R) / n`.
  - Output a small table of `(n, R, expected CI half-width, estimated cost)` options.
- `irt.py` (Phase 6): 2PL IRT on the binary item × system matrix (needs ≥ 5 systems).
  Use `girth` or `py-irt` behind an interface; cross-check on simulated data. Outputs item
  difficulty and discrimination; `prune(target_size)` selects items maximizing Fisher
  information over the ability range of the systems of interest.
- `adaptive.py` (Phase 6): allocate extra repeats to flaky / high-variance items. Document
  that data-dependent allocation needs the simulation test to confirm coverage still holds.

### 3.4 `interventions/`

#### 3.4.1 RAG grounding (`rag.py`) — flagship causal metric

App protocol (user implements; bypasses the retriever so contexts can be injected):

```python
class RAGApp(Protocol):
    async def answer(self, question: str, contexts: list[str]) -> str: ...
```

Conditions per item (each run with `R` repeats):

| Condition | Contexts given | Purpose |
|---|---|---|
| `full` | all retrieved chunks | baseline |
| `none` | `[]` | what the model knows without context |
| `loo_k` | all except chunk k | per-chunk attribution |
| `cf` | supporting chunk with one key fact edited to a plausible false value | does the answer follow the context? |

Outcome functions (pluggable; prefer deterministic when possible):
- `correct(answer, expected_output) -> float in [0,1]`: exact/regex, NLI, or calibrated judge.
- `follows_cf(answer, cf_edit) -> float in [0,1]`: answer states the counterfactual value.

Metrics (all reported as `ScoreEstimate`-style with CIs):
- **Context Reliance** `CR = P(correct | full) − P(correct | none)`.
- **Counterfactual Adherence** `CA = P(follows_cf | cf)`. Primary grounding signal.
- **Chunk effect** `drop_k = P(correct | full) − P(correct | loo_k)`. Compare with DeepEval
  Contextual Precision's ranking to find chunks that are ranked high but causally unused.
- **Item classification**: `grounded` (CA ≥ τ), `parametric` (correct under `none` and
  CA < τ), `confabulating` (wrong under `full`, CA < τ), `mixed`.
- **False-faithful flag**: DeepEval Faithfulness ≥ its threshold AND item is `parametric`.
  This is the headline finding the benchmark must demonstrate.

Expected behaviour under a knowledge conflict is product-dependent. Config
`conflict_policy: follow_context | flag_conflict`. With `flag_conflict`, "follows" means the
answer surfaces the conflict rather than silently choosing either value.

Counterfactual generation (`cf_gen.py`):
1. Identify the supporting chunk (highest `drop_k`, or user-specified).
2. Identify the key fact span answering the question (LLM extraction, returns span offsets).
3. Generate a plausible replacement of the same type (number→number, date→date, name→name).
4. Validity checks, all must pass or the item is skipped with a recorded reason:
   - edit is local: character-level diff touches only the target span (± punctuation);
   - contradiction: NLI or judge confirms new chunk contradicts the original on that fact;
   - naturalness: the edited chunk stays fluent (judge yes/no or LM perplexity ratio bound).
5. Users may supply hand-written counterfactuals (`cf_overrides.jsonl`); preferred in
   high-stakes domains.

#### 3.4.2 Perturbation engine (`perturb.py`, Phase 4)

Each perturbation declares a **metamorphic relation**: `invariant` (output meaning should
not change) or `directional` (output should change in a stated way).

Built-ins: paraphrase question, reorder multiple-choice options, insert irrelevant distractor
sentence into context, formatting change, typo noise, demographic/attribute swap from a
user-configured list of pairs.

Measurements:
- Invariance rate: fraction of items where output meaning is unchanged (bidirectional NLI or
  equivalence judge), with CI.
- Metric effect: paired `Δ` on any metric between original and perturbed, with CI.
- Counterfactual fairness gap for attribute swaps: paired `Δ` per attribute pair, Holm-adjusted.

Every generated perturbation passes a validity check (meaning-preserving for `invariant`).
Invalid perturbations are dropped and counted, never silently kept.

#### 3.4.3 Chain-of-thought faithfulness (`cot.py`, Phase 7, optional)

Only for models/APIs that allow prefilling the assistant turn and expose the reasoning text.
- Early answering: truncate reasoning at fractions {0, .25, .5, .75, 1}, force a final answer,
  measure agreement with the full-reasoning answer; report area over the curve.
- Mistake insertion: corrupt one reasoning step; measure the rate at which the answer changes.
Low sensitivity means the stated reasoning is not what drives the answer. Document clearly
that hidden-reasoning models cannot be tested this way.

### 3.5 `judge_audit/` (Phase 3)

- `bias_probes.py` — randomized interventions on the judge, each with paired CIs:
  - **Position** (pairwise judges): judge every pair in both orders. Report
    `P(choose first) − 0.5` and order-consistency rate.
  - **Verbosity**: pad the answer with semantically null text (restating the question, polite
    filler) verified null by NLI. Effect = `score(padded) − score(original)`.
  - **Formatting**: same content as Markdown vs plain prose.
  - **Authorship label**: add a "response written by <model>" line; effect of the label.
  - **Self-preference (matched)**: requires human labels. Among items humans rate equal
    quality, compare judge scores for outputs from the judge's own family vs others.
  - Flag any effect whose CI excludes 0 and whose magnitude exceeds a configured tolerance.
- `calibration.py` — against a human-labeled set:
  - Spearman correlation, weighted Cohen's kappa (binarized at threshold), confusion matrix.
  - Isotonic regression mapping judge score → P(human pass). Expected calibration error with
    bootstrap CI. Save the fitted map for use by `ppi.py` and gates.
- `sampling.py` — chooses which items to send for human labeling: simple random sample with
  recorded seed (required for PPI validity), optional stratification by judge score.
  Exports a CSV/JSONL labeling sheet and imports completed labels.
- `ppi.py` — prediction-powered inference for the human-quality mean.
  - Estimand: mean human score over the full dataset.
  - Estimator (PPI for the mean, with PPI++ power tuning `λ`):
    `θ̂ = λ·mean(f(X_unlabeled)) + mean(Y_labeled − λ·f(X_labeled))`, CI from the two
    variance terms. Implement directly (it is short) and cross-check numerically against
    `ppi_py` (PyPI `ppi-python`) in tests; confirm its function names from its source.
  - Report the effective sample size gain versus human-only estimation.
  - Refuse to run if labeled items were not produced by `sampling.py` (random-sample check),
    unless `--i-know-the-labels-are-random`.
- `conformal.py` — selective judging with a guaranteed error rate:
  - Judge confidence from logprobs when the provider returns them, else agreement across k
    repeated judge samples.
  - Choose the confidence threshold by Learn-then-Test with a binomial (Clopper-Pearson) tail
    bound so that, with probability ≥ 1−δ, the error rate among accepted items ≤ α.
  - Output per item: `accept` with judge verdict or `abstain` (route to human); report coverage.
- `jury.py` — k judges from different model families; mean or majority aggregation;
  Krippendorff's alpha across judges; disagreement surfaced per item.

### 3.6 `attribution/` (Phase 5) — agent step-level blame

- `trace.py`: internal `Trace` = ordered `Step`s (`llm_call`, `tool_call`, `retrieval`,
  `handoff`), each with inputs, outputs, timing. Importers: DeepEval traces (via the adapter)
  and OpenTelemetry GenAI spans exported as JSON (Phase 7).
- `cassette.py`: record mode wraps tools and stores `(tool, canonical_args) -> output`.
  Replay mode serves recorded outputs on exact match. On novel inputs after an intervention:
  - read-only tools marked `safe_live=True` may be called live;
  - otherwise use a simulated tool (LLM given the tool's schema and recorded examples),
    and mark the rollout `fidelity="simulated"`;
  - side-effecting tools are never called live unless `allow_live_side_effects=True`.
- Harness protocol (user implements; a reference implementation lives in `bench/`):

```python
class AgentHarness(Protocol):
    async def run(self, task: str, *, forced_prefix: list[Step], tools: ToolBackend) -> Trace: ...
    def success(self, task: str, trace: Trace) -> bool: ...
```

- Oracle sources for step k (recorded in results, since they differ in strength):
  reference trajectory, human edit, or a stronger-model repair (weakest evidence).
- Estimand for a failed task and step k:
  `effect_k = P(success | prefix<k, do(step_k = oracle)) − P(success | prefix<k, natural step_k)`.
  Estimate both terms with R rollouts from the **same prefix**. Do not compare against the
  single original failure, which may be bad luck.
- Search strategy: coarse screen of all steps with R=2, then refine the top 3 with larger R.
- Outputs: per-step effects with CIs; **decisive step** = earliest step whose effect CI lower
  bound exceeds τ; across many failed tasks, a histogram of decisive steps by step type and
  tool name.
- `ReplayDivergenceError` if the replayed prefix does not reproduce the recorded steps.

### 3.7 `checks/` (Phase 4)

Deterministic, cheap checks that run before any judge: JSON schema, regex, user-provided
Python verifier callables, local NLI entailment (optional extra `causeval[nli]`, a DeBERTa-class
NLI cross-encoder via transformers). The metamorphic relation registry lives here and is shared
with `interventions/perturb.py`.

### 3.8 Observational causal estimation (Phase 7, optional)

For comparing versions from production logs without an A/B test. AIPW estimate of the
version effect on a metric, via DoWhy/EconML, with user-declared covariates (query length,
topic cluster, user segment). Always run DoWhy refuters (placebo treatment, random common
cause, data subset) and print a fixed warning that unobserved confounding cannot be ruled out.

## 4. Public API and CLI sketch

```python
from causeval import Experiment, MetricSpec

exp = Experiment(
    app=my_app,  # async callable or RAGApp
    dataset=goldens,  # list of DeepEval Goldens or dicts
    metrics=[MetricSpec("FaithfulnessMetric", {"threshold": 0.7})],
    repeats=5,
    seed=0,
)
run = await exp.a_run()
print(run.summary())  # estimate, CI, variance components, flaky items
```

```bash
causeval plan    --config eval.yaml --pilot runs/pilot.json --detect 0.03 --power 0.8
causeval run     --config eval.yaml --out runs/
causeval compare --baseline runs/a.json --candidate runs/b.json
causeval gate    --baseline runs/a.json --candidate runs/b.json --margin 0.02
causeval ground  --config rag.yaml                 # Phase 2
causeval audit-judge --config judge.yaml           # Phase 3
causeval attribute   --config agent.yaml           # Phase 5
```

## 5. Phases and acceptance criteria

Each phase ends with all offline tests green, ruff/mypy clean, `PROGRESS.md` updated.

**Phase 0 — Scaffold and baseline**
- uv project, src layout, GitHub Actions running offline tests on 3.10–3.12.
- `core` (schemas, cache, concurrency, cost, provenance, errors), `adapters`, `tests/fakes`.
- Accept: import-guard test; #3356-pattern concurrency test passes with the factory; cache hit
  test; failed judge call yields `error` not a score.
- Deliverable: `bench/baseline_variance.py` (live) runs 30 goldens × R=10 × 3 DeepEval metrics
  and writes a report of within-item variance and the fraction of items whose pass/fail
  flips across repeats. This quantifies the problem and becomes the README's motivation.

**Phase 1 — Statistics**
- `estimate`, `compare`, `gate`, `planner`, CLI `run/compare/gate/plan`.
- Accept: B1 passes (Section 6).

**Phase 2 — RAG causal grounding**
- `interventions/rag.py`, `cf_gen.py`, CLI `ground`.
- Accept: B2 passes.

**Phase 3 — Judge audit**
- `bias_probes`, `calibration`, `sampling`, `ppi`, `conformal`, `jury`, CLI `audit-judge`.
- Accept: B3 and B4 pass.

**Phase 4 — Perturbations and checks**
- `perturb.py`, `checks/`.
- Accept: on a fake app with an injected sensitivity to one attribute swap and one distractor
  type, the engine detects exactly those (CI excludes 0) and not the clean perturbations.

**Phase 5 — Agent attribution**
- `trace`, `cassette`, harness, attribution, CLI `attribute`.
- Accept: B5 passes.

**Phase 6 — IRT and adaptive sampling**
- Accept: B6 passes; adaptive allocation keeps CI coverage within B1 bounds.

**Phase 7 — Optional extensions**
- CoT faithfulness, observational causal, OTel import, framework harness adapters
  (LangGraph, OpenAI Agents, Pydantic AI), HTML report, pytest plugin.

## 6. Validation benchmark (`bench/`) — proving "better"

Each benchmark has an offline version (fakes, runs in CI) and a live version (real models,
run manually, results committed under `bench/results/`).

- **B1 Stats coverage.** Simulated scores with known μ and variance components. Over 1,000
  simulations: 95% CI coverage in [0.93, 0.97] for n ∈ {20, 50, 200}, R ∈ {1, 3, 10}; gate
  false-regression rate ≤ 0.05 when true Δ = 0; report power curves for Δ ∈ {0.01..0.1}.
- **B2 RAG grounding.** Two question sets: (a) a fictional knowledge base (invented company,
  invented policies) that no model can answer from memory; (b) well-known facts where the
  context agrees with world knowledge. Three app variants: grounded, parametric (ignores
  context), mixed. Accept: Counterfactual Adherence separates grounded from parametric items
  with AUROC ≥ 0.9. Also report DeepEval Faithfulness AUROC on the same split; the expected
  (to be verified, not assumed) result is that it cannot separate set (b).
- **B3 Judge bias recovery.** Fake judge with injected position bias 0.15 and verbosity bias
  +0.08. Accept: probes recover both within their CIs across 200 simulations with ~95%
  coverage, and report ~0 for non-injected biases. Live: publish real judge bias tables.
- **B4 PPI.** Synthetic judge with known systematic bias vs a known human mean. Accept: PPI CI
  covers the true human mean at ~95%; naive judge-mean CI coverage is reported (expected to be
  poor); PPI CI is narrower than human-only CI at equal label count.
- **B5 Agent attribution.** Reference agent in a toy tool environment (calculator, lookup,
  unit converter). Inject a fault at a known step k (wrong tool argument). Accept: decisive step
  equals k in ≥ 90% of tasks; offline with a scripted fake LLM, live with a real model.
- **B6 IRT.** Simulated 2PL data: parameter recovery correlation ≥ 0.9. Real data from ≥ 5
  systems: pruned set of 30–50% of items keeps system ranking Kendall τ ≥ 0.9.

## 7. Statistical correctness tests

Every function in `stats/`, `judge_audit/ppi.py`, `judge_audit/conformal.py` has a simulation
test with a known ground truth, marked `@pytest.mark.sim`. Fast versions (≤ 200 sims) run in
the offline suite; full versions (≥ 1,000 sims) run nightly. Coverage and error-rate
assertions use binomial tolerance bounds, not exact equality, so they are not flaky.

## 8. Cost model

Calls ≈ `n_items × n_metrics × R × n_conditions × calls_per_metric`. RAG grounding with
k chunks has `3 + k` conditions; agent attribution is `O(steps × R)` rollouts per failure.
The planner must show this before any live run. Caching makes re-analysis free.
IRT pruning and adaptive repeats exist mainly to bring these costs down.

## 9. Risks

- **Cost blow-up** — mitigated by planner, budget confirmation, cache, pruning.
- **Unnatural interventions** confound results — mitigated by validity checks; invalid
  interventions are dropped and counted.
- **Outcome functions that use a judge** inherit its bias — prefer deterministic outcomes;
  otherwise use a Phase 3-calibrated judge and say so in provenance.
- **Agents that cannot resume from a forced prefix** — attribution unavailable; document the
  harness protocol clearly and ship adapters incrementally.
- **Non-random human labels** break PPI — enforced by `sampling.py`.
- **DeepEval API churn** — pin `<5`, adapter tests run against the pinned version; a weekly
  CI job tests the latest release.

## 10. Open questions (resolve in PROGRESS.md)

1. Package name and license (default Apache 2.0, matching DeepEval).
2. Default judge model and provider for live benchmarks.
3. Whether to upstream the #3356 fix to DeepEval as a PR (recommended, low effort).
4. Which agent framework adapter to build first in Phase 7.

## 11. References

- Angelopoulos et al., *Prediction-Powered Inference*, arXiv:2301.09633; *PPI++*, arXiv:2311.01453.
- Angelopoulos et al., *Learn then Test*, arXiv:2110.01052; Angelopoulos & Bates, conformal
  prediction introduction, arXiv:2107.07511.
- Miller, *Adding Error Bars to Evals*, arXiv:2411.00640.
- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, arXiv:2306.05685.
- Lanham et al., *Measuring Faithfulness in Chain-of-Thought Reasoning*, arXiv:2307.13702.
- Longpre et al., *Entity-Based Knowledge Conflicts in Question Answering*, arXiv:2109.05052.
- Kusner et al., *Counterfactual Fairness*, arXiv:1703.06856.
- Polo et al., *tinyBenchmarks* (IRT for LLM evaluation), arXiv:2402.14992.
- DoWhy (py-why/dowhy), EconML (py-why/EconML), ppi_py (aangelopoulos/ppi_py).
- DeepEval issues referenced: #3356 (shared metric objects), #1613 (telemetry opt-out),
  discussion #1674 (score inconsistency).
