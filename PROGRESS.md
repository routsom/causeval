# Progress

Current phase: **All core phases (0-6) done; Phase 7 causal extensions done (framework adapters/HTML/pytest-plugin deferred)**

## Phase status

- [x] Phase 0 — Scaffold and baseline
  - [x] uv project, src layout, ruff/mypy/pytest config, GitHub Actions (offline tests, 3.10–3.12)
  - [x] `core/`: schemas, llm_cache, concurrency, cost planner, provenance, errors
  - [x] `adapters/`: deepeval_import guard, metric_factory, judge_model adapter
  - [x] `tests/fakes/`: fake judge (bias + noise knobs), fake RAG app, fake agent
  - [x] Tests: import guard, #3356 concurrency pattern, cache hit, failed-call handling
  - [x] `bench/baseline_variance.py` (live + offline smoke) and first baseline report
        (offline smoke committed; live report pending API keys)
- [x] Phase 1 — Statistics (B1)
  - [x] `stats/estimate.py` — cluster bootstrap CI (studentized/percentile/BCa), one-way
        random-effects ANOVA variance components, flaky detection, small-sample t-interval
  - [x] `stats/compare.py` — paired cluster bootstrap of Δ, Wilcoxon, Holm-adjusted CI
        levels across metrics, McNemar + pass-rate delta
  - [x] `stats/gate.py` — three-valued non-inferiority verdict, Holm, exit codes
  - [x] `stats/planner.py` — power-based (n, R) planning from a pilot pair
  - [x] `Experiment` orchestrator + `RunResult.summary()`; CLI `run/compare/gate/plan`
  - [x] B1 sims: CI coverage grid, false-regression ≤ 0.05 at Δ=0, power curve (all @sim)
- [x] Phase 2 — RAG causal grounding (B2)
  - [x] `interventions/cf_gen.py` — single-fact counterfactual edits, local/contradiction/
        naturalness validity gates, injectable LLM steps, JSONL overrides loader
  - [x] `interventions/rag.py` — full/none/loo_k/cf conditions, Context Reliance,
        Counterfactual Adherence, per-chunk effects, grounded/parametric/confabulating/mixed
        classification, false-faithful flag, cluster-bootstrap CIs
  - [x] CLI `ground` (config-driven, resolves a RAGApp or factory, writes JSON + summary)
  - [x] `bench/rag_grounding.py` + B2 acceptance test (CA AUROC 1.000 offline, ≥0.9 under noise)
- [x] Phase 3 — Judge audit, calibration, PPI, conformal (B3, B4)
  - [x] `judge_audit/bias_probes.py` — position (pairwise, both orders) + paired-transform
        probes (verbosity/formatting/authorship) with paired cluster-bootstrap CIs and flagging
  - [x] `judge_audit/calibration.py` — Spearman, Cohen's kappa, confusion, PAVA isotonic map
        (no sklearn), ECE with bootstrap CI
  - [x] `judge_audit/sampling.py` — simple + stratified random labeling plans (recorded seed),
        JSONL labeling-sheet export/import; feeds PPI's random-sample guard
  - [x] `judge_audit/ppi.py` — PPI++ mean with power-tuning λ, effective-n, random-sample
        guard; `ppi_py` numerical cross-check (CI-only; wheel won't build locally)
  - [x] `judge_audit/conformal.py` — Learn-then-Test selective judging (fixed grid + Bonferroni
        Clopper-Pearson), confidence from logprobs or k-sample agreement
  - [x] `judge_audit/jury.py` — k-judge aggregation (mean/majority), Krippendorff α, disagreement
  - [x] CLI `audit-judge` (calibration + optional PPI from labeled/unlabeled JSONL)
  - [x] B3 (bias recovery) + B4 (PPI) benchmarks and acceptance tests; sim tests for PPI,
        conformal risk-control, and B3 coverage
- [x] Phase 4 — Perturbations and deterministic checks
  - [x] `checks/deterministic.py` — minimal dependency-free JSON Schema subset, regex,
        user verifier callables, normalized-equivalence, optional NLI equivalence (nli extra)
  - [x] `checks/relations.py` — shared metamorphic-relation registry + built-in transforms
        (paraphrase, formatting, typo, reorder MC options, attribute-swap factory, distractor)
  - [x] `interventions/perturb.py` — engine: invariance rate + paired metric-effect CI +
        Holm-adjusted fairness gaps for attribute swaps; invalid perturbations dropped & counted
  - [x] Acceptance test: fake app sensitive to one attribute swap + one distractor; engine
        flags exactly those (CI excludes 0) and none of the clean perturbations
- [x] Phase 5 — Agent attribution (B5)
  - [x] `attribution/trace.py` — Trace/Step schema, `from_records`, canonical args, defensive
        `from_deepeval_trace` importer
  - [x] `attribution/harness.py` — `AgentHarness`/`ToolBackend` protocols, `ToolSpec`,
        `DictToolBackend`
  - [x] `attribution/cassette.py` — record/replay with the novel-input policy (safe_live live,
        side-effecting blocked, simulator fallback) + `verify_prefix` divergence check
  - [x] `attribution/attribute.py` — counterfactual step effects (same-prefix R rollouts),
        coarse-screen-then-refine, earliest-decisive-step, `AttributionCase`, histogram
  - [x] CLI `attribute` (resolves an `AttributionCase` factory)
  - [x] B5 benchmark + acceptance test: fault injected at a known step; decisive step == fault
        step in 100% of tasks offline (target ≥90%)
- [x] Phase 6 — IRT and adaptive sampling (B6)
  - [x] `stats/irt.py` — native 2PL joint-MAP fit (L-BFGS, analytic gradient, N(0,1) ability
        prior), Fisher information, `prune` (top-info items), `rank_systems`
  - [x] `stats/adaptive.py` — Neyman repeat allocation (∝ within-item sd), zero-variance skip,
        per-item cap with overflow redistribution
  - [x] B6 benchmark + acceptance: 2PL recovery corr ≥ 0.9 (b/a/θ) on simulated data; pruning
        to 50% keeps system ranking Kendall τ ≥ 0.9; adaptive allocation keeps CI coverage
        within B1 bounds (coverage sim)
  - [x] `girth` cross-check (CI-only; wheel not installed locally)
- [x] Phase 7 — Optional extensions (causal subset; framework adapters/HTML/pytest-plugin deferred)
  - [x] `stats/observational.py` — AIPW doubly-robust ATE with K-fold cross-fitting, plus
        placebo / random-common-cause / data-subset refuters and the fixed
        unobserved-confounding warning
  - [x] `interventions/cot.py` — CoT faithfulness: early-answering agreement curve + AOC,
        mistake-insertion sensitivity, cluster-bootstrap CIs (needs prefill+visible reasoning)
  - [x] `attribution/trace.py:from_otel_spans` — OpenTelemetry GenAI span JSON importer
  - [ ] Deferred (opt-in, need uninstalled frameworks): LangGraph/OpenAI-Agents/Pydantic-AI
        harness adapters, HTML report, pytest plugin

## Decisions log

<!-- Newest first. Format: YYYY-MM-DD — decision — reason -->
- 2026-09-30 — Phase 7 scoped to the **causal extensions** (observational AIPW, CoT faithfulness, OTel import); framework harness adapters, HTML report, and pytest plugin deferred. Per the "don't build machinery without a concrete need" rule: the deferred items need uninstalled frameworks (LangGraph/OpenAI-Agents/Pydantic-AI) and are integration/presentation surface that can't be meaningfully tested offline, whereas the causal pieces are thesis-central and fully offline-testable. User approved this scope.
- 2026-09-30 — Observational effect uses **cross-fit AIPW** (doubly robust), not plain IPW or a single outcome model. It's consistent if *either* the propensity or the outcome model is right, and K-fold cross-fitting keeps it root-n valid without assuming the nuisance models are perfect. Nuisances are ridge logistic (propensity) + ridge OLS per arm (outcomes), implemented natively; DoWhy/EconML remain optional and are not required for the estimator. Refuters are behavioural checks (placebo→CI covers 0, random-common-cause→estimate stable, subset→stable), and the result always carries the fixed "unobserved confounding cannot be ruled out" warning.
- 2026-09-29 — 2PL fit is a **joint MAP via L-BFGS with an N(0,1) ability prior**, not the alternating coordinate ascent I first wrote. Coordinate ascent that re-standardised θ every iteration oscillated and never converged (poor recovery). Joint L-BFGS over all params with analytic gradients converges; the ability prior pins the scale/location the likelihood leaves free, and θ is standardised once at the end. `girth` (MML) stays a CI-only cross-check.
- 2026-09-29 — B6 **recovery uses many systems (500), pruning uses few well-separated systems (10)**. Discrimination `a_i` is estimated from `n_systems` observations, so ≥0.9 recovery needs hundreds of systems (a consistency demonstration); the SPEC's "≥5 systems" is the identifiability floor, which the pruning test exercises. Pruning ranks by mean score, so near-tied random abilities reshuffle under subsetting; using evenly-spaced abilities (the realistic "systems of interest") and keeping 50% gives τ ≥ 0.9 robustly (min 0.91 over 10 seeds).
- 2026-09-29 — Adaptive allocation is **Neyman (∝ within-item sd)** and its coverage is confirmed by simulation, per SPEC's warning about data-dependent allocation. The grand-mean-of-item-means estimator stays unbiased under any allocation and the cluster bootstrap resamples the realised item means, so coverage holds (sim: ≥0.90 offline).
- 2026-09-29 — **Decisive step = earliest step whose effect CI lower bound clears τ**, per SPEC. Intervening `do(step_k = oracle)` on a *downstream* step also repairs the outcome (forcing a later value fixes the final answer), so multiple steps can show a large effect; taking the earliest returns the root cause, not a symptom. Verified: B5 decisive-step accuracy is 100% offline with the fault placed uniformly across the 3 plan steps.
- 2026-09-29 — Attribution estimates **both terms (do-oracle and natural) with fresh R rollouts from the same prefix**, never against the single recorded failure (SPEC: "may be bad luck"). The reference agent injects independent execution noise (`flip=0.1`) so rollouts vary and the effect CIs are real rather than degenerate; effect CI is a normal-approx difference of two Bernoulli rates.
- 2026-09-29 — B5's toy tools are deterministic read-only (`safe_live=True`), so the offline benchmark calls them live and the cassette record/replay + novel-input policy is exercised by dedicated `test_cassette.py` tests rather than inside the B5 loop. Keeps the acceptance focused on the attribution statistics while still delivering+testing the cassette module.
- 2026-09-29 — `from_deepeval_trace` is a **thin defensive adapter** (`getattr` for spans/input/output), not a hard-coded DeepEval trace schema — respects CLAUDE.md rule 9 (read the installed source, don't guess) given DeepEval's trace API churn; the generic `from_records` path is what the harness/tests use and is fully covered.
- 2026-09-29 — The **metamorphic-relation registry lives in `checks/relations.py`** and `interventions/perturb.py` imports it (not the reverse) — SPEC says the registry is "shared with perturb.py". `interventions` may depend on `checks` (both sit downstream of `stats`; only `stats` has an import restriction), and `checks` stays free of `interventions`. Transforms operate on a tiny `Example(question, contexts)` so the registry needs no `interventions` types.
- 2026-09-29 — Perturbation **validity = "not a no-op"** by default (a transform that leaves the input unchanged is inapplicable and is dropped+counted), rather than always running an NLI meaning-preserving check. Our built-ins are meaning-preserving by construction; live runs plug a bidirectional-NLI/judge validity fn into the relation. This keeps Phase 4 fully offline and makes "attribute token absent" / "no context for a distractor" cleanly droppable.
- 2026-09-29 — JSON Schema validation is a **hand-written minimal subset** (type/required/properties/items), not `jsonschema` — avoids adding a dependency for the small structured-output check we need; `bool` is explicitly rejected where `number`/`integer` is required (Python quirk). NLI equivalence stays behind the `nli` extra with a guarded `transformers` import.
- 2026-09-29 — Conformal selective judging uses a **fixed threshold grid + Bonferroni** Clopper-Pearson bounds, not a fixed-sequence-from-the-top test. SPEC names "Learn-then-Test with a Clopper-Pearson tail bound"; a top-down fixed sequence stops at the very first (highest) threshold, which accepts ~1 item and has a CP bound near 1, so it would abstain on everything. A pre-specified grid with Bonferroni (`delta/n_grid`) controls the family-wise error so selecting the lowest safe threshold is valid; the sim test confirms P(true error > α) ≤ δ.
- 2026-09-29 — Isotonic calibration uses a hand-written **PAVA** (pool-adjacent-violators), not sklearn — avoids adding scikit-learn as a dependency for one function; the map is stored as (x, y) knots with clamped step interpolation and refit inside the ECE bootstrap.
- 2026-09-29 — B3's synthetic judge and dataset get **independent seed streams** (`SeedSequence.spawn`). Seeding the judge's noise and the item qualities from the same seed correlated them and inflated the recovered position bias to 0.21 (true 0.15); independent streams recover 0.15 at ~95% coverage. General rule for future benchmarks: never share a seed between the data-generating process and the noise process.
- 2026-09-29 — PPI CI is implemented directly (PPI++ mean, `λ* = Cov(Y,f)/(Var(f)(1+n/N))`) and **cross-checked against `ppi_py`** in a CI-only test (its `numba`/`llvmlite` wheel won't build on macOS x86_64, same class as the `nli` extra); the point estimate + two-term variance match `ppi_py.ppi_mean_ci` exactly, whose power-tuning kwarg is `lhat`.
- 2026-09-29 — Counterfactual Adherence (`P(follows_cf|cf)`), not Context Reliance, is B2's primary grounding signal — CR conflates "context helps" with "answer depends on context"; a parametric model that already knows the fact has CR≈0 but so does a grounded model given redundant context. CA intervenes directly on the fact, so it separates grounded from parametric behaviour cleanly (offline AUROC 1.000, ≥0.9 at 0.8/0.2 follow rates). Faithfulness AUROC on set (b) is a live-only comparison (expected to fail to separate), not asserted offline.
- 2026-09-29 — `edit_is_local` uses prefix/suffix matching around the value span, **not** difflib opcodes — old/new values that share characters (`2019`→`1994` share `19`) make an opcode diff report multiple edit regions and wrongly reject a valid local edit. Anchoring on the first occurrence of `original_value` and checking the surrounding text is unchanged is exact for single-fact edits.
- 2026-09-29 — Invalid counterfactuals are **skipped and counted** (`Counterfactual.skipped(reason)`), never raised, so one bad edit doesn't abort a grounding run; `a_ground` records `n_cf_skipped` and those items simply carry `counterfactual_adherence=None`. Matches the CLAUDE.md rule to record a failed intervention explicitly rather than substitute a default.
- 2026-09-29 — Default CI method is the **studentized** (bootstrap-t) cluster bootstrap, not percentile — SPEC §3.3 names percentile as default with BCa optional, but both undercover a symmetric statistic (the mean) at n=20 and fail B1's [0.93, 0.97] target. Empirical coverage at n=20 (2000 sims): percentile 0.924, BCa 0.926, t-interval 0.951, bootstrap-t 0.950. Bootstrap-t is second-order accurate for the mean, keeps the clustered resampling, and passes B1. `percentile` and `bca` remain selectable via `method=`.
- 2026-09-29 — Holm across metrics is applied as **Holm-adjusted confidence levels** inside `compare()` (ordered by Wilcoxon p), so the returned CI + verdict already reflect the family correction — matches SPEC's "Gate uses Holm-adjusted confidence levels"; keeps all the paired-diff data in one place.
- 2026-09-29 — `Experiment` calls the app once per item, then repeats only the metric measurement R times — isolates judge/measurement noise for Phase 1. App-side interventions (RAG ablation, perturbations) resample the input in later phases.
- 2026-09-29 — License Apache 2.0; package name kept as `causeval` for now — matches DeepEval; resolves open Q1's license half.
- 2026-09-29 — `SPEC.md`/`PROGRESS.md` live at repo root, not `docs/` — updated CLAUDE.md's pointer to match reality.
- 2026-09-29 — mypy `python_version = "3.12"` (not 3.10) — numpy 2.x stubs use `type` alias syntax mypy rejects below 3.12. Runtime 3.10 compat is enforced by ruff (`target-version = py310`) + the CI test matrix instead.
- 2026-09-29 — deepeval treated as untyped in mypy (`follow_imports = skip`) — it ships `py.typed` but is only partially typed, which breaks strict mode; we read its source directly per rule 9.
- 2026-09-29 — Commit `uv.lock`; CI uses `uv sync --frozen` — reproducible installs. `nli` extra (torch) has no macOS x86_64 wheel, so `--all-extras` fails locally; core+dev sync is clean.
- 2026-09-28 — Wrap DeepEval (`>=4.2,<5`) instead of forking — upstream moves too fast to maintain a fork.

## Open questions

<!-- Copy from SPEC §10 as they are resolved or raised. -->
1. ~~Package name and license~~ — license = Apache 2.0 (done); name `causeval` provisional, rename freely.
2. ~~Default judge model/provider for live benchmarks~~ — used Anthropic **Claude Haiku 4.5** (and 5 other Claude models for B6); live baseline + B2-B6 reports committed under `bench/results/*_live.*`.
3. Upstream the #3356 fix as a DeepEval PR?
4. First agent framework adapter for Phase 7.
5. `torch`/`nli` extra has no wheel for macOS x86_64 — pin a compatible torch or document Linux-only for the `nli` extra.

## Session notes

<!-- Short notes per session: what changed, what's next, anything surprising. -->
- 2026-10-01 (live benchmarks) — Ran all live benchmarks against Anthropic Claude (Haiku 4.5; B6 used 6 Claude models as systems). Reports committed under `bench/results/*_live.*` and summarized in the README live table. Headlines: **B3** position bias −0.29 (picks the 2nd answer ~79% of the time) + verbosity −0.06, both flagged; **B2** CA AUROC 0.833 separating fictional (grounded) vs well-known (parametric), with real sycophancy on 2/12 items; **B4** PPI unbiased with effective-n ~41 from 12 labels (naive judge happened to be well-calibrated on clean items); **B5** decisive-step 4/5 (real agents self-correct injected faults — a robustness finding — and the SDK build rejects temperature=0); **B6** pruning keeps τ=1.0 but frontier models cluster near ceiling. Baseline: clean items 1.0/flip0; borderline items wide CIs + a faithfulness-leniency signal. All honest, with caveats in the README footnotes.
  - Infra notes for future live runs: the installed `anthropic` SDK build **rejects the `temperature` kwarg** on `messages.create` (and DeepEval's AnthropicModel); omit it or use `generation_kwargs`. The key is an `sk-ant-usr-` (workspace-scoped) key — it works for Messages but not the Admin API. Inline-key Bash commands get recorded into `.claude/settings.local.json` (gitignored, never committed) — scrub it after. All `bench/results/*_live.*` were key-scanned before commit.
- 2026-09-30 (Phase 7 causal extensions complete) — Built `stats/observational.py` (cross-fit AIPW ATE + placebo/random-common-cause/subset refuters + unobserved-confounding warning), `interventions/cot.py` (CoT faithfulness: early-answering AOC + mistake sensitivity), and `attribution/trace.py:from_otel_spans` (OTel GenAI import). 186 offline tests pass (10 new), ruff/format/mypy strict clean. Scope approved by user: framework harness adapters (LangGraph/OpenAI-Agents/Pydantic-AI), HTML report, and pytest plugin are deferred (need uninstalled frameworks; integration/presentation surface, not offline-testable).
  - AIPW sim confirms it covers the true ATE (~0.95) on confounded logs where the naive difference-in-means coverage collapses (≤0.5). CoT: a reasoning-driven fake model shows high AOC + mistake sensitivity, a fixed-answer model shows ~0 for both.
  - All core phases 0-6 + benchmarks B1-B6 are done; this closes the SPEC's implementable scope. The remaining Phase 7 items are opt-in integrations to add when a concrete framework/target exists.
- 2026-09-29 (Phase 6 complete) — Built `stats/irt.py` (native 2PL joint-MAP fit + Fisher-info pruning + system ranking) and `stats/adaptive.py` (Neyman repeat allocation), plus the B6 benchmark + report. 176 offline tests pass (17 new), ruff/format/mypy strict clean. Acceptance: 2PL recovery b=0.99 / a=0.95 / θ=0.98 (all ≥0.9); pruning to 50% keeps ranking Kendall τ = 1.0 at seed 0 (min 0.91 over seeds); adaptive coverage sim holds within B1 bounds.
  - Surprise: my first IRT fit (alternating coordinate ascent with per-iteration θ re-standardisation) oscillated and never converged. Switched to joint L-BFGS MAP with an N(0,1) ability prior and analytic gradients - converges, and the prior pins the metric the likelihood leaves free. Discrimination is the data-hungry parameter (needs hundreds of systems for ≥0.9 recovery); pruning uses a few well-separated systems instead. See decisions log.
  - `girth` cross-check written but skips locally (wheel not installed, like ppi_py/transformers); runs in CI.
  - Next: Phase 7 optional extensions (CoT faithfulness `interventions/cot.py`, OTel trace import, observational causal estimation via DoWhy/EconML). No new acceptance benchmark; these are opt-in. All six core phases (0-6) and their benchmarks (B1-B6) are now done.
- 2026-09-29 (Phase 5 complete) — Built the whole `attribution/` layer (trace, harness protocols, record/replay cassette, counterfactual attribution engine), the CLI `attribute`, and the B5 benchmark + report. 159 offline tests pass (17 new), ruff/format/mypy strict clean. Acceptance met: with a wrong-argument fault injected at a known step, the decisive step equals the fault step in **100%** of tasks offline (target ≥90%), across faults placed uniformly over the 3 plan steps.
  - Key correctness lever: decisive step = *earliest* step clearing τ. Intervening on a downstream step also fixes the outcome, so several steps can look causal; earliest = root cause. Both effect terms use fresh same-prefix R rollouts (never the single recorded failure), with `flip=0.1` execution noise so the CIs are real. See decisions log.
  - Cassette record/replay + novel-input policy (safe_live/side-effecting/simulator) is delivered and unit-tested; B5 itself uses live deterministic tools. `from_deepeval_trace` is a defensive adapter (rule 9).
  - Next: Phase 6 IRT + adaptive sampling (B6) — `stats/irt.py` (2PL fit, likely via the `girth` extra), item pruning that preserves system ranking (Kendall τ ≥ 0.9), adaptive repeats. Acceptance: simulated 2PL parameter-recovery correlation ≥ 0.9; on ≥5 systems a 30-50% pruned set keeps ranking τ ≥ 0.9. Write the parameter-recovery simulation test first (statistical).
- 2026-09-29 (Phase 4 complete) — Built `checks/` (deterministic checks + the shared metamorphic-relation registry) and `interventions/perturb.py` (the perturbation engine). 142 offline tests pass (20 new), ruff/format/mypy strict clean. Acceptance met: on a fake app with an injected sensitivity to the doctor→nurse swap and a URGENT distractor, the engine flags exactly those two (metric-effect CI excludes 0) and none of the six clean perturbations; the attribute-swap fairness gaps are Holm-adjusted (sensitive p_adj < 0.05, benign = 1.0).
  - No new B-letter benchmark for Phase 4 (SPEC's benchmark list jumps B4→B5); the acceptance is the injected-sensitivity test above, kept deterministic and offline.
  - `checks` hosts the relation registry; `perturb` imports it (dependency direction noted in the decisions log). NLI equivalence is behind the `nli` extra (guarded `transformers` import), added to the mypy untyped-module overrides.
  - Next: Phase 5 agent attribution (B5) — `attribution/trace.py` (Trace/Step, DeepEval-trace importer), deterministic replay, step-level blame. Inject a fault at a known step k in a toy tool env (calculator/lookup/unit-converter); accept: decisive step == k in ≥90% of tasks. Offline with a scripted fake LLM. The fake agent already exists in `tests/fakes/fake_agent.py`.
- 2026-09-29 (Phase 3 complete) — Built the whole `judge_audit/` layer (bias_probes, calibration, sampling, ppi, conformal, jury), the CLI `audit-judge`, and the B3/B4 benchmarks + reports. 122 offline tests pass (36 new), ruff/format/mypy strict clean. Acceptance: **B3** recovers injected position 0.15 (coverage ~0.96) and verbosity 0.08 (~0.93) inside their CIs, non-injected ~0; **B4** PPI covers the true mean (~0.98, conservative), naive judge-mean coverage 0.00 (biased), PPI CI ~half the human-only width.
  - Surprise worth remembering: sharing a seed between a synthetic judge and its dataset correlates the judge's noise with the item qualities and biases the recovered effect (0.15 → 0.21). Fixed with `SeedSequence.spawn`; see decisions log. Any future synthetic benchmark should spawn independent seed streams.
  - Conformal LTT needed a fixed grid + Bonferroni, not fixed-sequence-from-the-top (which abstains on everything because the top threshold accepts ~1 item). Sim test verifies the risk-control guarantee.
  - `ppi_py` cross-check is written but skips locally (numba/llvmlite wheel won't build on macOS x86_64, like the `nli` extra); it runs in CI. Formula verified by hand against `ppi_py.ppi_mean_ci` (kwarg `lhat`).
  - Next: Phase 4 perturbations + deterministic checks (`interventions/` perturbations, `checks/` schema/NLI/metamorphic). No new acceptance benchmark letter for Phase 4; keep the metamorphic-relation and check-gate tests deterministic and offline.
- 2026-09-29 (Phase 2 complete) — Built `interventions/cf_gen.py` + `interventions/rag.py`, CLI `ground`, and `bench/rag_grounding.py` with the B2 acceptance test. 86 offline tests pass (27 new), ruff/format/mypy strict clean. B2 CA AUROC = 1.000 offline (grounded vs parametric), ≥0.9 under 0.8/0.2 follow-rate noise; report committed under `bench/results/`.
  - CA is the primary signal, not CR (see decisions log). `edit_is_local` uses prefix/suffix anchoring, not difflib, because shared characters between old/new values (`2019`→`1994`) fool an opcode diff. Invalid counterfactuals are skipped-and-counted, never raised.
  - LLM-backed cf-generation steps (span extraction, replacement, contradiction/naturalness checks) are injectable callables so the engine is fully testable offline; the deterministic `contains_outcome`/`follows_cf_outcome` outcome fns cover the offline path, with judge-based variants deferred to live.
  - Next: Phase 3 judge audit — bias probes (position 0.15, verbosity +0.08), human calibration, PPI, conformal abstention; B3/B4. Write the recovery-simulation tests first (statistical, so test-first).
- 2026-09-29 (Phase 1 complete) — Built `stats/` (estimate, compare, gate, planner) + the `Experiment` orchestrator + CLI `run/compare/gate/plan`. 59 offline tests pass (34 new), ruff/format/mypy strict clean. B1 sims are `@pytest.mark.sim` with lenient binomial bounds offline; the strict [0.93,0.97] / ≥1000-sim versions are the nightly target (SPEC §7).
  - Surprise worth remembering: percentile/BCa bootstrap **undercover the mean at n=20** (~0.925) and miss B1; switched the default to the studentized bootstrap-t (see decisions log with the empirical table). Any future estimator work should keep coverage on the (n,R) grid in mind.
  - CLI verified end-to-end via the console script (`causeval gate` → exit 1 on a real regression; `causeval plan` renders the (n,R) table). Fake-metric runs have zero within-variance, so their CIs are degenerate (zero-width) — expected; real judges will have noise.
  - Next: Phase 2 RAG causal grounding — `interventions/rag.py`, `cf_gen.py`, CLI `ground`, and B2 (Counterfactual Adherence separates grounded vs parametric with AUROC ≥ 0.9). The `RAGApp` protocol + fake already exist in `tests/fakes/fake_rag.py`.
- 2026-09-29 (Phase 0 complete) — Scaffolded the whole `src/causeval` layout, `core/` + `adapters/`, offline fakes, and the four acceptance tests (import guard, #3356 concurrency, cache hit, failed-call). All offline tests (25) + ruff + ruff format + mypy strict pass. Baseline benchmark runs offline (synthetic smoke committed under `bench/results/`); the live variant needs a judge model + API keys (open Q2).
  - DeepEval 4.2.6 inspected directly: custom `DeepEvalBaseLLM.a_generate` may return a plain string; DeepEval parses it against any schema via `trimAndLoadJson` (see `adapters/judge_model.py`). `BaseMetric` exposes `score`/`reason`/`success`/`error`/`evaluation_cost`; a soft failure sets `metric.error` without raising, which `a_measure_once` maps to a failed `Measurement`.
  - Next: Phase 1 stats (`estimate`, `compare`, `gate`, `planner`) + B1 simulation tests. Write the simulation test first (rule: statistical tasks are test-first).
