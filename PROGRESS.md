# Progress

Current phase: **3 done; starting 4 — Perturbations and deterministic checks**

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
- [ ] Phase 4 — Perturbations and deterministic checks
- [ ] Phase 5 — Agent attribution (B5)
- [ ] Phase 6 — IRT and adaptive sampling (B6)
- [ ] Phase 7 — Optional extensions

## Decisions log

<!-- Newest first. Format: YYYY-MM-DD — decision — reason -->
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
2. Default judge model/provider for live benchmarks (needed to produce the live baseline report).
3. Upstream the #3356 fix as a DeepEval PR?
4. First agent framework adapter for Phase 7.
5. `torch`/`nli` extra has no wheel for macOS x86_64 — pin a compatible torch or document Linux-only for the `nli` extra.

## Session notes

<!-- Short notes per session: what changed, what's next, anything surprising. -->
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
