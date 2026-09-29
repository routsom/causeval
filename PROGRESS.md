# Progress

Current phase: **1 done; starting 2 — RAG causal grounding**

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
- [ ] Phase 2 — RAG causal grounding (B2)
- [ ] Phase 3 — Judge audit, calibration, PPI, conformal (B3, B4)
- [ ] Phase 4 — Perturbations and deterministic checks
- [ ] Phase 5 — Agent attribution (B5)
- [ ] Phase 6 — IRT and adaptive sampling (B6)
- [ ] Phase 7 — Optional extensions

## Decisions log

<!-- Newest first. Format: YYYY-MM-DD — decision — reason -->
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
- 2026-09-29 (Phase 1 complete) — Built `stats/` (estimate, compare, gate, planner) + the `Experiment` orchestrator + CLI `run/compare/gate/plan`. 59 offline tests pass (34 new), ruff/format/mypy strict clean. B1 sims are `@pytest.mark.sim` with lenient binomial bounds offline; the strict [0.93,0.97] / ≥1000-sim versions are the nightly target (SPEC §7).
  - Surprise worth remembering: percentile/BCa bootstrap **undercover the mean at n=20** (~0.925) and miss B1; switched the default to the studentized bootstrap-t (see decisions log with the empirical table). Any future estimator work should keep coverage on the (n,R) grid in mind.
  - CLI verified end-to-end via the console script (`causeval gate` → exit 1 on a real regression; `causeval plan` renders the (n,R) table). Fake-metric runs have zero within-variance, so their CIs are degenerate (zero-width) — expected; real judges will have noise.
  - Next: Phase 2 RAG causal grounding — `interventions/rag.py`, `cf_gen.py`, CLI `ground`, and B2 (Counterfactual Adherence separates grounded vs parametric with AUROC ≥ 0.9). The `RAGApp` protocol + fake already exist in `tests/fakes/fake_rag.py`.
- 2026-09-29 (Phase 0 complete) — Scaffolded the whole `src/causeval` layout, `core/` + `adapters/`, offline fakes, and the four acceptance tests (import guard, #3356 concurrency, cache hit, failed-call). All offline tests (25) + ruff + ruff format + mypy strict pass. Baseline benchmark runs offline (synthetic smoke committed under `bench/results/`); the live variant needs a judge model + API keys (open Q2).
  - DeepEval 4.2.6 inspected directly: custom `DeepEvalBaseLLM.a_generate` may return a plain string; DeepEval parses it against any schema via `trimAndLoadJson` (see `adapters/judge_model.py`). `BaseMetric` exposes `score`/`reason`/`success`/`error`/`evaluation_cost`; a soft failure sets `metric.error` without raising, which `a_measure_once` maps to a failed `Measurement`.
  - Next: Phase 1 stats (`estimate`, `compare`, `gate`, `planner`) + B1 simulation tests. Write the simulation test first (rule: statistical tasks are test-first).
