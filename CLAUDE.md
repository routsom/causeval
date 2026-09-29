# causeval

A statistically rigorous, causal evaluation layer for LLM apps, built on top of DeepEval.
DeepEval measures. causeval adds three things it lacks: **uncertainty** (is the score real?),
**causality** (what caused this output or failure?), and **judge validity** (can we trust the judge?).

`causeval` is a working name (free on PyPI as of Sep 2026). Rename freely.

## Read before working

- `SPEC.md` — full design, module contracts, phase acceptance criteria. Read the section
  for the current phase before writing code.
- `PROGRESS.md` — phase status and decisions log. Update it at the end of every session.

Work phases in order. Do not start a phase until the previous phase's acceptance criteria pass.

## Commands

```bash
uv sync --all-extras                 # install
uv run pytest -m "not live"          # fast offline tests (default, must always pass)
uv run pytest -m live                # real LLM calls; needs API keys; run only when asked
uv run ruff check . && uv run ruff format --check .
uv run mypy src/causeval             # strict on src/, relaxed on tests/
uv run python -m causeval.bench      # validation benchmark (Phase 0+)
```

## Architecture (src/causeval/)

```
adapters/       DeepEval wrapper: metric factory, safe import, judge model adapter
core/           result schemas (pydantic), LLM call cache, cost planner, provenance
stats/          repeated sampling, clustered bootstrap, paired tests, gates, IRT
interventions/  RAG context ablation + counterfactual context, perturbations, CoT tests
judge_audit/    bias audits, human calibration, PPI estimates, conformal abstention
attribution/    agent trace recording, deterministic replay, step-level blame
checks/         deterministic checks: schema, NLI, metamorphic relations
bench/          validation benchmark with known ground truth
```

Dependency direction: `core` <- `adapters` <- `stats` <- everything else. `stats` never
imports from `interventions`, `judge_audit` or `attribution`.

## Non-negotiable rules

1. **Wrap DeepEval, never fork it.** Depend on `deepeval>=4.2,<5`. Do not patch its internals
   or monkeypatch its classes. If you must copy DeepEval code, keep its Apache 2.0 header and
   record the file and the change in `NOTICE`.
2. **Import DeepEval only through `causeval.adapters.deepeval_import`.** That module sets
   `DEEPEVAL_TELEMETRY_OPT_OUT=1` and `DEEPEVAL_DISABLE_DOTENV=1` (unless the user already set
   them) *before* the first `import deepeval`. No other module may import deepeval directly.
   A test enforces this.
3. **Fresh metric object per measurement.** Always build metrics via the adapter factory,
   never share an instance across concurrent calls (see DeepEval issue #3356).
4. **Never report a bare score.** Every public result carries `n_items`, `n_repeats`,
   a confidence interval, the method used, and provenance. No exceptions in CLI output either.
5. **No telemetry, no network calls, no side effects at import time.** Nothing is sent
   anywhere except the LLM calls the user configured.
6. **No hardcoded model prices or model names in logic.** Prices come from user config.
   Model names appear only in config defaults and examples.
7. **Offline tests must not call real LLMs.** Use `tests/fakes/` (a fake judge with
   configurable bias and noise, a fake RAG app, a fake agent). Live tests get `@pytest.mark.live`.
8. **Every statistical method needs a simulation test** proving its coverage or error rate
   on synthetic data with a known answer (see SPEC "Statistical correctness tests").
9. **Don't guess DeepEval's API.** Read the installed source under
   `.venv/lib/python*/site-packages/deepeval/` before writing adapter code. Key surfaces:
   `deepeval.metrics.base_metric.BaseMetric` (`measure`, `a_measure`, `score`, `reason`,
   `threshold`, `async_mode`), `deepeval.test_case.LLMTestCase`,
   `deepeval.models.base_model.DeepEvalBaseLLM`.

## Conventions

- Python 3.10+, full type hints, pydantic v2 models for all public results.
- Async-first: public APIs are `async def a_*` with a thin sync wrapper. All LLM calls go
  through `core.llm_cache` with a global semaphore (default concurrency 8).
- Every LLM call is cached by content hash (model, params, messages, seed) in SQLite at
  `.causeval/cache.db`. Repeated samples use distinct `repeat_index` in the key on purpose.
- Randomness: every function that samples takes `rng: np.random.Generator`. No global seeds.
- Numerics: numpy/scipy. No pandas in `src/` (fine in `bench/` and notebooks).
- Errors: raise typed exceptions from `core.errors`. Never swallow a failed judge call and
  substitute a default score; record it as a failed measurement and exclude it explicitly.
- Public functions get docstrings stating the estimand (what exactly is being estimated).

## Workflow per task

1. Read the relevant SPEC section and the current `PROGRESS.md`.
2. Write the simulation or unit test first when the task is statistical.
3. Implement, run offline tests, ruff, mypy.
4. Update `PROGRESS.md`: what changed, decisions made, open questions.
5. Commit with a conventional message (`feat(stats): ...`, `test(interventions): ...`).

If a SPEC requirement looks wrong or infeasible, stop and write the concern in
`PROGRESS.md` under "Open questions" instead of silently changing the design.
