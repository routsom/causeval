# Contributing to causeval

Thanks for your interest in improving causeval. This project turns LLM evaluation scores into
defensible evidence, so the bar for correctness - especially statistical correctness - is high.
This guide explains how to set up, what the non-negotiable rules are, and how to get a change
merged.

By contributing you agree that your contributions are licensed under the project's
[Apache 2.0 license](./LICENSE).

## Table of contents

- [Ways to contribute](#ways-to-contribute)
- [Development setup](#development-setup)
- [The checks your PR must pass](#the-checks-your-pr-must-pass)
- [Non-negotiable rules](#non-negotiable-rules)
- [Writing statistical code](#writing-statistical-code)
- [Project layout & workflow](#project-layout--workflow)
- [Commit & PR conventions](#commit--pr-conventions)
- [Reporting bugs & requesting features](#reporting-bugs--requesting-features)

## Ways to contribute

- **Fix a bug** - start from a failing test that reproduces it (see below).
- **Add a benchmark's live results** - run a `bench/` module against a real model and open a PR
  with the generated report under `bench/results/`.
- **Implement a planned extension** - framework harness adapters (LangGraph / OpenAI Agents /
  Pydantic AI), an HTML report, or a pytest plugin (see the roadmap in [`PROGRESS.md`](./PROGRESS.md)).
- **Improve docs** - the README, docstrings (which must state the *estimand*), or examples.

For anything larger than a small fix, please open an issue first so we can agree on the
approach before you invest time.

## Development setup

causeval uses [uv](https://docs.astral.sh/uv/) and targets Python 3.10-3.12.

```bash
git clone https://github.com/routsom/causeval.git
cd causeval
uv sync                 # core dev environment
uv sync --all-extras    # + optional backends (nli/irt/ppi/observational)
```

> The `nli` extra pulls in `torch`, which has no wheel for some platforms (e.g. macOS x86_64).
> If `--all-extras` fails there, `uv sync` (core + dev) is enough for almost all work.

## The checks your PR must pass

CI runs these on Python 3.10, 3.11, and 3.12. Run them locally before pushing:

```bash
uv run pytest -m "not live"                    # fast offline suite - MUST always pass
uv run ruff check . && uv run ruff format --check .
uv run mypy src/causeval                        # strict typing on src/
```

Optional / situational:

```bash
uv run pytest -m sim        # statistical simulation tests (slower; also run offline)
uv run pytest -m live       # real LLM calls - needs API keys; run only when relevant
```

A PR is ready when the offline suite, ruff, ruff-format, and mypy are all green.

## Non-negotiable rules

These are enforced by tests and reviews, not just documented. The authoritative list lives in
[`CLAUDE.md`](./CLAUDE.md); the essentials:

1. **Wrap DeepEval, never fork it.** Depend on `deepeval>=4.2,<5`; never patch its internals or
   monkeypatch its classes. If you must copy DeepEval code, keep its Apache 2.0 header and
   record it in [`NOTICE`](./NOTICE).
2. **Import DeepEval only through `causeval.adapters.deepeval_import`.** It sets the
   telemetry/dotenv opt-outs *before* the first `import deepeval`. A test enforces that no other
   module imports deepeval directly.
3. **A fresh metric object per measurement.** Build metrics via the adapter factory; never share
   an instance across concurrent calls (DeepEval issue #3356).
4. **Never report a bare score.** Every public result carries `n_items`, `n_repeats`, a
   confidence interval, the method used, and provenance - CLI output included.
5. **No telemetry, no network calls, no side effects at import time.**
6. **No hardcoded model prices or model names in logic.** They come from user config; names
   appear only in config defaults and examples.
7. **Offline tests must not call real LLMs.** Use the fakes in `tests/fakes/`. Live tests get
   `@pytest.mark.live`.
8. **Every statistical method needs a simulation test** proving its coverage or error rate on
   synthetic data with a known answer.
9. **Don't guess DeepEval's API.** Read the installed source under
   `.venv/lib/python*/site-packages/deepeval/` before writing adapter code.

## Writing statistical code

This is the heart of the project, so it has extra expectations:

- **Test-first.** For a statistical change, write the simulation or unit test before the
  implementation. A new estimator needs a test proving its coverage/error rate on data with a
  known ground truth (mark it `@pytest.mark.sim`).
- **State the estimand.** Every public function's docstring must say exactly what it estimates
  (e.g. "mean human score over the full dataset").
- **Randomness is explicit.** Every function that samples takes an `rng: np.random.Generator`.
  No global seeds.
- **Numerics are numpy/scipy.** No pandas in `src/` (fine in `bench/` and notebooks).
- **Cross-check where a reference exists.** PPI is cross-checked against `ppi_py`, IRT against
  `girth` - behind the optional extras, skipped when not installed.

## Project layout & workflow

Read [`SPEC.md`](./SPEC.md) (design + module contracts) and [`PROGRESS.md`](./PROGRESS.md)
(phase status + decision log) before non-trivial work.

```
core/           result schemas, LLM cache, cost planner, provenance   (foundation)
adapters/       the DeepEval wrapper: metric factory, import guard, judge adapter
stats/          repeated sampling, bootstrap CIs, paired tests, gates, IRT, adaptive
interventions/  RAG grounding, counterfactuals, perturbations, CoT faithfulness
judge_audit/    bias probes, calibration, PPI, conformal, jury
attribution/    agent trace, record/replay, step-level blame
checks/         deterministic checks + the metamorphic-relation registry
bench/          validation benchmarks with known ground truth
```

Dependency direction is one-way: `core <- adapters <- stats <- everything else`. In
particular, `stats` must never import from `interventions`, `judge_audit`, or `attribution`.

If a SPEC requirement looks wrong or infeasible, don't silently change the design - raise it in
an issue or note it under "Open questions" in `PROGRESS.md`.

## Commit & PR conventions

- **Conventional commits**: `feat(stats): ...`, `fix(adapters): ...`, `test(interventions): ...`,
  `docs: ...`. Keep the subject imperative and scoped to the area you touched.
- **Use a plain dash `-`, not an em dash**, in commit messages and docs.
- **Don't hand-edit** `CHANGELOG.md` or any file marked auto-generated.
- **One logical change per PR.** Update `PROGRESS.md` (what changed, decisions, open questions)
  when you complete a unit of work.
- Fill out the pull-request template; make sure the offline suite, ruff, and mypy are green and
  say so in the PR description.

## Reporting bugs & requesting features

Please use the issue templates:

- **Bug report** - include a minimal reproduction, the expected vs actual behaviour, and your
  Python / causeval / deepeval versions. For a metric or judge issue, include the result's
  `provenance` block if you can - it captures seed, versions, and dataset hash.
- **Feature request** - describe the evaluation problem you're trying to solve, not just the API
  you imagine. causeval's scope is uncertainty, causality, and judge validity on top of
  DeepEval; features outside that (see the non-goals in `SPEC.md`) are likely out of scope.

Thanks for helping make LLM evaluation rigorous. 🙏
