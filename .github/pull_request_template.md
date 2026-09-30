<!-- Thanks for contributing to causeval! Please fill this out so review is fast. -->

## Summary

<!-- What does this PR change, and why? Link any related issue: "Closes #123". -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Statistical method / estimator
- [ ] Benchmark (offline or live results)
- [ ] Docs
- [ ] Refactor / chore

## Checklist

- [ ] `uv run pytest -m "not live"` passes
- [ ] `uv run ruff check . && uv run ruff format --check .` passes
- [ ] `uv run mypy src/causeval` passes
- [ ] New/changed public functions have docstrings that state the **estimand**
- [ ] For statistical changes: a simulation test proves coverage/error rate on a known ground truth (`@pytest.mark.sim`)
- [ ] No new bare scores - every result carries n, repeats, a CI, method, and provenance
- [ ] DeepEval is imported only through `causeval.adapters.deepeval_import`; no other rule in `CLAUDE.md` is broken
- [ ] `PROGRESS.md` updated if this completes a unit of work

## Notes for reviewers

<!-- Anything non-obvious: a design decision, a tradeoff, a follow-up left for later. -->
