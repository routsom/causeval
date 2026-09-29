# causeval

A statistically rigorous, causal evaluation layer for LLM apps, built on top of
[DeepEval](https://github.com/confident-ai/deepeval).

DeepEval **measures**. causeval adds three things it lacks:

- **Uncertainty** — is the score real? Repeated sampling, variance decomposition,
  clustered-bootstrap confidence intervals, paired comparisons, and CI-based gates.
- **Causality** — what caused this output or failure? RAG context ablation and
  counterfactual context, input perturbations, agent step-level counterfactual replay.
- **Judge validity** — can we trust the judge? Bias audits, human calibration,
  prediction-powered inference (PPI), and conformal abstention.

> `causeval` is a working name. See `SPEC.md` for the full design and `PROGRESS.md`
> for phase status.

## Status

Early development. Work proceeds in phases (see `SPEC.md` §5). Current phase is tracked
in `PROGRESS.md`.

## Non-negotiable rules

causeval **wraps** DeepEval, never forks it, and never reports a bare score: every public
result carries an item count, repeat count, confidence interval, method, and provenance.
See `CLAUDE.md` for the full list.

## Development

```bash
uv sync --all-extras                 # install
uv run pytest -m "not live"          # fast offline tests (default)
uv run pytest -m live                # real LLM calls; needs API keys
uv run ruff check . && uv run ruff format --check .
uv run mypy src/causeval             # strict on src/
```

## License

Apache 2.0 (matching DeepEval). See `LICENSE` and `NOTICE`.
