"""Validation benchmarks with known ground truth (SPEC §6).

Each benchmark has an offline version (fakes, runs in CI) and a live version (real models,
run manually, results committed under ``bench/results/``). pandas is allowed here.

Phase 0 ships ``baseline_variance``: it quantifies DeepEval's single-sample flakiness and
becomes the README's motivation.
"""
