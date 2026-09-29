"""B3 acceptance: probes recover injected position (0.15) + verbosity (0.08) biases.

Accept (SPEC 6): both recovered within their CIs at ~95% coverage across simulations, and
non-injected biases (formatting, authorship) report ~0 and are not flagged. Offline uses a
lenient binomial bound with 150 sims; the strict >=1000-sim / [0.93, 0.97] version is nightly.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from causeval.bench import judge_bias as b3


def test_b3_single_run_recovers_and_flags(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(b3, "RESULTS_DIR", tmp_path)
    payload, json_path, md_path = b3.run_offline(n_pairs=80, n_items=80, seed=0)
    probes = payload["probes"]
    assert probes["position"]["in_ci"] and probes["position"]["flagged"]
    assert probes["verbosity"]["in_ci"] and probes["verbosity"]["flagged"]
    assert not probes["formatting"]["flagged"]
    assert not probes["authorship"]["flagged"]
    assert json_path.exists() and md_path.exists()


@pytest.mark.sim
def test_b3_coverage_of_injected_biases() -> None:
    n_sims = 150
    hits = {"position": 0, "verbosity": 0}
    ests = {"position": [], "verbosity": []}
    truth = {"position": b3.POSITION_BIAS, "verbosity": b3.VERBOSITY_BIAS}
    for s in range(n_sims):
        probes = asyncio.run(b3.run_probes(n_pairs=60, n_items=60, seed=5000 + s))
        for name in hits:
            e = probes[name]
            ests[name].append(e.estimate)
            if e.ci_low <= truth[name] <= e.ci_high:
                hits[name] += 1
    for name in hits:
        coverage = hits[name] / n_sims
        assert coverage >= 0.88, f"{name} coverage {coverage:.3f} too low"
        assert np.mean(ests[name]) == pytest.approx(truth[name], abs=0.03)


@pytest.mark.sim
def test_b3_noninjected_biases_report_near_zero() -> None:
    n_sims = 60
    means = {"formatting": [], "authorship": []}
    for s in range(n_sims):
        probes = asyncio.run(b3.run_probes(n_pairs=50, n_items=50, seed=7000 + s))
        for name in means:
            means[name].append(probes[name].estimate)
    for name, vals in means.items():
        assert np.mean(vals) == pytest.approx(0.0, abs=0.01), f"{name} not ~0"
