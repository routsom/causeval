"""CLI: run persistence, compare/gate exit codes, and plan output (offline)."""

from __future__ import annotations

from pathlib import Path

from causeval import Experiment, MetricSpec
from causeval.cli import main, save_run
from tests.fakes.fake_judge import FakeScoringMetric


def _make_run(shift: float, *, n: int = 20, repeats: int = 3):
    spec = MetricSpec(builder=lambda: FakeScoringMetric(), label="fake")
    dataset = [
        {
            "item_id": f"i{i:02d}",
            "input": f"q{i}",
            "actual_output": f"{0.5 + shift + 0.005 * i:.4f}",
        }
        for i in range(n)
    ]
    return Experiment(dataset=dataset, metrics=[spec], repeats=repeats, seed=0).run()


def _write_pair(tmp_path: Path, base_shift: float, cand_shift: float) -> tuple[Path, Path]:
    base_path = tmp_path / "baseline.json"
    cand_path = tmp_path / "candidate.json"
    save_run(_make_run(base_shift), base_path)
    save_run(_make_run(cand_shift), cand_path)
    return base_path, cand_path


def test_gate_pass_exit_zero(tmp_path: Path, capsys) -> None:
    base, cand = _write_pair(tmp_path, 0.0, 0.10)  # candidate clearly better
    code = main(["gate", "--baseline", str(base), "--candidate", str(cand), "--margin", "0.02"])
    out = capsys.readouterr().out
    assert code == 0
    assert "overall: pass" in out


def test_gate_regression_exit_one(tmp_path: Path, capsys) -> None:
    base, cand = _write_pair(tmp_path, 0.0, -0.10)  # candidate clearly worse
    code = main(["gate", "--baseline", str(base), "--candidate", str(cand), "--margin", "0.02"])
    out = capsys.readouterr().out
    assert code == 1
    assert "overall: regression" in out


def test_compare_reports_without_exit_code(tmp_path: Path, capsys) -> None:
    base, cand = _write_pair(tmp_path, 0.0, 0.05)
    code = main(["compare", "--baseline", str(base), "--candidate", str(cand), "--margin", "0.02"])
    out = capsys.readouterr().out
    assert code == 0
    assert "fake" in out and "verdict" in out


def test_plan_prints_table(tmp_path: Path, capsys) -> None:
    base, cand = _write_pair(tmp_path, 0.0, 0.03)
    code = main(
        [
            "plan",
            "--pilot-baseline",
            str(base),
            "--pilot-candidate",
            str(cand),
            "--metric",
            "fake",
            "--detect",
            "0.03",
            "--power",
            "0.8",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Plan to detect" in out


def test_ground_from_config(tmp_path: Path, capsys) -> None:
    dataset = tmp_path / "rag.jsonl"
    dataset.write_text(
        '{"item_id": "a", "question": "year?", "contexts": ["Founded in 2019."], '
        '"expected_output": "2019", '
        '"cf": {"chunk_index": 0, "original_value": "2019", "cf_value": "1994", '
        '"edited_chunk": "Founded in 1994."}}\n'
    )
    config = tmp_path / "rag.yaml"
    config.write_text(
        "name: ground_smoke\n"
        f"dataset: {dataset}\n"
        "repeats: 3\n"
        "app: tests.fakes.cli_helpers:build_grounded_app\n"
    )
    out_dir = tmp_path / "runs"
    code = main(["ground", "--config", str(config), "--out", str(out_dir)])
    out = capsys.readouterr().out
    assert code == 0
    assert "RAG grounding" in out
    assert len(list(out_dir.glob("*_ground_smoke.json"))) == 1


def test_audit_judge_from_config(tmp_path: Path, capsys) -> None:
    import json

    # labeled set: judge tracks the human label with a small positive bias.
    labeled = tmp_path / "labeled.jsonl"
    labeled.write_text(
        "\n".join(
            json.dumps(
                {"item_id": f"i{i}", "judge_score": min(1.0, i / 30 + 0.1), "human_label": i / 30}
            )
            for i in range(30)
        )
        + "\n"
    )
    unlabeled = tmp_path / "unlabeled.jsonl"
    unlabeled.write_text(
        "\n".join(json.dumps({"item_id": f"u{i}", "judge_score": i / 40}) for i in range(40)) + "\n"
    )
    config = tmp_path / "judge.yaml"
    config.write_text(
        "name: judge_smoke\n"
        f"labeled: {labeled}\n"
        f"unlabeled: {unlabeled}\n"
        "threshold: 0.5\n"
        "i_know_the_labels_are_random: true\n"
    )
    out_dir = tmp_path / "runs"
    code = main(["audit-judge", "--config", str(config), "--out", str(out_dir)])
    out = capsys.readouterr().out
    assert code == 0
    assert "calibration" in out and "PPI human-mean" in out
    written = list(out_dir.glob("*_judge_smoke.json"))
    assert len(written) == 1
    payload = json.loads(written[0].read_text())
    assert "calibration" in payload and "ppi" in payload


def test_attribute_from_config(tmp_path: Path, capsys) -> None:
    import json

    config = tmp_path / "agent.yaml"
    config.write_text("name: attr_smoke\ncase: tests.fakes.cli_helpers:build_attribution_case\n")
    out_dir = tmp_path / "runs"
    code = main(["attribute", "--config", str(config), "--out", str(out_dir)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Attribution for task" in out and "DECISIVE" in out
    written = list(out_dir.glob("*_attr_smoke.json"))
    assert len(written) == 1
    payload = json.loads(written[0].read_text())
    assert payload["decisive_step"] is not None


def test_version(capsys) -> None:
    code = main(["version"])
    assert code == 0
    assert "causeval" in capsys.readouterr().out


def test_run_from_config(tmp_path: Path, capsys) -> None:
    # A config whose app + metric are importable module paths; keeps `run` offline.
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"item_id": "a", "input": "q1"}\n{"item_id": "b", "input": "q2"}\n')
    config = tmp_path / "eval.yaml"
    config.write_text(
        "name: smoke\n"
        f"dataset: {dataset}\n"
        "repeats: 2\n"
        "seed: 0\n"
        "app: tests.fakes.cli_helpers:constant_app\n"
        "metrics:\n"
        "  - builder: tests.fakes.cli_helpers:build_fake_metric\n"
        "    label: fake\n"
    )
    out_dir = tmp_path / "runs"
    code = main(["run", "--config", str(config), "--out", str(out_dir)])
    assert code == 0
    written = list(out_dir.glob("*_smoke.json"))
    assert len(written) == 1
    assert "wrote" in capsys.readouterr().out
