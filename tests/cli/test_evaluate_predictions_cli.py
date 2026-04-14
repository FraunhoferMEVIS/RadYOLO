from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from radyolo import main


def test_evaluate_predictions_cli_uses_only_common_cases(tmp_path) -> None:
    predictions_dir = tmp_path / "predictions"
    labels_dir = tmp_path / "labels"
    output_path = tmp_path / "results.json"
    predictions_dir.mkdir()
    labels_dir.mkdir()

    matching_label = "0 0.5 0.5 0.5 0.5 0.5 0.5\n"
    matching_prediction = [
        {
            "class": 0,
            "confidence": 0.95,
            "box_relative_xyzxyz": [0.25, 0.25, 0.25, 0.75, 0.75, 0.75],
        }
    ]

    (labels_dir / "case_common.txt").write_text(matching_label, encoding="utf-8")
    (predictions_dir / "case_common.json").write_text(
        json.dumps(matching_prediction),
        encoding="utf-8",
    )

    (labels_dir / "case_label_only.txt").write_text(matching_label, encoding="utf-8")
    (predictions_dir / "case_prediction_only.json").write_text(
        json.dumps(matching_prediction),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "evaluate-predictions",
            str(predictions_dir),
            str(labels_dir),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert result.output.strip() == f"Wrote results to: {output_path!s}"
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["case_counts"]["common"] == 1
    assert summary["case_counts"]["missing_in_predictions"] == 1
    assert summary["case_counts"]["missing_in_labels"] == 1
    assert summary["missing_cases"]["missing_in_predictions"] == ["case_label_only"]
    assert summary["missing_cases"]["missing_in_labels"] == ["case_prediction_only"]
    assert summary["metrics"]["metrics/precision"] == pytest.approx(1.0)
    assert summary["metrics"]["metrics/recall"] == pytest.approx(1.0)
    assert summary["metrics"]["metrics/mAP10"] == pytest.approx(0.995, abs=1e-6)
    assert summary["metrics"]["metrics/mAP10Agn"] == pytest.approx(0.995, abs=1e-6)
    assert summary["metrics"]["metrics/mAP10-95"] == pytest.approx(0.995, abs=1e-6)
    assert summary["metrics"]["metrics/mAP10-95Agn"] == pytest.approx(0.995, abs=1e-6)


def test_evaluate_predictions_cli_fails_without_common_cases(tmp_path) -> None:
    predictions_dir = tmp_path / "predictions"
    labels_dir = tmp_path / "labels"
    output_path = tmp_path / "results.json"
    predictions_dir.mkdir()
    labels_dir.mkdir()

    (labels_dir / "case_label_only.txt").write_text(
        "0 0.5 0.5 0.5 0.5 0.5 0.5\n",
        encoding="utf-8",
    )
    (predictions_dir / "case_prediction_only.json").write_text("[]", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "evaluate-predictions",
            str(predictions_dir),
            str(labels_dir),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 1
    assert "No common case names found" in result.output
    assert not output_path.exists()