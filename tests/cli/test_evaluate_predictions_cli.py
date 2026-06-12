from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from radyolo import main


def test_evaluate_predictions_cli_uses_only_common_cases(tmp_path) -> None:
    predictions_dir = tmp_path / "predictions"
    labels_dir = tmp_path / "labels"
    output_dir = tmp_path / "results"
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
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert result.output.strip() == f"Wrote evaluation files to: {output_dir!s}"

    metrics_path = output_dir / "metrics.json"
    metadata_path = output_dir / "metadata.json"
    curves_path = output_dir / "curves.json"
    assert metrics_path.exists()
    assert metadata_path.exists()
    assert curves_path.exists()

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    curves_payload = json.loads(curves_path.read_text(encoding="utf-8"))

    metrics_summary = metrics_payload["summary"]
    assert metrics_summary["metrics/precision"] == pytest.approx(1.0)
    assert metrics_summary["metrics/recall"] == pytest.approx(1.0)
    assert metrics_summary["metrics/mAP10"] == pytest.approx(0.995, abs=1e-6)
    assert metrics_summary["metrics/mAP10_agnostic"] == pytest.approx(0.995, abs=1e-6)
    assert metrics_summary["metrics/mAP10_95"] == pytest.approx(0.995, abs=1e-6)
    assert metrics_summary["metrics/mAP10_95_agnostic"] == pytest.approx(0.995, abs=1e-6)
    assert metrics_summary["metrics/FROC"] == pytest.approx(1.0, abs=1e-6)

    assert metadata_payload["case_counts"]["common"] == 1
    assert metadata_payload["case_counts"]["missing_in_predictions"] == 1
    assert metadata_payload["case_counts"]["missing_in_labels"] == 1
    assert metadata_payload["missing_cases"]["missing_in_predictions"] == ["case_label_only"]
    assert metadata_payload["missing_cases"]["missing_in_labels"] == ["case_prediction_only"]
    assert metadata_payload["objects_counts"]["predictions"] == 1
    assert metadata_payload["objects_counts"]["references"] == 1

    assert "froc" in curves_payload
    assert "precision_recall" in curves_payload
    assert len(curves_payload["froc"]["avg_fp_per_image"]) > 0
    assert len(curves_payload["froc"]["sensitivity"]) > 0
    assert len(curves_payload["precision_recall"]["recall_axis"]) == 1000
    assert len(curves_payload["precision_recall"]["mean_precision"]) == 1000


def test_evaluate_predictions_cli_fails_without_common_cases(tmp_path) -> None:
    predictions_dir = tmp_path / "predictions"
    labels_dir = tmp_path / "labels"
    output_dir = tmp_path / "results"
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
            str(output_dir),
        ],
    )

    assert result.exit_code == 1
    assert "No common case names found" in result.output
    assert not output_dir.exists()