from __future__ import annotations

import json
from typing import TYPE_CHECKING

from click.testing import CliRunner

from radyolo import main
from radyolo.config.spatial import PatchSize

if TYPE_CHECKING:
    from pathlib import Path

    from radyolo.inference.predict import PredictionConfig


def test_predict_cli_passes_clip_override(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"checkpoint")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    captured: dict[str, PredictionConfig] = {}

    def fake_run_prediction(config: PredictionConfig) -> int:
        captured["config"] = config
        return 0

    monkeypatch.setattr("radyolo.run_prediction", fake_run_prediction)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "predict",
            str(model_path),
            str(input_dir),
            str(output_dir),
            "--no-clip-boxes-to-image",
        ],
    )

    assert result.exit_code == 0
    assert captured["config"].clip_boxes_to_image is False
    assert captured["config"].confidence_threshold == 0.1
    assert captured["config"].nms_iou_threshold == 0.2
    assert captured["config"].device == "0"
    assert captured["config"].overwrite_patch_size is None
    assert captured["config"].overwrite_normalization is None


def test_predict_cli_uses_current_defaults(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"checkpoint")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    captured: dict[str, PredictionConfig] = {}

    def fake_run_prediction(config: PredictionConfig) -> int:
        captured["config"] = config
        return 0

    monkeypatch.setattr("radyolo.run_prediction", fake_run_prediction)

    runner = CliRunner()
    result = runner.invoke(main, ["predict", str(model_path), str(input_dir), str(output_dir)])

    assert result.exit_code == 0
    assert captured["config"].confidence_threshold == 0.1
    assert captured["config"].nms_iou_threshold == 0.2
    assert captured["config"].device == "0"
    assert captured["config"].clip_boxes_to_image is False
    assert captured["config"].overwrite_patch_size is None
    assert captured["config"].overwrite_normalization is None


def test_predict_cli_passes_patch_size_and_normalization_overrides(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"checkpoint")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    captured: dict[str, PredictionConfig] = {}

    def fake_run_prediction(config: PredictionConfig) -> int:
        captured["config"] = config
        return 0

    monkeypatch.setattr("radyolo.run_prediction", fake_run_prediction)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "predict",
            str(model_path),
            str(input_dir),
            str(output_dir),
            "--overwrite-patch-size",
            "24",
            "32",
            "40",
            "--overwrite-normalization",
            "{type: zscore}",
        ],
    )

    assert result.exit_code == 0
    assert captured["config"].overwrite_patch_size == PatchSize(x=24, y=32, z=40)
    assert captured["config"].overwrite_normalization == {"type": "zscore"}


def test_predict_cli_uses_shared_preprocessing(
    trained_run,
    synthetic_dataset,
    tmp_path,
) -> None:
    model_path = trained_run / "weights" / "best.pt"
    output_dir = tmp_path / "prediction_output"
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "predict",
            str(model_path),
            str(synthetic_dataset / "images"),
            str(output_dir),
            "--device",
            "cpu",
            "--clip-boxes-to-image",
        ],
    )

    assert result.exit_code == 0
    prediction_files = sorted(output_dir.glob("*.json"))
    assert prediction_files, "prediction json files missing"
    with prediction_files[0].open(encoding="utf-8") as file:
        parsed = json.load(file)
    assert isinstance(parsed, list)
