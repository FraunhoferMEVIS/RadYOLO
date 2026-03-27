from __future__ import annotations

from typing import TYPE_CHECKING

from click.testing import CliRunner

from radyolo import main

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
    assert captured["config"].conf == 0.1
    assert captured["config"].iou == 0.2
    assert captured["config"].device == "0"


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
    assert captured["config"].conf == 0.1
    assert captured["config"].iou == 0.2
    assert captured["config"].device == "0"
    assert captured["config"].clip_boxes_to_image is False
