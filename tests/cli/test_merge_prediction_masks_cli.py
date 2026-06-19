from __future__ import annotations

from typing import TYPE_CHECKING

from click.testing import CliRunner

from radyolo import main

if TYPE_CHECKING:
    from pathlib import Path


def test_merge_prediction_masks_cli_uses_default_confidence_threshold(
    monkeypatch,
    tmp_path: Path,
) -> None:
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    captured: dict[str, object] = {}

    def fake_merge_prediction_masks(*, predictions_dir: Path, confidence_threshold: float) -> Path:
        captured["predictions_dir"] = predictions_dir
        captured["confidence_threshold"] = confidence_threshold
        output_dir = predictions_dir / "masks_merged"
        output_dir.mkdir(exist_ok=True)
        return output_dir

    monkeypatch.setattr("radyolo.merge_prediction_masks_fn", fake_merge_prediction_masks)

    runner = CliRunner()
    result = runner.invoke(main, ["merge-prediction-masks", str(predictions_dir)])

    assert result.exit_code == 0
    assert captured["predictions_dir"] == predictions_dir
    assert captured["confidence_threshold"] == 0.1
    assert f"Wrote merged masks to: {predictions_dir / 'masks_merged'}" in result.output


def test_merge_prediction_masks_cli_passes_custom_confidence_threshold(
    monkeypatch,
    tmp_path: Path,
) -> None:
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    captured: dict[str, object] = {}

    def fake_merge_prediction_masks(*, predictions_dir: Path, confidence_threshold: float) -> Path:
        captured["predictions_dir"] = predictions_dir
        captured["confidence_threshold"] = confidence_threshold
        output_dir = predictions_dir / "masks_merged"
        output_dir.mkdir(exist_ok=True)
        return output_dir

    monkeypatch.setattr("radyolo.merge_prediction_masks_fn", fake_merge_prediction_masks)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "merge-prediction-masks",
            str(predictions_dir),
            "--confidence-threshold",
            "0.35",
        ],
    )

    assert result.exit_code == 0
    assert captured["predictions_dir"] == predictions_dir
    assert captured["confidence_threshold"] == 0.35
