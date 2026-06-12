from __future__ import annotations

import json
from pathlib import Path

from radyolo.inference import predict as predict_module
from radyolo.inference.predict import PredictionConfig, run_prediction


def _touch_nii_gz(path: Path) -> None:
    path.write_bytes(b"")


def test_run_prediction_filters_files_by_split_selection(monkeypatch, tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"checkpoint")
    split_file = tmp_path / "data_splits.json"

    _touch_nii_gz(input_dir / "case_a.nii.gz")
    _touch_nii_gz(input_dir / "case_b.nii.gz")
    _touch_nii_gz(input_dir / "case_c.nii.gz")

    split_file.write_text(
        json.dumps([{"train": ["case_a"], "val": ["case_b"], "test": ["case_c"]}]),
        encoding="utf-8",
    )

    captured: dict[str, list[Path]] = {}

    def fake_run_prediction_pipeline(prediction_cfg: PredictionConfig, files: list[Path]) -> int:
        del prediction_cfg
        captured["files"] = files
        return 0

    monkeypatch.setattr(predict_module, "_run_prediction_pipeline", fake_run_prediction_pipeline)

    result = run_prediction(
        PredictionConfig(
            model_path=model_path,
            input_dir=input_dir,
            output_dir=output_dir,
            confidence_threshold=0.1,
            nms_iou_threshold=0.2,
            device="cpu",
            clip_boxes_to_image=False,
            split_file=split_file,
            split_fold=0,
            split_group="test",
        )
    )

    assert result == 0
    assert [p.name for p in captured["files"]] == ["case_c.nii.gz"]


def test_run_prediction_returns_error_for_missing_split_group(monkeypatch, tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"checkpoint")
    split_file = tmp_path / "data_splits.json"

    _touch_nii_gz(input_dir / "case_a.nii.gz")
    split_file.write_text(json.dumps([{"train": ["case_a"]}]), encoding="utf-8")

    called_pipeline = {"value": False}

    def fake_run_prediction_pipeline(prediction_cfg: PredictionConfig, files: list[Path]) -> int:
        del prediction_cfg, files
        called_pipeline["value"] = True
        return 0

    monkeypatch.setattr(predict_module, "_run_prediction_pipeline", fake_run_prediction_pipeline)

    result = run_prediction(
        PredictionConfig(
            model_path=model_path,
            input_dir=input_dir,
            output_dir=output_dir,
            confidence_threshold=0.1,
            nms_iou_threshold=0.2,
            device="cpu",
            clip_boxes_to_image=False,
            split_file=split_file,
            split_fold=0,
            split_group="val",
        )
    )

    assert result == 1
    assert called_pipeline["value"] is False
