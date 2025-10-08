from __future__ import annotations

from typing import TYPE_CHECKING

from tests.conftest import (
    analyze_dataset,
    create_data_splits,
    export_onnx,
    run_validation,
)
from yolo11_3d.data.convert_to_nndetection import convert_dataset_to_nndetection

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path


def test_create_data_splits(synthetic_dataset: Path) -> None:
    splits_file = synthetic_dataset / "data_splits.json"
    if splits_file.exists():
        splits_file.unlink()
    out = create_data_splits(synthetic_dataset, k=5, seed=0)
    assert out == splits_file
    assert splits_file.exists(), "data_splits.json not created"


def test_analyze_dataset(synthetic_dataset: Path) -> None:
    out = analyze_dataset(synthetic_dataset, workers=0)
    assert out.exists()
    fingerprint = synthetic_dataset / "dataset_fingerprint.json"
    assert fingerprint.exists(), "dataset_fingerprint.json not created"


def test_train_minimal(trained_run: Path) -> None:
    assert (trained_run / "weights" / "best.pt").exists(), "best.pt missing"
    assert (trained_run / "results.csv").exists(), "results.csv missing"


def test_onnx_export(trained_run: Path) -> None:
    model_path = trained_run / "weights" / "last.pt"
    export_path = export_onnx(model_path)
    onnx_path = model_path.with_suffix(".onnx")
    assert onnx_path.exists(), "ONNX export not created"
    assert export_path == onnx_path, "Returned export path mismatch"


def test_test_command(trained_run: Path) -> None:
    model_path = trained_run / "weights" / "best.pt"
    val_dir = run_validation(model_path, group="val")
    assert val_dir.exists(), "Validation output directory missing"
    predictions = val_dir / "predictions.json"
    assert predictions.exists(), "predictions.json missing"


def test_convert_to_nndetection(synthetic_dataset: Path, tmp_path: Path) -> None:
    out_root = tmp_path / "nndet"
    out_root.mkdir()
    task_dir = convert_dataset_to_nndetection(
        dataset_dir=synthetic_dataset,
        out_root=out_root,
        task_id=123,
        task_name="Tiny",
        fold=0,
        include_val_in_train=True,
        modality_name="CT",
        workers=0,
    )
    expected_dir = out_root / "Task123_Tiny"
    assert task_dir.exists(), "Task folder not created"
    dataset_yaml = expected_dir / "dataset.yaml"
    assert dataset_yaml.exists(), "nnDetection dataset.yaml missing"
