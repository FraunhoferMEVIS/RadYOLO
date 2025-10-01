from __future__ import annotations

from pathlib import Path

import torch
from conftest import run_cli


def test_create_data_splits(
    cli_executable: list[str],
    synthetic_dataset: Path,
) -> None:
    splits_file = synthetic_dataset / "data_splits.json"
    if splits_file.exists():
        splits_file.unlink()
    proc = run_cli([
        *cli_executable,
        "create-data-splits",
        synthetic_dataset.as_posix(),
        "--k",
        "5",
        "--seed",
        "0",
    ])
    assert proc.returncode == 0, proc.stderr
    assert splits_file.exists(), "data_splits.json not created"


def test_analyze_dataset(cli_executable: list[str], synthetic_dataset: Path) -> None:
    proc = run_cli([
        *cli_executable,
        "analyze-dataset",
        synthetic_dataset.as_posix(),
        "--workers",
        "0",
    ])
    assert proc.returncode == 0, proc.stderr
    fingerprint = synthetic_dataset / "dataset_fingerprint.json"
    assert fingerprint.exists(), "dataset_fingerprint.json not created"


def test_train_minimal(trained_run: Path) -> None:
    assert (trained_run / "weights" / "best.pt").exists(), "best.pt missing"
    assert (trained_run / "results.csv").exists(), "results.csv missing"


def test_onnx_export(cli_executable: list[str], trained_run: Path) -> None:
    model_path = trained_run / "weights" / "last.pt"
    proc = run_cli([*cli_executable, "onnx-export", model_path.as_posix()])
    assert proc.returncode == 0, proc.stderr
    onnx_path = model_path.with_suffix(".onnx")
    assert onnx_path.exists(), "ONNX export not created"


def test_test_command(cli_executable: list[str], trained_run: Path) -> None:
    # Use validation group produced by fixture dataset
    model_path = trained_run / "weights" / "best.pt"
    dataset_root = None
    # Recover dataset path from train config stored in checkpoint
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    config = ckpt["train_config"]
    dataset_root = Path(config.data)  # raw dict inside checkpoint
    proc = run_cli([
        *cli_executable,
        "test",
        model_path.as_posix(),
        dataset_root.as_posix(),
        "--group",
        "val",
    ])
    assert proc.returncode == 0, proc.stderr
    # Validation artifacts
    val_dir = trained_run / "validation" / "val"
    assert val_dir.exists(), "Validation output directory missing"
    predictions = val_dir / "predictions.json"
    assert predictions.exists(), "predictions.json missing"


def test_convert_to_nndetection(
    cli_executable: list[str], synthetic_dataset: Path, tmp_path: Path
) -> None:
    out_root = tmp_path / "nndet"
    out_root.mkdir()
    proc = run_cli([
        *cli_executable,
        "convert-to-nndetection",
        synthetic_dataset.as_posix(),
        "--out-root",
        out_root.as_posix(),
        "--task-id",
        "123",
        "--task-name",
        "Tiny",
        "--fold",
        "0",
    ])
    assert proc.returncode == 0, proc.stderr
    # Expect a folder like Task123_Tiny
    task_dir = out_root / "Task123_Tiny"
    assert task_dir.exists(), "Task folder not created"
    # Expect dataset.yaml (nnDetection format)
    dataset_yaml = task_dir / "dataset.yaml"
    assert dataset_yaml.exists(), "nnDetection dataset.yaml missing"
