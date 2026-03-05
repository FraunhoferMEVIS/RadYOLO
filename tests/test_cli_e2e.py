from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np

from radyolo.data.convert_to_nndetection import convert_dataset_to_nndetection
from tests.conftest import (
    analyze_dataset,
    create_data_splits,
    export_onnx,
    preprocess_data,
    run_validation,
)

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


def test_preprocess_data(synthetic_dataset: Path, tmp_path: Path) -> None:
    output_dataset = tmp_path / "preprocessed_dataset"
    out = preprocess_data(synthetic_dataset, output_dataset, workers=0)

    assert out == output_dataset
    input_images = sorted((synthetic_dataset / "images").glob("*.nii.gz"))
    output_images = sorted((output_dataset / "images").glob("*.npy"))

    assert len(output_images) == len(input_images), "number of preprocessed images is incorrect"
    input_masks = sorted((synthetic_dataset / "masks").glob("*.nii.gz"))
    output_masks = sorted((output_dataset / "masks").glob("*.npz"))

    assert len(output_masks) == len(input_masks), "number of preprocessed masks is incorrect"
    with np.load(output_masks[0]) as mask_archive:
        assert "mask" in mask_archive.files, "compressed mask archive missing 'mask' array"
    assert (output_dataset / "dataset.yaml").exists(), "dataset.yaml not copied"
    assert (output_dataset / "data_splits.json").exists(), "data_splits.json not copied"
    assert (output_dataset / "labels" / "case1.txt").exists(), "labels not copied"


def test_preprocess_data_preserves_image_dtype(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset_dtype"
    (dataset_path / "images").mkdir(parents=True)
    (dataset_path / "labels").mkdir(parents=True)

    image_array = np.arange(5 * 6 * 7, dtype=np.int16).reshape((5, 6, 7))
    image = nib.Nifti1Image(image_array, affine=np.eye(4))
    nib.save(image, dataset_path / "images" / "case_dtype.nii.gz")
    (dataset_path / "labels" / "case_dtype.txt").write_text("", encoding="utf-8")

    output_dataset = tmp_path / "preprocessed_dtype"
    preprocess_data(dataset_path, output_dataset, workers=0)

    converted_image = np.load(output_dataset / "images" / "case_dtype.npy")
    assert converted_image.dtype == np.int16


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
