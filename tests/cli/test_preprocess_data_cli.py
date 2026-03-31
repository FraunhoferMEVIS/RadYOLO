from __future__ import annotations

import nibabel as nib
import numpy as np
from click.testing import CliRunner

from radyolo import main


def test_preprocess_data_cli(synthetic_dataset, tmp_path) -> None:
    output_dataset = tmp_path / "preprocessed_dataset"
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "preprocess-data",
            str(synthetic_dataset),
            str(output_dataset),
            "--workers",
            "0",
        ],
    )

    assert result.exit_code == 0
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


def test_preprocess_data_cli_preserves_image_dtype(tmp_path) -> None:
    dataset_path = tmp_path / "dataset_dtype"
    (dataset_path / "images").mkdir(parents=True)
    (dataset_path / "labels").mkdir(parents=True)

    image_array = np.arange(5 * 6 * 7, dtype=np.int16).reshape((5, 6, 7))
    image = nib.Nifti1Image(image_array, affine=np.eye(4))
    nib.save(image, dataset_path / "images" / "case_dtype.nii.gz")
    (dataset_path / "labels" / "case_dtype.txt").write_text("", encoding="utf-8")

    output_dataset = tmp_path / "preprocessed_dtype"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "preprocess-data",
            str(dataset_path),
            str(output_dataset),
            "--workers",
            "0",
            "--image-dtype",
            "int16",
        ],
    )

    assert result.exit_code == 0
    converted_image = np.load(output_dataset / "images" / "case_dtype.npy")
    assert converted_image.dtype == np.int16


def test_preprocess_data_cli_resamples_to_fixed_extent(synthetic_dataset, tmp_path) -> None:
    output_dataset = tmp_path / "preprocessed_resampled_extent"
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "preprocess-data",
            str(synthetic_dataset),
            str(output_dataset),
            "--workers",
            "0",
            "--image-extent",
            "20",
            "18",
            "16",
        ],
    )

    assert result.exit_code == 0
    converted_image = np.load(output_dataset / "images" / "case1.npy")
    assert converted_image.shape == (16, 18, 20)

    with np.load(output_dataset / "masks" / "case1.npz") as mask_archive:
        mask = mask_archive["mask"]
    assert mask.shape == (16, 18, 20)


def test_preprocess_data_cli_resamples_to_fixed_voxel_size_and_normalizes(tmp_path) -> None:
    dataset_path = tmp_path / "dataset_voxel_size"
    (dataset_path / "images").mkdir(parents=True)
    (dataset_path / "labels").mkdir(parents=True)

    image_array = np.arange(4 * 5 * 6, dtype=np.float32).reshape((4, 5, 6))
    affine = np.diag((2.0, 3.0, 4.0, 1.0))
    image = nib.Nifti1Image(image_array, affine=affine)
    nib.save(image, dataset_path / "images" / "case_voxel.nii.gz")
    (dataset_path / "labels" / "case_voxel.txt").write_text("", encoding="utf-8")

    output_dataset = tmp_path / "preprocessed_voxel_size"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "preprocess-data",
            str(dataset_path),
            str(output_dataset),
            "--workers",
            "0",
            "--voxel-size",
            "2.0",
            "1.5",
            "1.0",
            "--normalization",
            "{type: zscore}",
            "--image-dtype",
            "float16",
        ],
    )

    assert result.exit_code == 0
    converted_image = np.load(output_dataset / "images" / "case_voxel.npy")
    assert converted_image.dtype == np.float16
    assert converted_image.shape == (8, 10, 12)
    assert abs(float(converted_image.astype(np.float32).mean())) < 0.05
