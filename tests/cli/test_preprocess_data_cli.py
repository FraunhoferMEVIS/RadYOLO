from __future__ import annotations

import blosc2
import nibabel as nib
import numpy as np
import shutil
from click.testing import CliRunner

from radyolo import main
from radyolo.data.preprocess_data import _compute_storage_shapes


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
    output_images = sorted((output_dataset / "images").glob("*.b2nd"))

    assert len(output_images) == len(input_images), "number of preprocessed images is incorrect"
    input_masks = sorted((synthetic_dataset / "masks").glob("*.nii.gz"))
    output_masks = sorted((output_dataset / "masks").glob("*.b2nd"))

    assert len(output_masks) == len(input_masks), "number of preprocessed masks is incorrect"
    loaded_mask = np.asarray(blosc2.open(str(output_masks[0]), mode="r")[:])
    assert loaded_mask.dtype == np.uint8
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
    converted_image = np.asarray(
        blosc2.open(str(output_dataset / "images" / "case_dtype.b2nd"), mode="r")[:]
    )
    assert converted_image.dtype == np.int16


def test_preprocess_data_cli_uses_full_image_chunk_for_fixed_extent_blosc2(
    synthetic_dataset,
    tmp_path,
) -> None:
    output_dataset = tmp_path / "preprocessed_blosc2_extent"
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
    output_store = output_dataset / "images" / "case1.b2nd"
    output_mask_store = output_dataset / "masks" / "case1.b2nd"
    assert output_store.exists()
    assert output_mask_store.exists()
    blosc_array = blosc2.open(str(output_store), mode="r")
    blosc_mask_array = blosc2.open(str(output_mask_store), mode="r")
    assert blosc_array.chunks == (20, 18, 16)
    assert blosc_mask_array.chunks == (20, 18, 16)
    loaded = np.asarray(blosc_array[:])
    loaded_mask = np.asarray(blosc_mask_array[:])
    assert loaded.shape == (20, 18, 16)
    assert loaded_mask.shape == (20, 18, 16)


def test_preprocess_data_cli_copies_custom_split_file_name(synthetic_dataset, tmp_path) -> None:
    dataset_path = tmp_path / "dataset_with_custom_split"
    shutil.copytree(synthetic_dataset, dataset_path)
    custom_split_file = dataset_path / "folds_alt.json"
    (dataset_path / "data_splits.json").replace(custom_split_file)

    output_dataset = tmp_path / "preprocessed_dataset_custom_split"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "preprocess-data",
            str(dataset_path),
            str(output_dataset),
            "--workers",
            "0",
            "--split-file-name",
            "folds_alt.json",
        ],
    )

    assert result.exit_code == 0
    assert (output_dataset / "folds_alt.json").exists(), "custom split file not copied"


def test_preprocess_data_cli_uses_smart_blosc2_chunks_for_voxel_size(tmp_path) -> None:
    dataset_path = tmp_path / "dataset_voxel_blosc2"
    (dataset_path / "images").mkdir(parents=True)
    (dataset_path / "labels").mkdir(parents=True)
    (dataset_path / "masks").mkdir(parents=True)

    image_array = np.arange(4 * 5 * 6, dtype=np.float32).reshape((4, 5, 6))
    affine = np.diag((2.0, 3.0, 4.0, 1.0))
    image = nib.Nifti1Image(image_array, affine=affine)
    mask = nib.Nifti1Image((image_array > image_array.mean()).astype(np.uint8), affine=affine)
    nib.save(image, dataset_path / "images" / "case_voxel.nii.gz")
    nib.save(mask, dataset_path / "masks" / "case_voxel.nii.gz")
    (dataset_path / "labels" / "case_voxel.txt").write_text("", encoding="utf-8")

    output_dataset = tmp_path / "preprocessed_voxel_blosc2"
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
        ],
    )

    assert result.exit_code == 0
    output_store = output_dataset / "images" / "case_voxel.b2nd"
    output_mask_store = output_dataset / "masks" / "case_voxel.b2nd"
    blosc_array = blosc2.open(str(output_store), mode="r")
    blosc_mask_array = blosc2.open(str(output_mask_store), mode="r")
    expected_block, expected_chunk = _compute_storage_shapes((4, 10, 24), np.dtype(np.float32))
    expected_mask_block, expected_mask_chunk = _compute_storage_shapes(
        (4, 10, 24),
        np.dtype(np.uint8),
    )
    assert blosc_array.blocks == expected_block
    assert blosc_array.chunks == expected_chunk
    assert blosc_mask_array.blocks == expected_mask_block
    assert blosc_mask_array.chunks == expected_mask_chunk


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
    converted_image = np.asarray(
        blosc2.open(str(output_dataset / "images" / "case1.b2nd"), mode="r")[:]
    )
    assert converted_image.shape == (20, 18, 16)

    mask = np.asarray(blosc2.open(str(output_dataset / "masks" / "case1.b2nd"), mode="r")[:])
    assert mask.shape == (20, 18, 16)


def test_preprocess_data_cli_saves_masks_as_blosc2(
    synthetic_dataset,
    tmp_path,
) -> None:
    output_dataset = tmp_path / "preprocessed_blosc2_masks"
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
    assert not (output_dataset / "masks" / "case1.npz").exists()
    mask_store = output_dataset / "masks" / "case1.b2nd"
    assert mask_store.exists()
    loaded_mask = np.asarray(blosc2.open(str(mask_store), mode="r")[:])
    assert loaded_mask.dtype == np.uint8


def test_preprocess_data_cli_resamples_to_fixed_voxel_size_and_normalizes(tmp_path) -> None:
    dataset_path = tmp_path / "dataset_voxel_size"
    (dataset_path / "images").mkdir(parents=True)
    (dataset_path / "labels").mkdir(parents=True)
    (dataset_path / "masks").mkdir(parents=True)

    image_array = np.arange(4 * 5 * 6, dtype=np.float32).reshape((4, 5, 6))
    affine = np.diag((2.0, 3.0, 4.0, 1.0))
    image = nib.Nifti1Image(image_array, affine=affine)
    mask = nib.Nifti1Image((image_array > image_array.mean()).astype(np.uint8), affine=affine)
    nib.save(image, dataset_path / "images" / "case_voxel.nii.gz")
    nib.save(mask, dataset_path / "masks" / "case_voxel.nii.gz")
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
    converted_image = np.asarray(
        blosc2.open(str(output_dataset / "images" / "case_voxel.b2nd"), mode="r")[:]
    )
    assert converted_image.dtype == np.float16
    assert converted_image.shape == (4, 10, 24)
    assert abs(float(converted_image.astype(np.float32).mean())) < 0.05
