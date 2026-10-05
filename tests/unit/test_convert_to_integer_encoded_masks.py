from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import pytest

from radyolo.data.convert_to_integer_encoded_masks import convert_dataset_to_integer_encoded_masks

if TYPE_CHECKING:
    from pathlib import Path


def test_convert_dataset_to_integer_encoded_masks_writes_classes_in_size_order(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    for subdir in ("images", "labels", "masks"):
        (dataset_dir / subdir).mkdir(parents=True, exist_ok=True)

    image = np.zeros((8, 8, 8), dtype=np.float32)
    nib.save(nib.Nifti1Image(image, affine=np.eye(4)), dataset_dir / "images" / "case1.nii.gz")
    (dataset_dir / "labels" / "case1.txt").write_text(
        "0 0.5 0.5 0.5 0.75 0.75 0.75\n1 0.5 0.5 0.5 0.25 0.25 0.25\n",
        encoding="utf-8",
    )

    masks = np.zeros((2, 8, 8, 8), dtype=np.uint8)
    masks[0, 1:7, 1:7, 1:7] = 1
    masks[1, 3:5, 3:5, 3:5] = 1
    nib.save(nib.Nifti1Image(masks, affine=np.eye(4)), dataset_dir / "masks" / "case1.nii.gz")

    output_dir = convert_dataset_to_integer_encoded_masks(dataset_dir)
    label_path = output_dir / "case1.nii.gz"

    assert label_path.exists(), "integer encoded mask missing"
    label_data = np.asanyarray(nib.load(label_path).dataobj)
    assert label_data[2, 2, 2] == 1
    assert label_data[3, 3, 3] == 2


def test_convert_dataset_to_integer_encoded_masks_squeezes_singleton_mask_axes(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    for subdir in ("images", "labels", "masks"):
        (dataset_dir / subdir).mkdir(parents=True, exist_ok=True)

    image = np.zeros((8, 8, 8), dtype=np.float32)
    nib.save(nib.Nifti1Image(image, affine=np.eye(4)), dataset_dir / "images" / "case1.nii.gz")
    (dataset_dir / "labels" / "case1.txt").write_text(
        "0 0.5 0.5 0.5 0.75 0.75 0.75\n1 0.5 0.5 0.5 0.25 0.25 0.25\n",
        encoding="utf-8",
    )

    masks = np.zeros((8, 8, 8, 1, 2), dtype=np.uint8)
    masks[1, 1, 1, 0, 0] = 1
    masks[6, 2, 4, 0, 0] = 1
    masks[2, 6, 5, 0, 0] = 1
    masks[3:5, 3:5, 3:5, 0, 1] = 1
    nib.save(nib.Nifti1Image(masks, affine=np.eye(4)), dataset_dir / "masks" / "case1.nii.gz")

    output_dir = convert_dataset_to_integer_encoded_masks(dataset_dir)
    label_path = output_dir / "case1.nii.gz"

    label_data = np.asanyarray(nib.load(label_path).dataobj)
    assert label_data.shape == (8, 8, 8)
    assert np.count_nonzero(label_data == 1) == 3
    assert np.count_nonzero(label_data == 2) == 2 * 2 * 2


def test_convert_dataset_to_integer_encoded_masks_writes_empty_mask_without_labels(
    tmp_path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    for subdir in ("images", "labels", "masks"):
        (dataset_dir / subdir).mkdir(parents=True, exist_ok=True)

    image = np.zeros((8, 8, 8), dtype=np.float32)
    nib.save(nib.Nifti1Image(image, affine=np.eye(4)), dataset_dir / "images" / "case1.nii.gz")

    masks = np.ones((8, 8, 8), dtype=np.uint8)
    nib.save(nib.Nifti1Image(masks, affine=np.eye(4)), dataset_dir / "masks" / "case1.nii.gz")

    output_dir = convert_dataset_to_integer_encoded_masks(dataset_dir)
    label_path = output_dir / "case1.nii.gz"

    label_data = np.asanyarray(nib.load(label_path).dataobj)
    assert label_data.shape == (8, 8, 8)
    assert np.count_nonzero(label_data) == 0


def test_conversion_skips_existing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    dataset_dir = tmp_path / "dataset"
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True)
    output_dir = dataset_dir / "masks_integer_encoded"
    output_dir.mkdir()
    (images_dir / "case1.nii.gz").touch()
    output_path = output_dir / "case1.nii.gz"
    output_path.write_bytes(b"existing output")

    def unexpected_load(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Already processed cases must not load their images")

    monkeypatch.setattr(nib, "load", unexpected_load)
    with caplog.at_level(logging.INFO):
        assert convert_dataset_to_integer_encoded_masks(dataset_dir) == output_dir

    assert output_path.read_bytes() == b"existing output"
    assert "1 of 1 cases already processed; converting 0 remaining cases." in caplog.text
    assert "0 cases skipped because of errors." in caplog.text


@pytest.mark.parametrize("error_type", [MemoryError, RuntimeError, OSError])
def test_conversion_continues_after_case_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error_type: type[Exception],
) -> None:
    dataset_dir = tmp_path / "dataset"
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True)
    for case_id in ("case1", "case2"):
        nib.save(
            nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), affine=np.eye(4)),
            images_dir / f"{case_id}.nii.gz",
        )

    original_save = nib.save

    def failing_save(image: nib.Nifti1Image, filename: Path, **kwargs: object) -> None:
        if filename.name == "case1.nii.gz":
            filename.write_bytes(b"partial output")
            msg = "simulated conversion failure"
            raise error_type(msg)
        original_save(image, filename, **kwargs)

    monkeypatch.setattr(nib, "save", failing_save)
    with caplog.at_level(logging.INFO):
        output_dir = convert_dataset_to_integer_encoded_masks(dataset_dir)

    assert not (output_dir / "case1.nii.gz").exists()
    assert (output_dir / "case2.nii.gz").is_file()
    assert "0 of 2 cases already processed; converting 2 remaining cases." in caplog.text
    assert f"Skipping case case1 because of {error_type.__name__}" in caplog.text
    assert "1 cases skipped because of errors." in caplog.text
