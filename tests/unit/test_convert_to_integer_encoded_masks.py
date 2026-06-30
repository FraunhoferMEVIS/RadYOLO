from __future__ import annotations

import nibabel as nib
import numpy as np

from radyolo.data.convert_to_integer_encoded_masks import convert_dataset_to_integer_encoded_masks


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
