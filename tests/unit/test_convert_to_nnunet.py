from __future__ import annotations

import json

import nibabel as nib
import numpy as np

from radyolo.data.convert_to_nnunet import (
    _compose_semantic_seg_from_masks,
    convert_dataset_to_nnunet,
)


def test_compose_semantic_seg_from_masks_writes_smaller_instances_last() -> None:
    mask_arr = np.zeros((2, 5, 5, 5), dtype=np.uint8)
    mask_arr[0, 1:5, 1:5, 1:5] = 1
    mask_arr[1, 2:4, 2:4, 2:4] = 1
    labels_tuples = [
        (0, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8),
        (1, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2),
    ]

    semantic = _compose_semantic_seg_from_masks(mask_arr, labels_tuples)

    assert semantic is not None
    assert semantic[2, 2, 2] == 2
    assert semantic[1, 1, 1] == 1


def test_convert_dataset_to_nnunet_writes_empty_mask_for_unlabeled_case(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    for subdir in ("images", "labels"):
        (dataset_dir / subdir).mkdir(parents=True, exist_ok=True)

    image = np.zeros((8, 8, 8), dtype=np.float32)
    nib.save(nib.Nifti1Image(image, affine=np.eye(4)), dataset_dir / "images" / "empty_case.nii.gz")
    (dataset_dir / "dataset.yaml").write_text("names: {0: lesion}\n", encoding="utf-8")
    (dataset_dir / "data_splits.json").write_text(
        json.dumps([{"train": ["empty_case"], "val": [], "test": []}]),
        encoding="utf-8",
    )

    out_dir = convert_dataset_to_nnunet(dataset_dir=dataset_dir, out_root=tmp_path, workers=0)

    label_path = out_dir / "labelsTr" / "empty_case.nii.gz"
    assert label_path.exists(), "empty nnU-Net label mask missing"
    label_data = np.asanyarray(nib.load(label_path).dataobj)
    assert np.count_nonzero(label_data) == 0


def test_convert_dataset_to_nnunet_writes_test_masks_to_labels_ts(
    synthetic_dataset, tmp_path
) -> None:
    out_dir = convert_dataset_to_nnunet(dataset_dir=synthetic_dataset, out_root=tmp_path, workers=0)

    test_label_path = out_dir / "labelsTs" / "case5.nii.gz"

    assert test_label_path.exists(), "test nnU-Net label mask missing"
    test_label_data = np.asanyarray(nib.load(test_label_path).dataobj)
    assert np.count_nonzero(test_label_data) > 0
