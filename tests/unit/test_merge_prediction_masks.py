from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from radyolo.inference.merge_prediction_masks import merge_prediction_masks


def _write_binary_mask(path: Path, active_voxels: tuple[slice, slice, slice]) -> None:
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[active_voxels] = 1
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(path))


def test_merge_prediction_masks_uses_descending_box_size(tmp_path: Path) -> None:
    predictions_dir = tmp_path / "predictions"
    masks_dir = predictions_dir / "masks"
    predictions_dir.mkdir()
    masks_dir.mkdir()

    _write_binary_mask(masks_dir / "case_det0.nii.gz", (slice(1, 9), slice(1, 9), slice(1, 9)))
    _write_binary_mask(masks_dir / "case_det1.nii.gz", (slice(4, 7), slice(4, 7), slice(4, 7)))

    (predictions_dir / "case.json").write_text(
        json.dumps(
            [
                {
                    "class": 0,
                    "confidence": 0.95,
                    "box_image_xyzxyz": [1.0, 1.0, 1.0, 9.0, 9.0, 9.0],
                    "mask_path": "case_det0.nii.gz",
                },
                {
                    "class": 1,
                    "confidence": 0.90,
                    "box_image_xyzxyz": [4.0, 4.0, 4.0, 7.0, 7.0, 7.0],
                    "mask_path": "case_det1.nii.gz",
                },
            ]
        ),
        encoding="utf-8",
    )

    output_dir = merge_prediction_masks(predictions_dir, confidence_threshold=0.1)

    merged = np.asanyarray(nib.load(output_dir / "case.nii.gz").dataobj)
    assert merged[2, 2, 2] == 1  # large object (class 0 -> label 1)
    assert merged[5, 5, 5] == 2  # small object wins overlap (class 1 -> label 2)


def test_merge_prediction_masks_applies_confidence_threshold(tmp_path: Path) -> None:
    predictions_dir = tmp_path / "predictions"
    masks_dir = predictions_dir / "masks"
    predictions_dir.mkdir()
    masks_dir.mkdir()

    _write_binary_mask(masks_dir / "case_det0.nii.gz", (slice(1, 9), slice(1, 9), slice(1, 9)))
    _write_binary_mask(masks_dir / "case_det1.nii.gz", (slice(4, 7), slice(4, 7), slice(4, 7)))

    (predictions_dir / "case.json").write_text(
        json.dumps(
            [
                {
                    "class": 0,
                    "confidence": 0.95,
                    "box_image_xyzxyz": [1.0, 1.0, 1.0, 9.0, 9.0, 9.0],
                    "mask_path": "case_det0.nii.gz",
                },
                {
                    "class": 1,
                    "confidence": 0.05,
                    "box_image_xyzxyz": [4.0, 4.0, 4.0, 7.0, 7.0, 7.0],
                    "mask_path": "case_det1.nii.gz",
                },
            ]
        ),
        encoding="utf-8",
    )

    output_dir = merge_prediction_masks(predictions_dir, confidence_threshold=0.1)

    merged = np.asanyarray(nib.load(output_dir / "case.nii.gz").dataobj)
    assert merged[5, 5, 5] == 1
