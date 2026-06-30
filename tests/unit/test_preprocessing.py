from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from radyolo.data import data as data_module
from radyolo.data.preprocess_data import _load_nifti_tensor


def test_load_nifti_tensor_optionally_squeezes_mask_axes(monkeypatch) -> None:
    fake_image = nib.nifti1.Nifti1Image(np.zeros((6, 8, 10, 1, 4), dtype=np.uint8), np.eye(4))
    monkeypatch.setattr("radyolo.data.preprocess_data.nib.loadsave.load", lambda _path: fake_image)

    tensor, voxel_size = _load_nifti_tensor(Path("mask.nii.gz"), squeeze=True)

    assert tuple(tensor.shape) == (4, 10, 8, 6)
    assert voxel_size.x == 1.0
    assert voxel_size.y == 1.0
    assert voxel_size.z == 1.0


def test_load_mask_array_squeezes_nifti_singleton_axes(monkeypatch, tmp_path) -> None:
    fake_image = nib.nifti1.Nifti1Image(np.zeros((6, 8, 10, 1, 4), dtype=np.uint8), np.eye(4))
    monkeypatch.setattr("radyolo.data.data.nib.loadsave.load", lambda _path: fake_image)

    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    (masks_dir / "case.nii.gz").write_bytes(b"placeholder")

    mask = data_module._load_mask_array(
        tmp_path,
        "case",
        "nifti",
        load_masks=True,
    )

    assert mask is not None
    assert mask.shape == (6, 8, 10, 4)
