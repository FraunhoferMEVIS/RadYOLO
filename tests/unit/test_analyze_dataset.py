from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np

from radyolo.data.analyze_dataset import (
    _foreground_channel_stats_and_samples,
    _image_stats,
    _labels_to_mask,
    _load_nifti,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_nifti(path: Path, array: np.ndarray, affine: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(array, affine=affine), path)


def test_foreground_channel_stats_and_samples_uses_foreground_voxels() -> None:
    array = np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2)
    foreground_mask = array >= 5

    stats, samples = _foreground_channel_stats_and_samples(array, foreground_mask)

    assert stats == {
        "min": 5.0,
        "max": 8.0,
        "sum": 26.0,
        "sumsq": 174.0,
        "count": 4.0,
    }
    assert samples.dtype == np.float32
    np.testing.assert_array_equal(samples, np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32))


def test_load_nifti_returns_data_and_spacing_in_zyx_order(tmp_path: Path) -> None:
    image_path = tmp_path / "case.nii.gz"
    array = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    affine = np.diag((0.5, 1.5, 2.5, 1.0))
    _write_nifti(image_path, array, affine)

    data, spacing = _load_nifti(image_path)

    np.testing.assert_array_equal(data, array)
    assert spacing == (2.5, 1.5, 0.5)


def test_labels_to_mask_builds_union_mask_from_normalized_boxes() -> None:
    entries = [(0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)]

    mask = _labels_to_mask(entries, (4, 4, 4))

    expected = np.zeros((4, 4, 4), dtype=bool)
    expected[1:3, 1:3, 1:3] = True
    np.testing.assert_array_equal(mask, expected)


def test_image_stats_reads_channels_first_nifti_and_foreground_stats(tmp_path: Path) -> None:
    image_path = tmp_path / "multi_channel.nii.gz"
    labels_path = tmp_path / "multi_channel.txt"
    channel0 = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    channel1 = channel0 + 100.0
    array = np.stack([channel0, channel1], axis=0)
    affine = np.diag((1.0, 2.0, 3.0, 1.0))
    _write_nifti(image_path, array, affine)
    labels_path.write_text("0 0.5 0.5 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    stats = _image_stats(image_path, labels_path, sample_cap=100)

    foreground0 = channel0[1:3, 1:3, 1:3].astype(np.float32)
    foreground1 = channel1[1:3, 1:3, 1:3].astype(np.float32)

    assert stats.shape == (4, 4, 4)
    assert stats.spacing == (3.0, 2.0, 1.0)
    assert len(stats.per_channel) == 2
    assert stats.per_channel[0] == {
        "min": float(foreground0.min()),
        "max": float(foreground0.max()),
        "sum": float(foreground0.sum(dtype=np.float64)),
        "sumsq": float(np.square(foreground0, dtype=np.float64).sum(dtype=np.float64)),
        "count": float(foreground0.size),
    }
    assert stats.per_channel[1] == {
        "min": float(foreground1.min()),
        "max": float(foreground1.max()),
        "sum": float(foreground1.sum(dtype=np.float64)),
        "sumsq": float(np.square(foreground1, dtype=np.float64).sum(dtype=np.float64)),
        "count": float(foreground1.size),
    }
    np.testing.assert_array_equal(stats.samples_per_channel[0], foreground0.reshape(-1))
    np.testing.assert_array_equal(stats.samples_per_channel[1], foreground1.reshape(-1))
