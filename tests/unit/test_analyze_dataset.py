from __future__ import annotations

import json
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np

from radyolo.config.spatial import PatchSize, VoxelSize
from radyolo.data.analyze_dataset import (
    _foreground_channel_stats_and_samples,
    _image_stats,
    _labels_to_mask,
    analyze_dataset,
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


def test_labels_to_mask_builds_union_mask_from_normalized_boxes() -> None:
    entries = [
        {
            "class_": 0,
            "box_relative_xyzwhd": (0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        }
    ]

    mask = _labels_to_mask(entries, PatchSize(x=4, y=4, z=4))

    expected = np.zeros((4, 4, 4), dtype=bool)
    expected[1:3, 1:3, 1:3] = True
    np.testing.assert_array_equal(mask, expected)


def test_image_stats_reads_channels_first_nifti_and_foreground_stats(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    labels_dir = dataset_path / "labels"
    labels_dir.mkdir(parents=True)

    image_path = tmp_path / "multi_channel.nii.gz"
    channel0 = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    channel1 = channel0 + 100.0
    array = np.stack([channel0, channel1], axis=0)
    affine = np.diag((1.0, 2.0, 3.0, 1.0))
    _write_nifti(image_path, array, affine)
    (labels_dir / "multi_channel.txt").write_text("0 0.5 0.5 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    stats = _image_stats(image_path, dataset_path, "multi_channel", sample_cap=100)

    foreground0 = channel0[1:3, 1:3, 1:3].astype(np.float32)
    foreground1 = channel1[1:3, 1:3, 1:3].astype(np.float32)

    assert stats.image_size == PatchSize(x=4, y=4, z=4)
    assert stats.voxel_size == VoxelSize(x=1.0, y=2.0, z=3.0)
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


def test_analyze_dataset_writes_resampled_summary_and_box_sizes(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir()

    case1_image = np.ones((10, 20, 30), dtype=np.float32)
    case2_image = np.ones((20, 40, 60), dtype=np.float32)
    _write_nifti(images_dir / "case1.nii.gz", case1_image, np.diag((1.0, 1.0, 1.0, 1.0)))
    _write_nifti(images_dir / "case2.nii.gz", case2_image, np.diag((0.5, 0.5, 0.5, 1.0)))
    for case_id in ("case1", "case2"):
        (labels_dir / f"{case_id}.txt").write_text(
            "0 0.5 0.5 0.5 0.5 0.5 0.5\n",
            encoding="utf-8",
        )

    fingerprint_path = analyze_dataset(dataset_path, workers=1)
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))

    assert fingerprint["median_voxel_size_mm"] == [0.75, 0.75, 0.75]
    assert fingerprint["voxel_size_p10_mm"] == [0.55, 0.55, 0.55]
    assert fingerprint["voxel_size_p90_mm"] == [0.95, 0.95, 0.95]
    assert fingerprint["median_resampled_imagesize"] == [13, 27, 40]
    assert fingerprint["resampled_imagesize_p10_vox"] == [13, 27, 40]
    assert fingerprint["resampled_imagesize_p90_vox"] == [13, 27, 40]
    assert fingerprint["box_size_median_vox"] == {"0": [6.5, 13.5, 20.0]}
    assert fingerprint["box_size_p05_vox"] == [6.5, 13.5, 20.0]
    assert fingerprint["box_size_p95_vox"] == [6.5, 13.5, 20.0]
    assert fingerprint["n_images_with_objects"] == 2
    assert fingerprint["n_images_without_objects"] == 0


def test_analyze_dataset_box_size_p95_uses_all_objects_from_all_classes(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir()

    image = np.ones((100, 100, 100), dtype=np.float32)
    _write_nifti(images_dir / "case1.nii.gz", image, np.diag((1.0, 1.0, 1.0, 1.0)))
    _write_nifti(images_dir / "case2.nii.gz", image, np.diag((1.0, 1.0, 1.0, 1.0)))

    (labels_dir / "case1.txt").write_text(
        "0 0.5 0.5 0.5 0.1 0.2 0.3\n1 0.5 0.5 0.5 0.4 0.5 0.6\n",
        encoding="utf-8",
    )
    (labels_dir / "case2.txt").write_text(
        "2 0.5 0.5 0.5 0.7 0.8 0.9\n",
        encoding="utf-8",
    )

    fingerprint_path = analyze_dataset(dataset_path, workers=1)
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))

    assert fingerprint["box_size_median_vox"] == {
        "0": [10.0, 20.0, 30.0],
        "1": [40.0, 50.0, 60.0],
        "2": [70.0, 80.0, 90.0],
    }
    assert fingerprint["box_size_p05_vox"] == [13.0, 23.0, 33.0]
    assert fingerprint["box_size_p95_vox"] == [67.0, 77.0, 87.0]


def test_analyze_dataset_counts_images_with_and_without_objects(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir()

    image = np.ones((12, 12, 12), dtype=np.float32)
    affine = np.diag((1.0, 1.0, 1.0, 1.0))
    _write_nifti(images_dir / "case_with.nii.gz", image, affine)
    _write_nifti(images_dir / "case_empty.nii.gz", image, affine)
    _write_nifti(images_dir / "case_missing.nii.gz", image, affine)

    (labels_dir / "case_with.txt").write_text("0 0.5 0.5 0.5 0.2 0.2 0.2\n", encoding="utf-8")
    (labels_dir / "case_empty.txt").write_text("", encoding="utf-8")
    (labels_dir / "case_missing.txt").write_text("", encoding="utf-8")

    fingerprint_path = analyze_dataset(dataset_path, workers=1)
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))

    assert fingerprint["n_images_with_objects"] == 1
    assert fingerprint["n_images_without_objects"] == 2


def test_analyze_dataset_summarizes_global_image_labels(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    labels_image_dir = dataset_path / "labels_image"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir()
    labels_image_dir.mkdir()

    image = np.ones((12, 12, 12), dtype=np.float32)
    affine = np.diag((1.0, 1.0, 1.0, 1.0))
    for case_id in ("case1", "case2", "case3"):
        _write_nifti(images_dir / f"{case_id}.nii.gz", image, affine)
        (labels_dir / f"{case_id}.txt").write_text("", encoding="utf-8")

    (labels_image_dir / "case1.json").write_text(
        json.dumps({"tumor_stage": 1, "survival_days": 100.0, "relapse": True}),
        encoding="utf-8",
    )
    (labels_image_dir / "case2.json").write_text(
        json.dumps({"tumor_stage": 2, "survival_days": 200.0, "relapse": False}),
        encoding="utf-8",
    )

    (dataset_path / "dataset.yaml").write_text(
        "names: {0: lesion}\n"
        "image_label_tasks:\n"
        "  tumor_stage:\n"
        "    type: multiclass\n"
        "    classes: {0: stage_i, 1: stage_ii, 2: stage_iii}\n"
        "  survival_days:\n"
        "    type: survival\n"
        "    masking: relapse\n"
        "    unit: days\n",
        encoding="utf-8",
    )

    fingerprint_path = analyze_dataset(dataset_path, workers=1)
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))

    assert fingerprint["image_label_tasks"] == {
        "survival_days": {
            "type": "survival",
            "n_labeled": 2,
            "n_missing": 1,
            "min": 100.0,
            "p05": 105.0,
            "median": 150.0,
            "p95": 195.0,
            "max": 200.0,
            "mean": 150.0,
            "std": 50.0,
            "n_events": 1,
            "n_censored": 1,
            "masking": "relapse",
            "unit": "days",
        },
        "tumor_stage": {
            "type": "multiclass",
            "n_labeled": 2,
            "n_missing": 1,
            "class_counts": {"1": 1, "2": 1},
            "classes": {"0": "stage_i", "1": "stage_ii", "2": "stage_iii"},
        },
    }


def test_analyze_dataset_treats_null_image_labels_as_missing(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    labels_image_dir = dataset_path / "labels_image"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir()
    labels_image_dir.mkdir()

    image = np.ones((12, 12, 12), dtype=np.float32)
    affine = np.diag((1.0, 1.0, 1.0, 1.0))
    for case_id in ("case1", "case2", "case3"):
        _write_nifti(images_dir / f"{case_id}.nii.gz", image, affine)
        (labels_dir / f"{case_id}.txt").write_text("", encoding="utf-8")

    (labels_image_dir / "case1.json").write_text(
        json.dumps({"tumor_stage": 1, "survival_days": 100.0, "relapse": True}),
        encoding="utf-8",
    )
    (labels_image_dir / "case2.json").write_text(
        json.dumps({"tumor_stage": None, "survival_days": None, "relapse": None}),
        encoding="utf-8",
    )

    (dataset_path / "dataset.yaml").write_text(
        "names: {0: lesion}\n"
        "image_label_tasks:\n"
        "  tumor_stage:\n"
        "    type: multiclass\n"
        "    classes: {0: stage_i, 1: stage_ii}\n"
        "  survival_days:\n"
        "    type: survival\n"
        "    masking: relapse\n"
        "    unit: days\n",
        encoding="utf-8",
    )

    fingerprint_path = analyze_dataset(dataset_path, workers=1)
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))

    assert fingerprint["image_label_tasks"] == {
        "survival_days": {
            "type": "survival",
            "n_labeled": 1,
            "n_missing": 2,
            "min": 100.0,
            "p05": 100.0,
            "median": 100.0,
            "p95": 100.0,
            "max": 100.0,
            "mean": 100.0,
            "std": 0.0,
            "n_events": 1,
            "n_censored": 0,
            "masking": "relapse",
            "unit": "days",
        },
        "tumor_stage": {
            "type": "multiclass",
            "n_labeled": 1,
            "n_missing": 2,
            "class_counts": {"1": 1},
            "classes": {"0": "stage_i", "1": "stage_ii"},
        },
    }
