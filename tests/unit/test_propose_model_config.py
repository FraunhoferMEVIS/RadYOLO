from __future__ import annotations

import pytest
import numpy as np

from radyolo.config.propose_model_config import (
    build_runtime_normalization_config,
    estimate_patch_memory_gb,
    propose_patch_size,
    resolve_dataset_normalization_tags,
)


def test_resolve_dataset_normalization_tags_defaults_missing_channels_to_mrperchannel() -> None:
    tags = resolve_dataset_normalization_tags({"normalization": {0: "CT"}}, 2)

    assert tags == {0: "CT", 1: "MRPerChannel"}


def test_build_runtime_normalization_config_uses_global_zscore_for_all_mrglobal() -> None:
    fingerprint = {
        "foreground_channel_stats": [
            {"mean": 1.0, "std": 2.0, "p00_5": -1.0, "p99_5": 3.0},
            {"mean": 4.0, "std": 5.0, "p00_5": -2.0, "p99_5": 6.0},
        ]
    }

    normalization = build_runtime_normalization_config(
        {0: "MRGlobal", 1: "MRGlobal"},
        fingerprint,
    )

    assert normalization == {"type": "zscore"}


def test_build_runtime_normalization_config_uses_ct_stats_and_per_channel_zscore() -> None:
    fingerprint = {
        "foreground_channel_stats": [
            {"mean": 10.0, "std": 2.5, "p00_5": -100.0, "p99_5": 200.0},
            {"mean": 1.0, "std": 1.5, "p00_5": -2.0, "p99_5": 4.0},
        ]
    }

    normalization = build_runtime_normalization_config(
        {0: "CT", 1: "MRPerChannel"},
        fingerprint,
    )

    assert normalization == [
        {
            "type": "ct",
            "mean": 10.0,
            "std": 2.5,
            "clip_min": -100.0,
            "clip_max": 200.0,
        },
        {"type": "zscore"},
    ]


def test_propose_patch_size_returns_stride_aligned_estimate() -> None:
    fingerprint = {
        "median_resampled_imagesize": [224, 224, 96],
        "box_size_median_vox": {"0": [40, 40, 18]},
        "box_size_p95_vox": [40, 40, 18],
    }

    patch_size, estimated_memory_gb = propose_patch_size(
        fingerprint,
        scale="s",
        batch_size=2,
        input_channels=1,
        num_classes=1,
        segmentation_output=True,
        include_stride4_output=False,
        prototype_masks=8,
    )

    assert patch_size.x % 32 == 0
    assert patch_size.y % 32 == 0
    assert patch_size.z % 32 == 0
    direct_estimate = estimate_patch_memory_gb(
        patch_size.get_xyz(),
        scale="s",
        batch_size=2,
        input_channels=1,
        num_classes=1,
        segmentation_output=True,
        include_stride4_output=False,
        prototype_masks=8,
    )
    assert direct_estimate == pytest.approx(estimated_memory_gb, abs=0.2)


def test_propose_patch_size_targets_256_cubed_volume_and_median_ratios() -> None:
    fingerprint = {
        "median_resampled_imagesize": [160, 96, 64],
        "box_size_median_vox": {"0": [20, 14, 10]},
        "box_size_p95_vox": [32, 20, 14],
    }

    patch_size, _ = propose_patch_size(
        fingerprint,
        scale="s",
        batch_size=2,
        input_channels=1,
        num_classes=1,
        segmentation_output=True,
        include_stride4_output=False,
        prototype_masks=8,
    )

    candidate = np.asarray(patch_size.get_xyz(), dtype=np.float64)
    median = np.asarray(fingerprint["median_resampled_imagesize"], dtype=np.float64)
    candidate_ratio = candidate / np.cbrt(np.prod(candidate))
    median_ratio = median / np.cbrt(np.prod(median))

    assert patch_size.x % 32 == 0
    assert patch_size.y % 32 == 0
    assert patch_size.z % 32 == 0
    assert np.prod(candidate) == pytest.approx(256**3, rel=0.25)
    assert np.linalg.norm(np.log(candidate_ratio / median_ratio)) < 0.2


def test_estimate_patch_memory_gb_increases_with_patch_volume() -> None:
    smaller = estimate_patch_memory_gb(
        (64, 64, 64),
        scale="n",
        batch_size=1,
        input_channels=1,
        num_classes=1,
        segmentation_output=False,
        include_stride4_output=False,
        prototype_masks=8,
    )
    larger = estimate_patch_memory_gb(
        (96, 96, 96),
        scale="n",
        batch_size=1,
        input_channels=1,
        num_classes=1,
        segmentation_output=False,
        include_stride4_output=False,
        prototype_masks=8,
    )

    assert larger > smaller


def test_estimate_patch_memory_gb_increases_with_loss_objects() -> None:
    fewer_objects = estimate_patch_memory_gb(
        (96, 96, 96),
        scale="n",
        batch_size=1,
        input_channels=1,
        num_classes=1,
        segmentation_output=True,
        include_stride4_output=False,
        prototype_masks=8,
        estimated_boxes_per_image=1,
    )
    more_objects = estimate_patch_memory_gb(
        (96, 96, 96),
        scale="n",
        batch_size=1,
        input_channels=1,
        num_classes=1,
        segmentation_output=True,
        include_stride4_output=False,
        prototype_masks=8,
        estimated_boxes_per_image=8,
    )

    assert more_objects > fewer_objects
