from __future__ import annotations

import pytest
import numpy as np

from radyolo.config.propose_model_config import (
    _candidate_rank,
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


def test_propose_patch_size_stays_within_budget() -> None:
    fingerprint = {
        "median_resampled_imagesize": [224, 224, 96],
        "box_size_median_vox": {"0": [40, 40, 18]},
        "box_size_p95_vox": [40, 40, 18],
    }

    patch_size, estimated_memory_gb = propose_patch_size(
        fingerprint,
        scale="s",
        gpu_ram_limit_gb=6.0,
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
    assert estimated_memory_gb <= 6.0
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
    assert direct_estimate == pytest.approx(estimated_memory_gb, abs=0.05)


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


def test_candidate_rank_prefers_resampled_axis_ratios_when_object_fit_is_equal() -> None:
    target_ratio_xyz = np.array([320.0, 160.0, 96.0])
    representative_object_xyz = np.array([48.0, 48.0, 48.0])
    preferred_lower_bound_xyz = np.array([96.0, 64.0, 64.0])

    ratio_aligned = _candidate_rank(
        np.array([192, 96, 64]),
        target_ratio_xyz=target_ratio_xyz,
        representative_object_xyz=representative_object_xyz,
        preferred_lower_bound_xyz=preferred_lower_bound_xyz,
        estimated_memory_gb=3.0,
    )
    ratio_distorted = _candidate_rank(
        np.array([160, 128, 64]),
        target_ratio_xyz=target_ratio_xyz,
        representative_object_xyz=representative_object_xyz,
        preferred_lower_bound_xyz=preferred_lower_bound_xyz,
        estimated_memory_gb=3.0,
    )

    assert ratio_aligned > ratio_distorted


def test_candidate_rank_prefers_more_median_objects() -> None:
    target_ratio_xyz = np.array([160.0, 160.0, 160.0])
    representative_object_xyz = np.array([96.0, 96.0, 32.0])
    preferred_lower_bound_xyz = np.array([128.0, 128.0, 64.0])

    more_objects = _candidate_rank(
        np.array([96, 96, 96]),
        target_ratio_xyz=target_ratio_xyz,
        representative_object_xyz=representative_object_xyz,
        preferred_lower_bound_xyz=preferred_lower_bound_xyz,
        estimated_memory_gb=3.0,
    )
    fewer_objects = _candidate_rank(
        np.array([128, 96, 64]),
        target_ratio_xyz=target_ratio_xyz,
        representative_object_xyz=representative_object_xyz,
        preferred_lower_bound_xyz=preferred_lower_bound_xyz,
        estimated_memory_gb=3.0,
    )

    assert more_objects > fewer_objects