from __future__ import annotations

import json

import pytest
import numpy as np
import yaml

from radyolo.config.propose_model_config import (
    build_runtime_normalization_config,
    propose_model_config as propose_model_config_fn,
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


def test_propose_patch_size_caps_by_stride_rounded_median_image_size() -> None:
    fingerprint = {
        "median_resampled_imagesize": [150, 95, 65],
        "box_size_median_vox": {"0": [40, 40, 18]},
        "box_size_p95_vox": [40, 40, 18],
    }

    patch_size = propose_patch_size(fingerprint)

    assert patch_size.get_xyz() == (160, 96, 96)
    assert patch_size.x % 32 == 0
    assert patch_size.y % 32 == 0
    assert patch_size.z % 32 == 0


def test_propose_patch_size_preserves_ratios_when_median_cap_does_not_bind() -> None:
    fingerprint = {
        "median_resampled_imagesize": [512, 320, 192],
        "box_size_median_vox": {"0": [20, 14, 10]},
        "box_size_p95_vox": [32, 20, 14],
    }

    patch_size = propose_patch_size(fingerprint)

    candidate = np.asarray(patch_size.get_xyz(), dtype=np.float64)
    median = np.asarray(fingerprint["median_resampled_imagesize"], dtype=np.float64)
    median_cap = np.ceil(median / 32.0) * 32.0
    candidate_ratio = candidate / np.cbrt(np.prod(candidate))
    median_ratio = median / np.cbrt(np.prod(median))

    assert patch_size.x % 32 == 0
    assert patch_size.y % 32 == 0
    assert patch_size.z % 32 == 0
    assert np.all(candidate <= median_cap)
    assert np.prod(candidate) == pytest.approx(256**3, rel=0.25)
    assert np.linalg.norm(np.log(candidate_ratio / median_ratio)) < 0.2


def test_propose_model_config_automatic_selects_patchwise_and_sets_voxel_size(
    tmp_path,
) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    (dataset_path / "dataset.yaml").write_text(
        "names: {0: lesion}\nnormalization: {0: CT}\n",
        encoding="utf-8",
    )
    (dataset_path / "dataset_fingerprint.json").write_text(
        json.dumps(
            {
                "median_resampled_imagesize": [512, 320, 192],
                "median_voxel_size_mm": [1.5, 1.0, 2.0],
                "foreground_channel_stats": [
                    {"mean": 10.0, "std": 2.5, "p00_5": -100.0, "p99_5": 200.0}
                ],
                "n_images_with_masks": 0,
            }
        ),
        encoding="utf-8",
    )

    result = propose_model_config_fn(dataset_path=dataset_path, model_scale="s")

    with result.output_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    assert result.training_mode == "patchwise"
    assert config["training_mode"] == "patchwise"
    assert config["voxel_size"] == {"x": 1.5, "y": 1.0, "z": 2.0}


def test_propose_model_config_explicit_wholeimage_leaves_voxel_size_null(tmp_path) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    (dataset_path / "dataset.yaml").write_text(
        "names: {0: lesion}\nnormalization: {0: CT}\n",
        encoding="utf-8",
    )
    (dataset_path / "dataset_fingerprint.json").write_text(
        json.dumps(
            {
                "median_resampled_imagesize": [512, 320, 192],
                "median_voxel_size_mm": [1.5, 1.0, 2.0],
                "foreground_channel_stats": [
                    {"mean": 10.0, "std": 2.5, "p00_5": -100.0, "p99_5": 200.0}
                ],
                "n_images_with_masks": 0,
            }
        ),
        encoding="utf-8",
    )

    result = propose_model_config_fn(
        dataset_path=dataset_path,
        model_scale="s",
        training_mode="wholeimage",
    )

    with result.output_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    assert result.training_mode == "wholeimage"
    assert config["training_mode"] == "wholeimage"
    assert config["voxel_size"] is None
