from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from radyolo.config.config import DefaultConfig

if TYPE_CHECKING:
    from pathlib import Path


def _make_minimal_config(tmp_path: Path) -> dict[str, object]:
    return {
        "input_channels": 1,
        "num_classes": 1,
        "model_scale": "n",
        "data": str(tmp_path),
        "fold": 0,
        "batch_size": 1,
        "patch_size": {"x": 16, "y": 16, "z": 16},
        "experiment_name": "confidence-threshold-test",
    }


def test_default_config_uses_split_validation_and_plotting_thresholds(tmp_path: Path) -> None:
    cfg = DefaultConfig.model_validate(_make_minimal_config(tmp_path))

    assert cfg.confidence_threshold_validation == 0.001
    assert cfg.confidence_threshold_plotting == 0.1
    assert cfg.nms_iou_threshold_candidates == [0.02, 0.04, 0.06, 0.1, 0.15, 0.2]


def test_split_thresholds_can_be_overridden_explicitly(tmp_path: Path) -> None:
    cfg = DefaultConfig.model_validate(
        {
            **_make_minimal_config(tmp_path),
            "confidence_threshold_validation": 0.02,
            "confidence_threshold_plotting": 0.15,
            "nms_iou_threshold_candidates": [0.25, 0.1, 0.25],
        }
    )

    assert cfg.confidence_threshold_validation == 0.02
    assert cfg.confidence_threshold_plotting == 0.15
    assert cfg.nms_iou_threshold_candidates == [0.1, 0.25]


def test_nms_threshold_candidates_must_be_in_unit_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nms_iou_threshold_candidates"):
        DefaultConfig.model_validate(
            {
                **_make_minimal_config(tmp_path),
                "nms_iou_threshold_candidates": [0.1, 1.2],
            }
        )


def test_rotation_angle_scalar_expands_to_all_axes(tmp_path: Path) -> None:
    cfg = DefaultConfig.model_validate(
        {
            **_make_minimal_config(tmp_path),
            "rotation_angle": 15,
        }
    )

    assert cfg.rotation_angle_x == (-15.0, 15.0)
    assert cfg.rotation_angle_y == (-15.0, 15.0)
    assert cfg.rotation_angle_z == (-15.0, 15.0)


def test_rotation_angle_scalar_does_not_override_explicit_axis_ranges(tmp_path: Path) -> None:
    cfg = DefaultConfig.model_validate(
        {
            **_make_minimal_config(tmp_path),
            "rotation_angle": 15,
            "rotation_angle_y": (-10.0, 10.0),
        }
    )

    assert cfg.rotation_angle_x == (-15.0, 15.0)
    assert cfg.rotation_angle_y == (-10.0, 10.0)
    assert cfg.rotation_angle_z == (-15.0, 15.0)
