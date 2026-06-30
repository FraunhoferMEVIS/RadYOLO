from __future__ import annotations

import pytest

from radyolo.config.config import DefaultConfig
from radyolo.training.loss import DetectionSegmentationLoss


def test_tal_topk_schedule_interpolates_linearly_across_epochs() -> None:
    assert (
        DetectionSegmentationLoss.interpolate_tal_topk(10, 4, current_epoch=1, final_epoch=5) == 10
    )
    assert (
        DetectionSegmentationLoss.interpolate_tal_topk(10, 4, current_epoch=2, final_epoch=5) == 8
    )
    assert (
        DetectionSegmentationLoss.interpolate_tal_topk(10, 4, current_epoch=3, final_epoch=5) == 7
    )
    assert (
        DetectionSegmentationLoss.interpolate_tal_topk(10, 4, current_epoch=4, final_epoch=5) == 6
    )
    assert (
        DetectionSegmentationLoss.interpolate_tal_topk(10, 4, current_epoch=5, final_epoch=5) == 4
    )


def test_tal_topk_schedule_clamps_after_final_epoch() -> None:
    assert (
        DetectionSegmentationLoss.interpolate_tal_topk(10, 4, current_epoch=9, final_epoch=5) == 4
    )


def test_tal_topk_schedule_handles_single_epoch_transition() -> None:
    assert (
        DetectionSegmentationLoss.interpolate_tal_topk(10, 4, current_epoch=1, final_epoch=1) == 4
    )


def test_default_config_rejects_non_positive_tal_topk_final_epoch(tmp_path) -> None:
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        DefaultConfig.model_validate(
            {
                "input_channels": 1,
                "num_classes": 1,
                "model_scale": "n",
                "data": tmp_path,
                "fold": 0,
                "patch_size": {"x": 32, "y": 32, "z": 32},
                "tal_topk_final_epoch": 0,
            }
        )
