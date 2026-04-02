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


def test_split_thresholds_can_be_overridden_explicitly(tmp_path: Path) -> None:
    cfg = DefaultConfig.model_validate(
        {
            **_make_minimal_config(tmp_path),
            "confidence_threshold_validation": 0.02,
            "confidence_threshold_plotting": 0.15,
        }
    )

    assert cfg.confidence_threshold_validation == 0.02
    assert cfg.confidence_threshold_plotting == 0.15
