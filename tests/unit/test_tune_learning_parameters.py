from __future__ import annotations

import pytest

from hyperparameter_tuning.tune_learning_parameters import _build_dataset_train_overrides


def test_build_dataset_train_overrides_accepts_named_patch_size_mapping() -> None:
    overrides = _build_dataset_train_overrides(
        {
            "data": "D:/datasets/example",
            "resampling": {"patch_size": {"x": 96, "y": 80, "z": 64}},
        },
        index=1,
    )

    assert overrides["patch_size"] == {"x": 96, "y": 80, "z": 64}


def test_build_dataset_train_overrides_normalizes_patch_size_sequence_to_named_axes() -> None:
    overrides = _build_dataset_train_overrides(
        {
            "data": "D:/datasets/example",
            "resampling": {"patch_size": [320, 320, 32]},
        },
        index=1,
    )

    assert overrides["patch_size"] == {"x": 320, "y": 320, "z": 32}
