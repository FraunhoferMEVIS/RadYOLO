from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hyperparameter_tuning.tune_learning_parameters import (
    _build_dataset_train_overrides,
    _build_initial_evaluation_point,
    _ensure_project_root_on_sys_path,
)


def test_ensure_project_root_on_sys_path_inserts_repo_root_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root_path = tmp_path / "RadYOLO"
    script_dir = project_root_path / "hyperparameter_tuning"
    script_path = script_dir / "tune_learning_parameters.py"
    monkeypatch.setattr(sys, "path", [str(script_dir)])

    project_root = _ensure_project_root_on_sys_path(script_path)
    _ensure_project_root_on_sys_path(script_path)

    assert project_root == project_root_path.resolve()
    assert sys.path[0] == str(project_root_path.resolve())
    assert sys.path.count(str(project_root_path.resolve())) == 1


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


def test_build_initial_evaluation_point_skips_invalid_choice_defaults() -> None:
    initial_point = _build_initial_evaluation_point(
        template_config={"optimizer": "Adam", "model_scale": "s"},
        effective_template_config={"optimizer": "Adam", "model_scale": "s"},
        search_space_cfg={
            "optimizer": {"type": "choice", "values": ["AdamW", "SGD"]},
            "model_scale": {"type": "choice", "values": ["n", "s", "m"]},
        },
        cast_hints={},
    )

    assert initial_point == {"model_scale": "s"}


def test_build_initial_evaluation_point_skips_one_minus_loguniform_defaults() -> None:
    initial_point = _build_initial_evaluation_point(
        template_config={"ema_decay": 0.9999, "prototype_masks": 8},
        effective_template_config={"ema_decay": 0.9999, "prototype_masks": 8},
        search_space_cfg={
            "ema_decay": {
                "type": "one_minus_loguniform",
                "lower": 0.1,
                "upper": 0.99999,
            },
            "prototype_masks": {
                "type": "qrandint",
                "lower": 4,
                "upper": 24,
                "q": 1,
            },
        },
        cast_hints={"prototype_masks": "int"},
    )

    assert initial_point == {"prototype_masks": 8}
