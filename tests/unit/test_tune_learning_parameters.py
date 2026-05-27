from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hyperparameter_tuning.tune_learning_parameters import (
    _alphabetical_restore_trial_sort_key,
    _build_dataset_train_overrides,
    _build_initial_evaluation_point,
    _compute_run_config_signature,
    _ensure_project_root_on_sys_path,
    _load_tune_settings,
    _get_cached_completed_dataset_metrics,
    _restore_completed_dataset_metrics,
    _should_resume_dataset_run,
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


def test_load_tune_settings_validates_template_without_dataset_yaml(
    tmp_path: Path,
    synthetic_dataset: Path,
) -> None:
    template_config_path = tmp_path / "template.yaml"
    template_config_path.write_text(
        (
            "input_channels: 1\n"
            "model_scale: n\n"
            "data: D:/datasets/cluster_only/template_dataset\n"
            "fold: 0\n"
            "epochs: 1\n"
            "nominal_batch_size: 1\n"
            "warmup_epochs: 0.0\n"
            "patience: 1\n"
            "batch_size: 1\n"
            "patch_size: {x: 32, y: 32, z: 64}\n"
            "normalization: {type: 'none'}\n"
            "device: cpu\n"
            "output_folder_base: runs\n"
            "experiment_name: template\n"
            "exist_ok: true\n"
            "optimizer: Adam\n"
            "verbose: false\n"
            "seed: 0\n"
            "resume: false\n"
            "amp: false\n"
            "use_cudnn_benchmark: false\n"
            "val: true\n"
            "plots: false\n"
            "augment: false\n"
        ),
        encoding="utf-8",
    )

    tune_config_path = tmp_path / "tune.yaml"
    tune_config_path.write_text(
        (
            f"template_config: {template_config_path.as_posix()}\n"
            "search_space:\n"
            "  workers:\n"
            "    type: choice\n"
            "    values: [0, 6]\n"
            "datasets:\n"
            f"  - data: {synthetic_dataset.as_posix()}\n"
        ),
        encoding="utf-8",
    )

    _, _, _, _, tune_settings, initial_evaluation_point = _load_tune_settings(
        tune_config_path
    )

    assert tune_settings["datasets"][0]["data"] == synthetic_dataset.as_posix()
    assert initial_evaluation_point == {"workers": 6}


def test_alphabetical_restore_trial_sort_key_orders_trial_ids_alphabetically() -> None:
    class TrialStub:
        def __init__(self, trial_id: str, local_path: str) -> None:
            self.trial_id = trial_id
            self.local_path = local_path

    trials = [
        TrialStub("c3", "c"),
        TrialStub("A1", "a"),
        TrialStub("b2", "b"),
    ]

    sorted_trial_ids = [
        trial.trial_id
        for trial in sorted(trials, key=_alphabetical_restore_trial_sort_key)
    ]

    assert sorted_trial_ids == ["A1", "b2", "c3"]


def test_should_resume_dataset_run_uses_matching_incomplete_progress_entry(tmp_path: Path) -> None:
    config_signature = _compute_run_config_signature({"epochs": 10, "resume": False})
    checkpoint_path = tmp_path / "runs" / "train" / "weights_last.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    should_resume = _should_resume_dataset_run(
        {"config_signature": config_signature, "status": "interrupted"},
        config_signature,
        checkpoint_path,
        tmp_path / "train_config.yaml",
    )

    assert should_resume is True


def test_should_resume_dataset_run_uses_existing_train_config_for_legacy_runs(tmp_path: Path) -> None:
    run_config = {"epochs": 10, "resume": False, "batch_size": 2}
    config_signature = _compute_run_config_signature(run_config)
    config_path = tmp_path / "train_config.yaml"
    config_path.write_text(
        "epochs: 10\nresume: false\nbatch_size: 2\n",
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "runs" / "train" / "weights_last.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    should_resume = _should_resume_dataset_run(
        None,
        config_signature,
        checkpoint_path,
        config_path,
    )

    assert should_resume is True


def test_get_cached_completed_dataset_metrics_requires_completed_status() -> None:
    config_signature = _compute_run_config_signature({"epochs": 5})

    cached_metrics = _get_cached_completed_dataset_metrics(
        {
            "config_signature": config_signature,
            "status": "completed",
            "primary_metric": 0.8,
            "secondary_metric": 0.7,
            "primary_best_epoch": 8,
            "secondary_best_epoch": 7,
            "training_minutes": 12.5,
        },
        config_signature,
    )

    assert cached_metrics == (0.8, 0.7, 8, 7, 12.5)


def test_restore_completed_dataset_metrics_finalizes_training_completed_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_signature = _compute_run_config_signature({"epochs": 12, "batch_size": 2})
    trial_progress = {
        "datasets": {
            "dataset_a": {
                "config_signature": config_signature,
                "status": "training_completed",
                "training_minutes": 9.75,
            }
        }
    }
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("epoch,metrics/mAP10-95,metrics/dice\n1,0.1,0.2\n", encoding="utf-8")

    monkeypatch.setattr(
        "hyperparameter_tuning.tune_learning_parameters._extract_metric_from_csv",
        lambda *args, **kwargs: (0.81, 0.72, 11, 10),
    )

    metrics = _restore_completed_dataset_metrics(
        trial_progress["datasets"]["dataset_a"],
        config_signature,
        csv_path,
        "metrics/mAP10-95",
        "metrics/dice",
        5,
        trial_progress,
        tmp_path,
        "dataset_a",
    )

    assert metrics == (0.81, 0.72, 11, 10, 9.75)
    assert trial_progress["datasets"]["dataset_a"]["status"] == "completed"
    assert trial_progress["datasets"]["dataset_a"]["primary_metric"] == pytest.approx(0.81)
