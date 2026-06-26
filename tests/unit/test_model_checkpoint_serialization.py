from __future__ import annotations

import datetime
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
import yaml

from radyolo.config.config import DefaultConfig, get_cfg, load_model_checkpoint, save_model_checkpoint
from radyolo.config.spatial import PatchSize, VoxelSize
from radyolo.models.model import build_model_from_config
from radyolo.training.train import Trainer
from radyolo.utils.util import ModelEMA

def test_load_model_checkpoint_rejects_legacy_numpy_metadata(
    minimal_train_config: Path,
    tmp_path: Path,
) -> None:
    config = get_cfg(minimal_train_config, {})
    model = build_model_from_config(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.GradScaler("cpu", enabled=False)
    checkpoint_path = tmp_path / "legacy_checkpoint.pt"
    legacy_checkpoint = {
        "epoch": 3,
        "best_fitness": 0.5,
        "model_state_dict": model.state_dict(),
        "ema_updates": 12,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "train_config": config.model_dump_json(),
        "train_metrics": {"metrics/fitness": np.float32(0.25)},
        "date": datetime.datetime.now(tz=datetime.UTC).isoformat(),
    }

    torch.save(legacy_checkpoint, checkpoint_path)

    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        load_model_checkpoint(checkpoint_path)


def test_save_model_checkpoint_writes_weights_only_safe_metadata(
    minimal_train_config: Path,
    tmp_path: Path,
) -> None:
    config = get_cfg(minimal_train_config, {})
    model = build_model_from_config(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.param_groups[0]["lr"] = np.float64(0.001)
    optimizer.param_groups[0]["initial_lr"] = np.float64(0.001)
    scaler = torch.GradScaler("cpu", enabled=False)
    checkpoint_path = tmp_path / "safe_checkpoint.pt"

    save_model_checkpoint(
        paths=[checkpoint_path],
        epoch=2,
        best_fitness=0.75,
        model=model,
        ema_updates=4,
        optimizer=optimizer,
        scaler=scaler,
        train_config=config,
        train_metrics={"metrics/fitness": np.float32(0.5)},
    )

    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint = load_model_checkpoint(checkpoint_path)

    assert isinstance(raw_checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"], float)
    assert isinstance(raw_checkpoint["train_metrics"]["metrics/fitness"], float)
    assert checkpoint["epoch"] == 2
    assert checkpoint["train_config"].experiment_name == config.experiment_name


def test_resume_preserves_typed_patch_size(
    minimal_train_config: Path,
    tmp_path: Path,
) -> None:
    config = get_cfg(minimal_train_config, {})
    run_root = tmp_path / "runs"
    experiment_name = "resume_e2e"
    model = build_model_from_config(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.GradScaler("cpu", enabled=False)
    checkpoint_path = run_root / experiment_name / "weights_last.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    save_model_checkpoint(
        paths=[checkpoint_path],
        epoch=1,
        best_fitness=0.25,
        model=model,
        ema_updates=1,
        optimizer=optimizer,
        scaler=scaler,
        train_config=config,
        train_metrics={"metrics/fitness": 0.25},
    )

    resume_config_path = tmp_path / "resume_train.yaml"
    with minimal_train_config.open(encoding="utf-8") as file:
        resume_config = yaml.safe_load(file)
    resume_config["output_folder_base"] = run_root.as_posix()
    resume_config["experiment_name"] = experiment_name
    resume_config["resume"] = True
    with resume_config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(resume_config, file, sort_keys=False)

    trainer = Trainer(cfg=resume_config_path)
    checkpoint, _ = trainer.setup_model(num_classes=config.num_classes)

    assert checkpoint is not None
    assert isinstance(trainer.config.patch_size, PatchSize)
    assert trainer.config.patch_size.get_zyx() == config.patch_size.get_zyx()


def test_save_model_copies_best_validation_artifacts(
    minimal_train_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.config = get_cfg(minimal_train_config, {})
    trainer.save_period = -1
    trainer.epoch = 0
    trainer.save_dir = Path(trainer.config.output_folder_base) / trainer.config.experiment_name
    trainer.save_dir.mkdir(parents=True, exist_ok=True)
    trainer.last_model_path = trainer.save_dir / "weights_last.pt"
    trainer.best_model_path = trainer.save_dir / "weights_best.pt"
    trainer.ewa_fitness = 0.42

    validation_summary = trainer.save_dir / "validation_summary.png"
    validation_summary.write_bytes(b"summary")
    validation_image_labels = trainer.save_dir / "validation_image_labels.png"
    validation_image_labels.write_bytes(b"image-label-summary")
    validation_samples = trainer.save_dir / "validation_samples"
    validation_samples.mkdir(exist_ok=True)
    (validation_samples / "sample.txt").write_text("best sample", encoding="utf-8")

    saved_paths: list[list[Path]] = []

    def fake_save_model_checkpoint(
        paths: list[Path],
        epoch: int,
        best_fitness: float,
        model: torch.nn.Module,
        ema_updates: int,
        optimizer: torch.optim.Optimizer,
        scaler: torch.GradScaler,
        train_config: DefaultConfig,
        train_metrics: dict[str, float],
    ) -> None:
        saved_paths.append(paths)

    monkeypatch.setattr("radyolo.training.train.save_model_checkpoint", fake_save_model_checkpoint)

    model = build_model_from_config(trainer.config)
    ema = ModelEMA(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.GradScaler("cpu", enabled=False)

    trainer.save_model(
        best_fitness=0.42,
        ema=ema,
        optimizer=optimizer,
        scaler=scaler,
        metrics={"metrics/mAP10-95": 0.3},
        fitness=0.42,
    )

    assert saved_paths == [[trainer.last_model_path, trainer.best_model_path]]
    assert (trainer.save_dir / "validation_summary_best.png").read_bytes() == b"summary"
    assert (trainer.save_dir / "validation_image_labels_best.png").read_bytes() == b"image-label-summary"
    assert (trainer.save_dir / "validation_samples_best" / "sample.txt").read_text(
        encoding="utf-8"
    ) == "best sample"


def test_resume_preserves_local_dataset_override_when_checkpoint_path_is_missing(
    minimal_train_config: Path,
    tmp_path: Path,
) -> None:
    config = get_cfg(minimal_train_config, {})
    checkpoint_config = config.model_copy(
        update={"data": Path("/Projects/deep_learning/input/missing_cluster_dataset")},
        deep=True,
    )
    run_root = tmp_path / "runs"
    experiment_name = "resume_local_dataset_override"
    model = build_model_from_config(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.GradScaler("cpu", enabled=False)
    checkpoint_path = run_root / experiment_name / "weights_last.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    save_model_checkpoint(
        paths=[checkpoint_path],
        epoch=1,
        best_fitness=0.25,
        model=model,
        ema_updates=1,
        optimizer=optimizer,
        scaler=scaler,
        train_config=checkpoint_config,
        train_metrics={"metrics/fitness": 0.25},
    )

    resume_config_path = tmp_path / "resume_train_local_dataset.yaml"
    with minimal_train_config.open(encoding="utf-8") as file:
        resume_config = yaml.safe_load(file)
    resume_config["output_folder_base"] = run_root.as_posix()
    resume_config["experiment_name"] = experiment_name
    resume_config["resume"] = True
    with resume_config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(resume_config, file, sort_keys=False)

    trainer = Trainer(cfg=resume_config_path)
    checkpoint, _ = trainer.setup_model(num_classes=config.num_classes)

    assert checkpoint is not None
    assert trainer.config.data == config.data


def test_default_config_uses_renamed_voxel_size_field(minimal_train_config: Path) -> None:
    config = get_cfg(minimal_train_config, {"voxel_size": {"x": 1.5, "y": 1.2, "z": 2.5}})

    assert config.voxel_size == VoxelSize(x=1.5, y=1.2, z=2.5)
    assert "voxel_size" in config.model_dump()
    assert "target_voxel_size" not in config.model_dump()


def test_default_config_uses_epoch_based_ema_tau_default(minimal_train_config: Path) -> None:
    config = get_cfg(minimal_train_config, {})

    assert config.ema_tau == 400.0


def test_get_cfg_infers_num_classes_from_dataset_names(minimal_train_config: Path) -> None:
    with minimal_train_config.open(encoding="utf-8") as file:
        config_data = yaml.safe_load(file)
    config_data.pop("num_classes", None)
    config_path = minimal_train_config.parent / "inferred_num_classes.yaml"
    config_path.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")

    config = get_cfg(config_path, {})

    assert config.num_classes == 1


def test_get_cfg_infers_input_channels_from_dataset_normalization(
    synthetic_dataset: Path,
    tmp_path: Path,
) -> None:
    dataset_yaml_path = synthetic_dataset / "dataset.yaml"
    dataset_yaml = yaml.safe_load(dataset_yaml_path.read_text(encoding="utf-8"))
    dataset_yaml["normalization"] = {"0": "MRPerChannel", "1": "CT"}
    dataset_yaml_path.write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")

    config_path = tmp_path / "inferred_input_channels.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_scale: n",
                f"data: {synthetic_dataset.as_posix()}",
                "fold: 0",
                "epochs: 2",
                "nominal_batch_size: 1",
                "warmup_epochs: 0.5",
                "patience: 1",
                "batch_size: 1",
                "patch_size: {x: 32, y: 32, z: 64}",
                "normalization: {type: 'none'}",
                "device: cpu",
                "workers: 0",
                f"output_folder_base: {(tmp_path / 'runs').as_posix()}",
                "experiment_name: e2e",
                "exist_ok: True",
                "optimizer: Adam",
                "seed: 0",
                "resume: False",
                "amp: False",
                "use_cudnn_benchmark: False",
                "val: True",
                "plots: False",
                "augment: False",
            ]
        ),
        encoding="utf-8",
    )

    config = get_cfg(config_path, {})

    assert config.input_channels == 2


def test_get_cfg_warns_when_input_channels_is_set_in_config(
    synthetic_dataset: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    dataset_yaml_path = synthetic_dataset / "dataset.yaml"
    dataset_yaml = yaml.safe_load(dataset_yaml_path.read_text(encoding="utf-8"))
    dataset_yaml["normalization"] = {"0": "MRPerChannel", "1": "CT"}
    dataset_yaml_path.write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")

    config_path = tmp_path / "deprecated_input_channels.yaml"
    config_path.write_text(
        "\n".join(
            [
                "input_channels: 1",
                "model_scale: n",
                f"data: {synthetic_dataset.as_posix()}",
                "fold: 0",
                "epochs: 2",
                "nominal_batch_size: 1",
                "warmup_epochs: 0.5",
                "patience: 1",
                "batch_size: 1",
                "patch_size: {x: 32, y: 32, z: 64}",
                "normalization: {type: 'none'}",
                "device: cpu",
                "workers: 0",
                f"output_folder_base: {(tmp_path / 'runs').as_posix()}",
                "experiment_name: e2e",
                "exist_ok: True",
                "optimizer: Adam",
                "seed: 0",
                "resume: False",
                "amp: False",
                "use_cudnn_benchmark: False",
                "val: True",
                "plots: False",
                "augment: False",
            ]
        ),
        encoding="utf-8",
    )

    with caplog.at_level("WARNING"):
        config = get_cfg(config_path, {})

    assert config.input_channels == 1
    assert "Config field 'input_channels' is deprecated" in caplog.text


def test_get_cfg_warns_and_uses_legacy_input_channels_when_dataset_lacks_normalization(
    synthetic_dataset: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    dataset_yaml_path = synthetic_dataset / "dataset.yaml"
    dataset_yaml = yaml.safe_load(dataset_yaml_path.read_text(encoding="utf-8"))
    dataset_yaml.pop("normalization", None)
    dataset_yaml_path.write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")

    config_path = tmp_path / "legacy_input_channels_missing_dataset_normalization.yaml"
    config_path.write_text(
        "\n".join(
            [
                "input_channels: 3",
                "model_scale: n",
                f"data: {synthetic_dataset.as_posix()}",
                "fold: 0",
                "epochs: 2",
                "nominal_batch_size: 1",
                "warmup_epochs: 0.5",
                "patience: 1",
                "batch_size: 1",
                "patch_size: {x: 32, y: 32, z: 64}",
                "normalization: {type: 'none'}",
                "device: cpu",
                "workers: 0",
                f"output_folder_base: {(tmp_path / 'runs').as_posix()}",
                "experiment_name: e2e",
                "exist_ok: True",
                "optimizer: Adam",
                "seed: 0",
                "resume: False",
                "amp: False",
                "use_cudnn_benchmark: False",
                "val: True",
                "plots: False",
                "augment: False",
            ]
        ),
        encoding="utf-8",
    )

    with caplog.at_level("WARNING"):
        config = get_cfg(config_path, {})

    assert config.input_channels == 3
    assert "dataset.yaml does not define a per-channel normalization mapping" in caplog.text


def test_get_cfg_warns_when_num_classes_is_set_in_config(
    synthetic_dataset: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config_path = tmp_path / "deprecated_num_classes.yaml"
    config_path.write_text(
        "\n".join(
            [
                "input_channels: 1",
                "num_classes: 99",
                "model_scale: n",
                f"data: {synthetic_dataset.as_posix()}",
                "fold: 0",
                "epochs: 2",
                "nominal_batch_size: 1",
                "warmup_epochs: 0.5",
                "patience: 1",
                "batch_size: 1",
                "patch_size: {x: 32, y: 32, z: 64}",
                "normalization: {type: 'none'}",
                "device: cpu",
                "workers: 0",
                f"output_folder_base: {(tmp_path / 'runs').as_posix()}",
                "experiment_name: e2e",
                "exist_ok: True",
                "optimizer: Adam",
                "seed: 0",
                "resume: False",
                "amp: False",
                "use_cudnn_benchmark: False",
                "val: True",
                "plots: False",
                "augment: False",
            ]
        ),
        encoding="utf-8",
    )

    with caplog.at_level("WARNING"):
        config = get_cfg(config_path, {})

    assert config.num_classes == 1
    assert "Config field 'num_classes' is deprecated and ignored" in caplog.text


@pytest.mark.parametrize(
    ("deprecated_key", "deprecated_value", "expected_field", "expected_value", "expected_message"),
    [
        (
            "nms_time_limit",
            "0.5",
            "nms_iou_threshold",
            0.2,
            "Config field 'nms_time_limit' is deprecated and ignored",
        ),
        (
            "deterministic",
            "True",
            "seed",
            0,
            "Config field 'deterministic' is deprecated and ignored",
        ),
        (
            "verbose",
            "False",
            "verbose",
            True,
            "Config field 'verbose' is deprecated and ignored; use logging configuration to control verbosity.",
        ),
    ],
)
def test_get_cfg_warns_when_deprecated_config_is_set(
    synthetic_dataset: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    deprecated_key: str,
    deprecated_value: str,
    expected_field: str,
    expected_value: float | int | bool,
    expected_message: str,
) -> None:
    config_path = tmp_path / f"deprecated_{deprecated_key}.yaml"
    config_path.write_text(
        "\n".join(
            [
                "input_channels: 1",
                "model_scale: n",
                f"data: {synthetic_dataset.as_posix()}",
                "fold: 0",
                "epochs: 2",
                "nominal_batch_size: 1",
                "warmup_epochs: 0.5",
                "patience: 1",
                "batch_size: 1",
                "patch_size: {x: 32, y: 32, z: 64}",
                "normalization: {type: 'none'}",
                "device: cpu",
                "workers: 0",
                f"output_folder_base: {(tmp_path / 'runs').as_posix()}",
                "experiment_name: e2e",
                "exist_ok: True",
                "optimizer: Adam",
                "seed: 0",
                "resume: False",
                "amp: False",
                "use_cudnn_benchmark: False",
                "val: True",
                "plots: False",
                "augment: False",
                f"{deprecated_key}: {deprecated_value}",
            ]
        ),
        encoding="utf-8",
    )

    with caplog.at_level("WARNING"):
        config = get_cfg(config_path, {})

    assert getattr(config, expected_field) == expected_value
    assert expected_message in caplog.text


def test_get_cfg_rejects_relative_dataset_path(minimal_train_config: Path, tmp_path: Path) -> None:
    with minimal_train_config.open(encoding="utf-8") as file:
        config_data = yaml.safe_load(file)
    config_data["data"] = "relative/dataset"
    config_path = tmp_path / "relative_dataset_path.yaml"
    config_path.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Configured dataset folder does not exist"):
        get_cfg(config_path, {})


def test_validation_fitness_uses_configured_weights(minimal_train_config: Path) -> None:
    trainer = Trainer(cfg=minimal_train_config)
    trainer.config = trainer.config.model_copy(update={"fitness_metric_weights": (0.5, 0.25, 0.25)})

    fitness = trainer._compute_validation_fitness(
        {
            "metrics/mAP10-95": 0.6,
            "metrics/mAP10-95Agn": 0.2,
            "metrics/dice": 0.4,
        }
    )

    assert fitness == pytest.approx(0.45)


def test_validation_fitness_uses_image_label_metric_weights(minimal_train_config: Path) -> None:
    trainer = Trainer(cfg=minimal_train_config)
    trainer.config = trainer.config.model_copy(
        update={"fitness_metric_weights": (0.4, 0.1, 0.1, 0.1, 0.2, 0.1)}
    )

    fitness = trainer._compute_validation_fitness(
        {
            "metrics/mAP10-95": 0.5,
            "metrics/mAP10-95Agn": 0.2,
            "metrics/dice": 0.4,
            "metrics/image_label_balanced_accuracy": 0.8,
            "metrics/image_label_multiclass_roc_auc": 0.7,
            "metrics/image_label_c_index": 0.6,
        }
    )

    assert fitness == pytest.approx(0.54)


def test_default_config_defaults_tal_topk_schedule_to_static_value(minimal_train_config: Path) -> None:
    config = get_cfg(minimal_train_config, {})

    assert config.tal_topk_final == config.tal_topk
    assert config.tal_topk_final_epoch == config.epochs


def test_default_config_migrates_legacy_prototype_booleans() -> None:
    config = DefaultConfig.model_validate(
        {
            "data": Path("/tmp/dataset"),
            "fold": 0,
            "experiment_name": "legacy_prototype_layer",
            "input_channels": 1,
            "num_classes": 1,
            "model_scale": "n",
            "patch_size": {"x": 32, "y": 32, "z": 64},
            "normalization": {"type": "none"},
            "full_resolution_prototypes": False,
            "half_resolution_prototypes": True,
            "higher_resolution_prototypes": False,
        }
    )

    assert config.prototype_layer == "half_resolution"
