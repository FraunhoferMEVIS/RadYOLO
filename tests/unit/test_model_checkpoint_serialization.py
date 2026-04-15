from __future__ import annotations

import datetime
import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
import yaml

from radyolo.config.config import DefaultConfig, get_cfg, load_model_checkpoint, save_model_checkpoint
from radyolo.config.spatial import PatchSize, VoxelSize
from radyolo.models.model import build_model_from_config
from radyolo.training.train import Trainer

if TYPE_CHECKING:
    from pathlib import Path


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
        "train_metrics": {"metrics/mAP-ewa": np.float32(0.25)},
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
        train_metrics={"metrics/mAP-ewa": np.float32(0.5)},
    )

    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint = load_model_checkpoint(checkpoint_path)

    assert isinstance(raw_checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"], float)
    assert isinstance(raw_checkpoint["train_metrics"]["metrics/mAP-ewa"], float)
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
    checkpoint_path = run_root / experiment_name / "weights" / "last.pt"
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
        train_metrics={"metrics/mAP-ewa": 0.25},
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


def test_default_config_uses_renamed_voxel_size_field(minimal_train_config: Path) -> None:
    config = get_cfg(minimal_train_config, {"voxel_size": {"x": 1.5, "y": 1.2, "z": 2.5}})

    assert config.voxel_size == VoxelSize(x=1.5, y=1.2, z=2.5)
    assert "voxel_size" in config.model_dump()
    assert "target_voxel_size" not in config.model_dump()
