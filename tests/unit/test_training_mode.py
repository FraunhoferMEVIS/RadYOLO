from __future__ import annotations

from pathlib import Path

import yaml

from radyolo.data.data import PatchWiseDataset, WholeImageDataset
from radyolo.models.model import build_model_from_config
from radyolo.training.train import Trainer


def _write_config_with_updates(
    minimal_train_config: Path,
    tmp_path: Path,
    **updates: object,
) -> Path:
    config_path = tmp_path / "train_mode.yaml"
    with minimal_train_config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config.update(updates)
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    return config_path


def test_trainer_uses_patchwise_dataset_for_training_and_validation(
    minimal_train_config: Path,
    tmp_path: Path,
) -> None:
    config_path = _write_config_with_updates(
        minimal_train_config,
        tmp_path,
        training_mode="patchwise",
        foreground_oversampling=0.25,
    )
    trainer = Trainer(cfg=config_path)
    model = build_model_from_config(trainer.config, num_classes=trainer.data.nc)

    train_loader = trainer.get_dataloader(trainer.config.data, model, fold=0, mode="train")
    val_loader = trainer.get_dataloader(trainer.config.data, model, fold=0, mode="val")

    assert isinstance(train_loader.dataset, PatchWiseDataset)
    assert isinstance(val_loader.dataset, PatchWiseDataset)
    assert train_loader.dataset.foreground_oversampling == 0.25
    assert val_loader.dataset.foreground_oversampling == 0.25
    assert train_loader.dataset.deterministic_sampling is False
    assert val_loader.dataset.deterministic_sampling is True
    assert train_loader.dataset.min_box_crop_fraction == 0.5
    assert val_loader.dataset.min_box_crop_fraction == 0.5


def test_trainer_keeps_wholeimage_validation_without_patch_oversampling(
    minimal_train_config: Path,
    tmp_path: Path,
) -> None:
    config_path = _write_config_with_updates(
        minimal_train_config,
        tmp_path,
        training_mode="wholeimage",
        foreground_oversampling=0.25,
    )
    trainer = Trainer(cfg=config_path)
    model = build_model_from_config(trainer.config, num_classes=trainer.data.nc)

    train_loader = trainer.get_dataloader(trainer.config.data, model, fold=0, mode="train")
    val_loader = trainer.get_dataloader(trainer.config.data, model, fold=0, mode="val")

    assert isinstance(train_loader.dataset, WholeImageDataset)
    assert isinstance(val_loader.dataset, WholeImageDataset)
    assert train_loader.dataset.foreground_oversampling == 0.25
    assert val_loader.dataset.foreground_oversampling == 0.0
    assert train_loader.dataset.min_box_crop_fraction == 0.5
    assert val_loader.dataset.min_box_crop_fraction == 0.5
