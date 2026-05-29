from __future__ import annotations

from pathlib import Path
import shutil

import torch
import yaml

from radyolo.config.config import get_cfg
from radyolo.config.spatial import PatchSize
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


def _make_trainer_for_dataloader(config_path: Path) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.config = get_cfg(config_path, {})
    trainer.distributed = False
    trainer.world_size = 1
    trainer.rank = 0
    trainer.device = torch.device("cpu")
    return trainer


def test_trainer_uses_patchwise_dataset_for_training_and_validation(
    minimal_train_config: Path,
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "patchwise_background_sampling_dataset"
    shutil.copytree(Path(yaml.safe_load(minimal_train_config.read_text(encoding="utf-8"))["data"]), dataset_root)
    for case_name in ("case2", "case3"):
        (dataset_root / "labels" / f"{case_name}.txt").write_text("", encoding="utf-8")

    config_path = _write_config_with_updates(
        minimal_train_config,
        tmp_path,
        data=dataset_root.as_posix(),
        training_mode="patchwise",
        foreground_oversampling=0.25,
        background_undersampling=0.5,
    )
    trainer = _make_trainer_for_dataloader(config_path)
    model = build_model_from_config(trainer.config, num_classes=1)

    train_loader = trainer.get_dataloader(trainer.config.data, model, fold=0, mode="train")
    val_loader = trainer.get_dataloader(trainer.config.data, model, fold=0, mode="val")

    assert isinstance(train_loader.dataset, PatchWiseDataset)
    assert isinstance(val_loader.dataset, PatchWiseDataset)
    assert train_loader.dataset.foreground_oversampling == 0.25
    assert val_loader.dataset.foreground_oversampling == 0.25
    assert train_loader.dataset.background_undersampling == 0.5
    assert val_loader.dataset.background_undersampling == 0.0
    assert train_loader.dataset.deterministic_sampling is False
    assert val_loader.dataset.deterministic_sampling is True
    assert len(train_loader.sampler) == 3
    assert len(val_loader.sampler) == len(val_loader.dataset)
    assert train_loader.dataset.extraction_patch_size == PatchSize(
        x=39,
        y=39,
        z=77,
    )
    assert val_loader.dataset.extraction_patch_size == trainer.config.patch_size
    assert train_loader.dataset.min_box_crop_fraction == 0.5
    assert val_loader.dataset.min_box_crop_fraction == 0.5


def test_trainer_keeps_wholeimage_validation_without_patch_oversampling(
    minimal_train_config: Path,
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "wholeimage_background_sampling_dataset"
    shutil.copytree(Path(yaml.safe_load(minimal_train_config.read_text(encoding="utf-8"))["data"]), dataset_root)
    for case_name in ("case2", "case3"):
        (dataset_root / "labels" / f"{case_name}.txt").write_text("", encoding="utf-8")

    config_path = _write_config_with_updates(
        minimal_train_config,
        tmp_path,
        data=dataset_root.as_posix(),
        training_mode="wholeimage",
        foreground_oversampling=0.25,
        background_undersampling=0.5,
    )
    trainer = _make_trainer_for_dataloader(config_path)
    model = build_model_from_config(trainer.config, num_classes=1)

    train_loader = trainer.get_dataloader(trainer.config.data, model, fold=0, mode="train")
    val_loader = trainer.get_dataloader(trainer.config.data, model, fold=0, mode="val")

    assert isinstance(train_loader.dataset, WholeImageDataset)
    assert isinstance(val_loader.dataset, WholeImageDataset)
    assert train_loader.dataset.foreground_oversampling == 0.25
    assert val_loader.dataset.foreground_oversampling == 0.0
    assert train_loader.dataset.background_undersampling == 0.5
    assert val_loader.dataset.background_undersampling == 0.0
    assert len(train_loader.sampler) == 3
    assert len(val_loader.sampler) == len(val_loader.dataset)
    assert train_loader.dataset.min_box_crop_fraction == 0.5
    assert val_loader.dataset.min_box_crop_fraction == 0.5


def test_wholeimage_dataset_can_apply_preprocessing(
    synthetic_dataset: Path,
) -> None:
    dataset = WholeImageDataset(
        src_path=synthetic_dataset,
        patch_size=PatchSize(x=16, y=16, z=32),
        augment=False,
        normalization={"type": "255_to_01"},
        apply_preprocessing=True,
    )

    sample = dataset.load_image_label_and_mask(0)

    assert sample["image"].shape == (1, 32, 16, 16)
    assert sample["shape"] == (32, 16, 16)
    assert sample["ratio_pad"] == ((0.5, 0.5, 0.5), (0, 0, 0))
    assert torch.isclose(sample["image"].max(), torch.tensor(1.0 / 255.0, dtype=torch.float32))


def test_patchwise_dataset_can_skip_normalization_for_preprocessed_inputs(
    synthetic_dataset: Path,
    monkeypatch,
) -> None:
    dataset = PatchWiseDataset(
        src_path=synthetic_dataset,
        patch_size=PatchSize(x=16, y=16, z=32),
        augment=False,
        normalization={"type": "255_to_01"},
        apply_preprocessing=False,
    )
    monkeypatch.setattr(dataset, "_sample_patch_center", lambda _shape, _patch_shape: (16, 16, 32))

    sample = dataset[0]

    assert sample["image"].shape == (1, 32, 16, 16)
    assert torch.isclose(sample["image"].max(), torch.tensor(1.0, dtype=torch.float32))
