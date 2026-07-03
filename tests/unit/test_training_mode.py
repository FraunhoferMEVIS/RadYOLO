from __future__ import annotations

from pathlib import Path
import shutil

import pytest
import torch
import yaml

from radyolo.config.config import get_training_config
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
    trainer.config = get_training_config(config_path, {})
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
    shutil.copytree(
        Path(yaml.safe_load(minimal_train_config.read_text(encoding="utf-8"))["data"]), dataset_root
    )
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


def test_trainer_save_metrics_warns_when_results_csv_is_busy(
    minimal_train_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    trainer = _make_trainer_for_dataloader(minimal_train_config)
    trainer.csv_path = tmp_path / "results.csv"
    trainer.epoch = 0

    original_open = Path.open

    def open_with_busy(self: Path, *args: object, **kwargs: object):
        if self == trainer.csv_path:
            raise OSError("Device or resource busy")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_with_busy)

    caplog.set_level("WARNING", logger="radyolo.training.train")
    trainer.save_metrics({"metrics/fitness": 0.5}, train_time_start=0.0)

    assert not trainer.csv_path.exists()
    assert "Could not save metrics CSV" in caplog.text


def test_trainer_keeps_wholeimage_validation_without_patch_oversampling(
    minimal_train_config: Path,
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "wholeimage_background_sampling_dataset"
    shutil.copytree(
        Path(yaml.safe_load(minimal_train_config.read_text(encoding="utf-8"))["data"]), dataset_root
    )
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


def test_wholeimage_dataset_can_defer_preprocessing_to_gpu(
    synthetic_dataset: Path,
) -> None:
    dataset = WholeImageDataset(
        src_path=synthetic_dataset,
        patch_size=PatchSize(x=16, y=16, z=32),
        augment=False,
        normalization={"type": "255_to_01"},
        apply_preprocessing=True,
        preprocessing_device="gpu",
    )

    sample = dataset.load_image_label_and_mask(0)

    assert sample["image"].shape == (64, 32, 32)
    assert sample["shape"] == (64, 32, 32)
    assert sample["ratio_pad"] == ((1.0, 1.0, 1.0), (0, 0, 0))
    assert torch.isclose(sample["image"].max(), torch.tensor(1.0, dtype=torch.float32))


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


def test_trainer_prepares_variable_shape_batch_with_gpu_preprocessing(
    minimal_train_config: Path,
    tmp_path: Path,
) -> None:
    config_path = _write_config_with_updates(
        minimal_train_config,
        tmp_path,
        preprocessing_device="gpu",
        normalization={"type": "255_to_01"},
        patch_size={"x": 16, "y": 16, "z": 32},
    )
    trainer = _make_trainer_for_dataloader(config_path)

    batch = {
        "img": (
            torch.ones((1, 64, 32, 32), dtype=torch.float32),
            torch.ones((1, 48, 24, 24), dtype=torch.float32),
        ),
        "ori_shape": ((64, 32, 32), (48, 24, 24)),
        "ratio_pad": (((1.0, 1.0, 1.0), (0, 0, 0)), ((1.0, 1.0, 1.0), (0, 0, 0))),
        "shape": ((64, 32, 32), (48, 24, 24)),
        "cls": torch.zeros((0, 1), dtype=torch.float32),
        "bboxes": torch.zeros((0, 6), dtype=torch.float32),
        "batch_idx": torch.zeros((0,), dtype=torch.int64),
        "masks": (None, None),
        "case_names": ("case1", "case2"),
        "image_labels": ({}, {}),
    }

    prepared = trainer._prepare_training_batch(batch, run_gpu_augmentations=False)

    assert isinstance(prepared["img"], torch.Tensor)
    assert prepared["img"].shape == (2, 1, 32, 16, 16)
    assert prepared["shape"] == ((32, 16, 16), (32, 16, 16))
    assert prepared["ratio_pad"] == (
        ((0.5, 0.5, 0.5), (0, 0, 0)),
        ((2 / 3, 2 / 3, 2 / 3), (0, 0, 0)),
    )
    assert torch.isclose(prepared["img"].max(), torch.tensor(1.0 / 255.0, dtype=torch.float32))


def test_wholeimage_gpu_preprocessing_rejects_cpu_augmentations(
    synthetic_dataset: Path,
    minimal_train_config: Path,
    tmp_path: Path,
) -> None:
    config = get_training_config(
        _write_config_with_updates(
            minimal_train_config,
            tmp_path,
            data=synthetic_dataset.as_posix(),
            preprocessing_device="gpu",
            zoom_augmentation_device="cpu",
        ),
        {},
    )

    with pytest.raises(ValueError, match="preprocessing_device='gpu'"):
        WholeImageDataset(
            src_path=synthetic_dataset,
            patch_size=PatchSize(x=16, y=16, z=32),
            augment=True,
            normalization={"type": "255_to_01"},
            apply_preprocessing=True,
            preprocessing_device="gpu",
            config=config,
        )
