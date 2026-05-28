from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

from radyolo.config.config import (
    DatasetConfig,
    get_cfg,
    load_model_checkpoint,
    save_model_checkpoint,
)
from radyolo.config.spatial import PatchSize, VoxelSize
from radyolo.data.analyze_dataset import analyze_dataset as analyze_dataset_fn
from radyolo.data.data import PatchWiseDataset, WholeImageDataset, build_dataloader
from radyolo.data.data_splits import k_fold_data_splits
from radyolo.data.preprocess_data import preprocess_dataset as preprocess_dataset_fn
from radyolo.export.onnx_export import export_model
from radyolo.inference.predict import PredictionConfig, run_prediction
from radyolo.models.model import build_model_from_config, load_checkpoint_model
from radyolo.utils.util import ModelEMA
from radyolo.validation.validate import Validator


def _write_nifti(path: Path, array: np.ndarray) -> None:
    img = nib.Nifti1Image(array.astype(np.float32), affine=np.eye(4))
    nib.save(img, str(path))


def _write_mask(path: Path, array: np.ndarray) -> None:
    img = nib.Nifti1Image(array.astype(np.uint8), affine=np.eye(4))
    nib.save(img, str(path))


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a small synthetic 3D dataset with a single class and predefined splits.

    Layout:
      dataset/
        dataset.yaml
        data_splits.json (single fold)
        images/*.nii.gz
        labels/*.txt
        masks/*.nii.gz
    """
    root = tmp_path_factory.mktemp("synthetic_dataset")
    for d in ("images", "labels", "masks"):
        (root / d).mkdir()

    vol_shape = (32, 32, 64)
    cases = ["case1", "case2", "case3", "case4", "case5"]
    for case in cases:
        vol = np.zeros(vol_shape, dtype=np.float32)
        vol[8:24, 8:24, 16:48] = 1.0
        _write_nifti(root / "images" / f"{case}.nii.gz", vol)
        (root / "labels" / f"{case}.txt").write_text(
            "0 0.5 0.5 0.5 0.5 0.5 0.5\n", encoding="utf-8"
        )
        _write_mask(root / "masks" / f"{case}.nii.gz", (vol > 0).astype(np.uint8))

    (root / "dataset.yaml").write_text(
        "nc: 1\nnames: {0: lesion}\n", encoding="utf-8"
    )
    splits = [{"train": cases[:3], "val": [cases[3]], "test": [cases[4]]}]
    (root / "data_splits.json").write_text(json.dumps(splits), encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def minimal_train_config(synthetic_dataset: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a minimal training config file for fast CPU test.

    Only essential fields are specified; all others rely on defaults.
    """
    cfg_dir = tmp_path_factory.mktemp("configs")
    cfg_path = cfg_dir / "train.yaml"
    output_folder_base = tmp_path_factory.mktemp("runs_root") / "runs"
    output_folder_base.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        (
            f"input_channels: 1\n"
            f"model_scale: n\n"
            f"data: {synthetic_dataset.as_posix()}\n"
            f"fold: 0\n"
            f"epochs: 1\n"
            f"nominal_batch_size: 1\n"
            f"warmup_epochs: 0.0\n"
            f"patience: 1\n"
            f"batch_size: 1\n"
            f"patch_size: {{x: 32, y: 32, z: 64}}\n"
            f"normalization: {{type: 'none'}}\n"
            f"device: cpu\n"
            f"workers: 0\n"
            f"output_folder_base: {output_folder_base.as_posix()}\n"
            f"experiment_name: e2e\n"
            f"exist_ok: True\n"
            f"optimizer: Adam\n"
            f"seed: 0\n"
            f"resume: False\n"
            f"amp: False\n"
            f"use_cudnn_benchmark: False\n"
            f"val: True\n"
            f"plots: False\n"
            f"augment: False\n"
        ),
        encoding="utf-8",
    )
    return cfg_path


@pytest.fixture(scope="session")
def trained_run(minimal_train_config: Path) -> Path:
    """Create a minimal checkpoint run and return the run directory.

    Fails fast if checkpoint creation does not produce both weights_last.pt and weights_best.pt.
    """
    cfg = minimal_train_config
    train_config = get_cfg(cfg, {})
    model = build_model_from_config(train_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.GradScaler("cpu", enabled=False)
    with cfg.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    run_dir = Path(data["output_folder_base"]) / str(data["experiment_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    save_model_checkpoint(
        paths=[run_dir / "weights_last.pt", run_dir / "weights_best.pt"],
        epoch=1,
        best_fitness=0.0,
        model=model,
        ema_updates=0,
        optimizer=optimizer,
        scaler=scaler,
        train_config=train_config,
        train_metrics={"metrics/fitness": 0.0},
    )
    if not (run_dir / "weights_last.pt").exists():
        msg_last = "weights_last.pt missing after checkpoint creation"
        raise FileNotFoundError(msg_last)
    if not (run_dir / "weights_best.pt").exists():
        msg_best = "weights_best.pt missing after checkpoint creation"
        raise FileNotFoundError(msg_best)
    return run_dir


# Helper functions for tests to call CLI-accessed functionality

def create_data_splits(dataset_path: Path, k: int = 5, seed: int = 0) -> Path:
    return k_fold_data_splits(
        dataset_path=dataset_path,
        seed=seed,
        k=k,
        group_by_prefix=False,
        prefix_sep="_",
    )


def analyze_dataset(dataset_path: Path, workers: int = 0) -> Path:
    return analyze_dataset_fn(dataset_path=dataset_path, workers=workers)


def preprocess_data(dataset_path: Path, output_path: Path, workers: int = 0) -> Path:
    return preprocess_dataset_fn(
        dataset_path=dataset_path,
        output_path=output_path,
        workers=workers,
        overwrite=False,
    )


def preprocess_data_with_options(
    dataset_path: Path,
    output_path: Path,
    workers: int = 0,
    *,
    image_extent: tuple[int, int, int] | None = None,
    voxel_size: tuple[float, float, float] | None = None,
    normalization: dict | list[dict] | None = None,
    image_format: str = "npy",
    image_dtype: str = "float32",
) -> Path:
    return preprocess_dataset_fn(
        dataset_path=dataset_path,
        output_path=output_path,
        workers=workers,
        overwrite=False,
        target_image_size=PatchSize.from_xyz(image_extent) if image_extent is not None else None,
        target_voxel_size=VoxelSize.from_xyz(voxel_size) if voxel_size is not None else None,
        normalization=cast("dict | list[dict]", normalization or {"type": "none"}),
        image_format=cast("str", image_format),
        image_dtype=cast("str", image_dtype),
    )


def export_onnx(last_model_path: Path) -> Path:
    checkpoint = load_model_checkpoint(last_model_path)
    train_config = checkpoint["train_config"]
    model = build_model_from_config(
        train_config,
        state_dict=checkpoint["model_state_dict"],
    )
    model.eval()
    input_shape = (train_config.input_channels, *train_config.patch_size.get_zyx())
    export_model(model, input_shape, last_model_path, internal_dtype="float32")
    return last_model_path.with_suffix(".onnx")


def run_validation(best_model_path: Path, group: str) -> Path:
    loaded_model = load_checkpoint_model(best_model_path)
    train_config = loaded_model.train_config
    device = loaded_model.device
    model = loaded_model.model
    dataset_root = Path(train_config.data)
    dataset_yaml = dataset_root / "dataset.yaml"
    with dataset_yaml.open(encoding="utf-8") as f:
        yaml_config = yaml.safe_load(f)
    dataset_config = DatasetConfig.model_validate(yaml_config)
    dataset_class = (
        PatchWiseDataset if train_config.training_mode == "patchwise" else WholeImageDataset
    )
    foreground_oversampling = (
        train_config.foreground_oversampling if train_config.training_mode == "patchwise" else 0.0
    )
    dataset = dataset_class(
        src_path=dataset_root,
        fold=train_config.fold,
        group=group,
        patch_size=train_config.patch_size,
        augment=False,
        min_stride=int(model.stride.min().item()) if hasattr(model, "stride") else 8,
        normalization=train_config.normalization,
        apply_preprocessing=train_config.apply_preprocessing,
        load_masks=train_config.segmentation_output,
        foreground_oversampling=foreground_oversampling,
        min_box_crop_fraction=train_config.min_box_crop_fraction,
    )
    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=1,
        workers=0,
        shuffle=False,
        device=device,
    )
    save_dir = best_model_path.parents[1] / "validation" / group
    validator = Validator(
        dataloader=dataloader,
        save_dir=save_dir,
        config=train_config,
        names=dataset_config.names or {0: "class_0"},
        save_predictions=True,
    )
    validator(trainer=None, model=model)
    return save_dir


def run_prediction_dir(model_path: Path, input_dir: Path, output_dir: Path) -> Path:
    result = run_prediction(
        PredictionConfig(
            model_path=model_path,
            input_dir=input_dir,
            output_dir=output_dir,
            confidence_threshold=0.1,
            nms_iou_threshold=0.2,
            device="cpu",
            clip_boxes_to_image=True,
            overwrite_patch_size=None,
            overwrite_normalization=None,
            num_load_workers=1,
            num_write_workers=1,
        )
    )
    if result != 0:
        msg = f"Prediction failed for model {model_path} on {input_dir}"
        raise RuntimeError(msg)
    return output_dir
