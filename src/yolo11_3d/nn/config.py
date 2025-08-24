from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, TypedDict

import annotated_types
import torch
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    DirectoryPath,
    Field,
    FilePath,
    NewPath,
    NonNegativeInt,
    model_validator,
)

if TYPE_CHECKING:
    import datetime

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = Path("datasets")  # Path(SETTINGS["datasets_dir"])
# WEIGHTS_DIR = Path(SETTINGS["weights_dir"])
# RUNS_DIR = Path(SETTINGS["runs_dir"])

LOGGER = logging.getLogger(__name__)

DimList = Annotated[list[int], annotated_types.Len(min_length=2, max_length=3)]


class ModelConfig(BaseModel):
    nc: int
    scale: Literal["n", "s", "m", "l", "x"]


class DefaultConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    # Train settings
    model: FilePath | NewPath  # path to model file
    data: DirectoryPath  # path to data folder containing dataset.yaml
    fold: int
    epochs: int = 100  # number of epochs to train for
    patience: int = (
        100  # epochs to wait for no observable improvement for early stopping of training
    )
    batch: int = 16  # number of images per batch (-1 for AutoBatch)
    imgsz: DimList  # input images size as int for train and val modes, or list[h,w] for predict and export modes
    normalization: Literal["none", "zScore", "255_to_01"] = "255_to_01" # Normalization to be applied during image loading
    save: bool = True  # save train checkpoints and predict results
    save_period: int = -1  # Save checkpoint every x epochs (disabled if < 1)
    device: str = ""  # device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
    workers: int = 8  # number of worker threads for data loading (per RANK if DDP)
    project: str = ""  # project name
    name: int | str  # experiment name, results saved to 'project/name' directory
    exist_ok: bool = False  # whether to overwrite existing experiment
    optimizer: Literal["SGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"] = (
        "auto"  # optimizer to use
    )
    verbose: bool = True  # whether to print verbose output
    seed: int = 0  # random seed for reproducibility
    deterministic: bool = True  # whether to enable deterministic mode
    resume: bool | FilePath = False  # resume training from last checkpoint
    amp: bool = True  # Automatic Mixed Precision (AMP) training, True runs AMP check

    # Val/Test Settings
    val: bool = True  # validate/test during training
    save_json: bool = False  # save results to JSON file
    save_hybrid: bool = False  # save hybrid version of labels (labels+additional predictions)
    conf: float | None = (
        None  # object confidence threshold for detection (default 0.25 predict, 0.001 val)
    )
    iou: float = 0.2  # intersection over union (IoU) threshold for NMS
    max_det: int = 300  # maximum number of detections per image
    half: bool = False  # use half precision (FP16)
    plots: bool = True  # save plots and images during train/val

    # Predict Settings
    source: DirectoryPath | None = None  # source directory for images or videos
    augment: bool = False  # apply image augmentation to prediction sources
    agnostic_nms: bool = False  # class-agnostic NMS

    # Hyperparameters
    lr0: float = 0.01  # initial learning rate
    lrf: float = 0.01  # final learning rate (lr0*lrf)
    momentum: float = 0.937  # SGD momentum/Adam beta1
    weight_decay: float = 5e-4  # optimizer weight decay 5e-4
    warmup_epochs: float = 3.0  # warmup epochs (fractions ok)
    warmup_momentum: float = 0.8  # warmup initial momentum
    warmup_bias_lr: float = 0.1  # warmup initial bias lr
    box: float = 7.5  # box loss gain
    cls: float = 0.5  # cls loss gain (scale with pixels)
    dfl: float = 1.5  # dfl loss gain
    mask_gain: float = 1.0  # mask loss gain
    nbs: int = 64  # nominal batch size

    # Augmentation
    max_zoom: float = Field(1.5, ge=0)
    min_zoom: float = Field(0.7, ge=0)
    out_zoom_padding: Literal["constant", "random"] = "constant"
    prob_zoom: float = Field(0.5, ge=0, le=1.0)
    prob_cutout: float = Field(0.5, ge=0, le=1.0)
    cutout_params: list[tuple[float, int]] = Field(
        default_factory=lambda: [
            (0.5, 1),
            (0.25, 2),
            (0.125, 4),
            (0.0625, 8),
            (0.03125, 16),
        ]
    )
    prob_mirroring: float = Field(0.5, ge=0, le=1.0)
    mirror_axes: list[Literal["x", "y", "z"]] = Field(default_factory=lambda: ["z", "y", "x"])
    do_batchgenerators_intensity_augmentation: bool = True
    do_intensity_augmentations_per_channel: bool = False

    save_dir: Path | None = None


class DatasetConfig(BaseModel):
    names: dict[NonNegativeInt, str] | None = None
    nc: NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_classes(self) -> Self:
        if self.names is not None:
            if not self.nc:
                self.nc = len(self.names)

            if len(self.names) == self.nc:
                if max(self.names.keys()) == self.nc - 1:
                    return self

                msg = (
                    f"{self.nc}-class dataset requires class indices 0-{self.nc - 1}, but you have invalid class indices "
                    f"{min(self.names.keys())}-{max(self.names.keys())} defined in your dataset YAML."
                )
                raise KeyError(msg)

            msg = f"'names' length: {len(self.names)} and 'nc: {self.nc}' must match."
            raise ValueError(msg)

        if not self.nc:
            self.names = {i: f"class_{i}" for i in range(self.nc)}
            return self

        msg = "key missing.\n either 'names' or 'nc' are required in all data YAMLs."
        raise ValueError(msg)


class ModelCheckpoint(TypedDict):
    epoch: int
    best_fitness: float
    model: None
    ema: torch.nn.Module
    updates: int
    optimizer: dict | None
    scaler: dict | None
    train_args: DefaultConfig
    train_metrics: dict
    train_results: dict
    date: datetime.datetime


def model_checkpoint_from_torch(checkpoint: Path) -> ModelCheckpoint:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return ModelCheckpoint(
        epoch=ckpt["epoch"],
        best_fitness=ckpt["best_fitness"],
        model=ckpt["model"],
        ema=ckpt["ema"],
        updates=ckpt["updates"],
        optimizer=ckpt["optimizer"],
        scaler=ckpt["scaler"],
        train_args=DefaultConfig.model_validate(ckpt["train_args"]),
        train_metrics=ckpt["train_metrics"],
        train_results=ckpt["train_results"],
        date=ckpt["date"],
    )


def get_cfg(config_path: str | Path, overrides: dict[str, Any] | None) -> DefaultConfig:
    with Path(config_path).open() as f:
        yaml_config = yaml.safe_load(f)
    if overrides:
        if "save_dir" not in yaml_config:
            overrides.pop("save_dir", None)
        yaml_config = {**yaml_config, **overrides}

    if yaml_config.get("name") == "model":
        yaml_config["name"] = yaml_config.get("model", "").split(".")[0]
        LOGGER.warning("'name=model' automatically updated to 'name=%s'.", yaml_config["name"])

    return DefaultConfig.model_validate(yaml_config)


def get_dataset_cfg(config_path: str | Path) -> DatasetConfig:
    with Path(config_path).open() as f:
        yaml_config = yaml.safe_load(f)
    return DatasetConfig.model_validate(yaml_config)


def get_model_cfg(config_path: str | Path) -> ModelConfig:
    with Path(config_path).open() as f:
        yaml_config = yaml.safe_load(f)
    return ModelConfig.model_validate(yaml_config)
