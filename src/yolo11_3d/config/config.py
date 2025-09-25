from __future__ import annotations

import annotated_types
import logging
import pathlib
import sys
import torch
import yaml

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, TypedDict
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

LOGGER = logging.getLogger(__name__)

DimList = Annotated[list[int], annotated_types.Len(min_length=2, max_length=3)]
    

class DefaultConfig(BaseModel):
    input_channels: int
    model_scale: Literal["n", "s", "m", "l", "x"]

    # Train settings
    data: DirectoryPath  # path to data folder containing dataset.yaml
    fold: int
    epochs: int = 100  # number of epochs to train for
    patience: int = (
        100  # epochs to wait for no observable improvement for early stopping of training
    )
    batch_size: int = 2  # number of images per batch
    imgsz: DimList  # input images size as int for train and val modes, or list[h,w] for predict and export modes
    normalization: dict | list[dict] = {"type": "none"}  # Normalization for image intensities. Can be {type: "none"/"zscore"/"255_to_01"/"ct"} to be applied globally or a list of such dicts for each channel. For ct you have to define mean, std, clip_min and clip_max in the dict.
    save: bool = True  # save train checkpoints
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
    use_cudnn_benchmark: bool = True # Set torch.backends.cudnn.benchmark = True which tests several convolution algorithms and chooses the best one

    # Val/Test Settings
    val: bool = True  # validate/test during training
    conf: float | None =  0.001 # object confidence threshold for detection during validation
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
    objectness_loss: Literal["bce", "focal"] = "bce" # loss function for objectness loss
    box: float = 7.5  # box loss gain
    cls: float = 0.5  # cls loss gain (scale with pixels)
    dfl: float = 1.5  # dfl loss gain
    mask_gain: float = 1.0  # mask loss gain
    class_separate_gain: float = 0.5 # class loss gain for separate softmax classification loss
    nbs: int = 64  # nominal batch size

    # Optional Segmentation Head
    segmentation_output: bool = True # add segmentation output to model
    prototype_masks: int = 8 # number of prototype masks for segmentation output
    higher_resolution_prototypes: bool = True # (bool) use higher vs. lower resolution prototype masks (half vs. quarter input resolution)
    # Detection head options
    separate_objectness_head: bool = False  # if True, add a separate objectness output and use CE for class logits
    include_stride4_output: bool = False  # add P2 detection head at stride 4

    # Zoom Augmentation
    max_zoom: float = Field(1.5, ge=0)
    min_zoom: float = Field(0.7, ge=0)
    out_zoom_padding: Literal["constant", "random"] = "constant"
    prob_zoom: float = Field(0.5, ge=0, le=1.0)
    zoom_uniform: bool = False
    # Cutout Augmentation
    prob_cutout: float = Field(0.5, ge=0, le=1.0)
    cutout_params: list[tuple[float, int]] = Field(
        default_factory=lambda: [
            (0.5, 1),
            (0.25, 2),
            (0.125, 4),
        ]
    )
    prob_mirroring: float = Field(0.5, ge=0, le=1.0)
    mirror_axes: list[Literal["x", "y", "z"]] = Field(default_factory=lambda: ["z", "y", "x"])
    # Rotation
    prob_rotation: float = Field(0.0, ge=0, le=1.0)
    rotation_angle_x: tuple[float, float] = (-20.0, 20.0)
    rotation_angle_y: tuple[float, float] = (-20.0, 20.0)
    rotation_angle_z: tuple[float, float] = (-20.0, 20.0)
    rotation_keep_box_volume: bool = True
    # Channel Dropout
    channel_dropout_probabilities: tuple[float, ...] = Field(default_factory=tuple)
    channel_dropout_fill_value: float = 0.0
    channel_dropout_use_mean: bool = False
    # Intensity augmentation
    do_batchgenerators_intensity_augmentation: bool = True
    do_intensity_augmentations_per_channel: bool = False
    intensity_noise_prob: float = 0.1
    intensity_noise_variance: tuple[float, float] = (0.0, 1.0)
    intensity_blur_prob: float = 0.2
    intensity_blur_sigma: tuple[float, float] = (0.5, 1.0)
    intensity_mult_brightness_prob: float = 0.15
    intensity_mult_brightness_range: tuple[float, float] = (0.75, 1.25)
    intensity_add_brightness_prob: float = 0.15
    intensity_add_brightness_mu: float = 0.0
    intensity_add_brightness_sigma: float = 2.0
    intensity_contrast_prob: float = 0.15
    intensity_contrast_range: tuple[float, float] = (0.75, 1.25)
    intensity_lowres_prob: float = 0.25
    intensity_lowres_scale: tuple[float, float] = (0.5, 1.0)
    intensity_gamma_prob: float = 0.4
    intensity_gamma_range: tuple[float, float] = (0.7, 1.5)
    intensity_gamma_p_invert: float = 0.25
    intensity_gamma_p_retain_stats: float = 1.0
    # GIN augmentation
    gin_layers: int = 4
    gin_channels: int = 2
    gin_prob_kernel_size_one: float = 0.8
    gin_prob: float = 0.2

    save_dir: Path | None = None

    @model_validator(mode="after")
    def _validate_normalization(self) -> Self:
        normalization = self.normalization
        allowed = {"none", "zscore", "255_to_01", "ct"}
        
        def _validate_entry(entry: Any) -> dict:
            if not isinstance(entry, dict):
                raise ValueError("Invalid normalization; must be a dict with a 'type' field or list of such dicts")
            if "type" not in entry:
                raise ValueError("Each normalization entry must have a 'type' field")
            if entry["type"] not in allowed:
                raise ValueError(f"Unsupported normalization type '{entry["type"]}'. Allowed: {sorted(allowed)}")
            if entry["type"] == "ct" and not ("mean" in entry.keys() and "std" in entry.keys() and "clip_min" in entry.keys() and "clip_max" in entry.keys()):
                    raise ValueError(f"ct normalization needs to define type, mean, std, clip_min and clip_max, but only has keys {entry.keys()}")
            return entry
        
        if isinstance(normalization, list):
            if not normalization:
                raise ValueError("normalization list must not be empty")
            for entry in normalization:
                _validate_entry(entry) 
        elif isinstance(normalization, dict):
            _validate_entry(normalization)
        else:
            raise ValueError("normalization must be a dict or a list of dicts")
        return self


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
    if sys.platform == "win32":
        pathlib.PosixPath = pathlib.WindowsPath
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

