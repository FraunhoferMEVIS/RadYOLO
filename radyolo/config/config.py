from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, TypedDict

import torch
import yaml
from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    FilePath,
    NonNegativeInt,
    model_validator,
)

from radyolo.config.spatial import PatchSize  # noqa: TC001, needs to be imported for pydantic model

if TYPE_CHECKING:
    from radyolo.models.model import HardcodedModel

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectionPostprocessConfig:
    conf: float
    iou: float
    agnostic_nms: bool
    max_det: int
    nms_time_limit: float | None
    nc: int
    multi_label: bool = False


def _validate_normalization_entry(entry: dict) -> dict:
    allowed = {"none", "zscore", "255_to_01", "ct"}
    if not isinstance(entry, dict):
        msg = "Invalid normalization entry; must be dict with a 'type' field"
        raise TypeError(msg)
    if "type" not in entry:
        msg = "Each normalization entry must have a 'type' field"
        raise ValueError(msg)
    if entry["type"] not in allowed:
        msg = f"Unsupported normalization type '{entry['type']}'. Allowed: {sorted(allowed)}"
        raise ValueError(msg)
    if entry["type"] == "ct" and not (
        "mean" in entry and "std" in entry and "clip_min" in entry and "clip_max" in entry
    ):
        msg = (
            f"ct normalization needs to define type, mean, std, clip_min "
            f"and clip_max, but only has keys {entry.keys()}"
        )
        raise ValueError(msg)
    return entry


class DefaultConfig(BaseModel):
    input_channels: int
    num_classes: int
    model_scale: Literal["n", "s", "m", "l", "x"]

    # Train settings
    data: Path  # path to data folder containing dataset.yaml
    fold: int  # fold from data_splits.json to use
    epochs: int = 1000  # number of epochs to train for
    patience: int = (
        100  # epochs to wait for no observable improvement for early stopping of training
    )
    batch_size: int = 2  # number of images per batch
    patch_size: PatchSize  # input patch size as {"x": x, "y": y, "z": z}
    foreground_oversampling: float = Field(
        0.0,
        ge=0.0,
        lt=1.0,
    )  # fraction of samples per train epoch that should be extra foreground oversamples
    # Normalization for image intensities. Can be {type: "none"/"zscore"/"255_to_01"/"ct"} to be
    # applied globally or a list of such dicts for each channel. For ct you have to define
    # mean, std, clip_min and clip_max in the dict.
    normalization: dict | list[dict] = {"type": "none"}
    save: bool = True  # save train checkpoints
    save_period: int = -1  # Save checkpoint every x epochs (disabled if < 1)
    device: str = ""  # device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
    workers: int = 6  # number of worker threads for data loading (per RANK if DDP)
    output_folder_base: str = ""  # base output folder for experiment directories
    experiment_name: int | str  # experiment name, results saved to 'output_folder_base/experiment_name'
    exist_ok: bool = False  # whether to overwrite existing experiment
    optimizer: Literal["SGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"] = (
        "auto"  # optimizer to use
    )
    verbose: bool = True  # whether to print verbose output
    seed: int = 0  # random seed for reproducibility
    deterministic: bool = True  # whether to enable deterministic mode
    resume: bool | FilePath = False  # resume training from last checkpoint
    amp: bool = True  # Automatic Mixed Precision (AMP) training, True runs AMP check
    # Set torch.backends.cudnn.benchmark = True which tests several convolution algorithms and
    # chooses the best one
    use_cudnn_benchmark: bool = True

    # Val/Test Settings
    val: bool = True  # validate/test during training
    conf: float | None = 0.001  # object confidence threshold for detection during validation
    iou: float = 0.2  # intersection over union (IoU) threshold for NMS
    nms_time_limit: float = 0.5  # time limit per image in seconds before NMS stops processing
    max_det: int = 300  # maximum number of detections per image
    half: bool = False  # use half precision (FP16)
    plots: bool = True  # save plots and images during train/val
    clip_boxes_to_image: bool = True  # clip predicted 3D boxes to image bounds in val/predict
    validation_plot_projection: Literal["max", "mean"] = (
        "max"  # projection reduction used for validation case plots
    )
    validation_plot_image_mode: Literal["default", "channel_difference"] = (
        "default"  # source image used for validation case plots
    )
    validation_plot_difference_channels: tuple[NonNegativeInt, NonNegativeInt] = (
        2,
        0,
    )  # channels used for validation difference plots as (minuend, subtrahend)

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
    objectness_loss: Literal["bce", "focal"] = "bce"  # loss function for objectness loss
    tal_topk: int = 10  # top-k anchors to keep per gt in task-aligned assigner
    tal_alpha: float = 0.5  # classification term exponent in task-aligned assigner
    tal_beta: float = 1.0  # IoU term exponent in task-aligned assigner
    gamma: float = 2.0  # quality focal loss gamma (used when objectness_loss="focal")
    grad_clip_max_norm: float = 10.0  # max norm for gradient clipping
    ema_decay: float = 0.9999  # exponential moving average decay
    ema_tau: float = 2000.0  # EMA ramp-up factor
    box: float = 7.5  # box loss gain
    cls: float = 0.5  # cls loss gain (scale with pixels)
    dfl: float = 1.5  # dfl loss gain
    dfl_reg_max: int = Field(16, ge=1)  # use direct edge regression when set to 1
    bbox_parametrization: Literal["ltrb", "center_extents"] = (
        "ltrb"  # bbox prediction format for the detection head
    )
    bbox_center_offset_limit: float = Field(
        4.0, gt=0.5
    )  # max absolute anchor-relative center offset for center_extents bbox prediction
    mask_gain: float = 1.0  # mask loss gain
    class_separate_gain: float = 0.5  # class loss gain for separate softmax classification loss
    nbs: int = 64  # nominal batch size

    # Optional Segmentation Head
    segmentation_output: bool = True  # add segmentation output to model
    prototype_masks: int = 8  # number of prototype masks for segmentation output
    # use higher vs. lower resolution prototype masks (half vs. quarter input resolution)
    higher_resolution_prototypes: bool = True

    # Detection head options
    # add separate objectness output, use cross entropy loss for softmax class probabilities
    separate_objectness_head: bool = False
    include_stride4_output: bool = False  # add P2 detection head at stride 4

    # Zoom Augmentation
    max_zoom: float = Field(1.5, ge=0)
    min_zoom: float = Field(0.7, ge=0)
    out_zoom_padding: Literal["constant", "random"] = "constant"
    prob_zoom: float = Field(0.5, ge=0, le=1.0)
    zoom_uniform: bool = False
    zoom_augmentation_device: Literal["cpu", "gpu"] = "cpu"
    # Cutout Augmentation
    prob_cutout: float = Field(0.5, ge=0, le=1.0)
    cutout_params: list[tuple[float, int]] = Field(
        default_factory=lambda: [
            (0.5, 1),
            (0.25, 2),
            (0.125, 4),
        ]
    )
    cutout_augmentation_device: Literal["cpu", "gpu"] = "cpu"
    prob_mirroring: float = Field(0.5, ge=0, le=1.0)
    mirror_axes: list[Literal["x", "y", "z"]] = Field(default_factory=lambda: ["z", "y", "x"])
    mirroring_augmentation_device: Literal["cpu", "gpu"] = "cpu"
    # Rotation
    prob_rotation: float = Field(0.0, ge=0, le=1.0)
    rotation_angle_x: tuple[float, float] = (-20.0, 20.0)
    rotation_angle_y: tuple[float, float] = (-20.0, 20.0)
    rotation_angle_z: tuple[float, float] = (-20.0, 20.0)
    rotation_keep_box_volume: bool = True
    rotation_augmentation_device: Literal["cpu", "gpu"] = "cpu"
    # Channel Dropout
    channel_dropout_probabilities: tuple[float, ...] = Field(default_factory=tuple)
    channel_dropout_fill_value: float = 0.0
    channel_dropout_use_mean: bool = False
    channel_dropout_augmentation_device: Literal["cpu", "gpu"] = "cpu"
    # Intensity augmentation
    do_batchgenerators_intensity_augmentation: bool = True
    do_intensity_augmentations_per_channel: bool = False
    intensity_noise_prob: float = 0.1
    intensity_noise_variance: tuple[float, float] = (0.0, 0.1)
    intensity_noise_augmentation_device: Literal["cpu", "gpu"] = "cpu"
    intensity_blur_prob: float = 0.2
    intensity_blur_sigma: tuple[float, float] = (0.5, 1.0)
    intensity_blur_augmentation_device: Literal["cpu", "gpu"] = "cpu"
    intensity_mult_brightness_prob: float = 0.15
    intensity_mult_brightness_range: tuple[float, float] = (0.75, 1.25)
    intensity_mult_brightness_augmentation_device: Literal["cpu", "gpu"] = "cpu"
    intensity_add_brightness_prob: float = 0.15
    intensity_add_brightness_mu: float = 0.0
    intensity_add_brightness_sigma: float = 0.4
    intensity_add_brightness_augmentation_device: Literal["cpu", "gpu"] = "cpu"
    intensity_contrast_prob: float = 0.15
    intensity_contrast_range: tuple[float, float] = (0.75, 1.25)
    intensity_contrast_augmentation_device: Literal["cpu", "gpu"] = "cpu"
    intensity_lowres_prob: float = 0.25
    intensity_lowres_scale: tuple[float, float] = (0.5, 1.0)
    intensity_lowres_augmentation_device: Literal["cpu", "gpu"] = "cpu"
    intensity_gamma_prob: float = 0.4
    intensity_gamma_range: tuple[float, float] = (0.7, 1.5)
    intensity_gamma_p_invert: float = 0.25
    intensity_gamma_p_retain_stats: float = 1.0
    intensity_gamma_augmentation_device: Literal["cpu", "gpu"] = "cpu"
    # GIN augmentation
    gin_layers: int = 4
    gin_channels: int = 2
    gin_prob_kernel_size_one: float = 0.8
    gin_prob: float = 0.3
    gin_augmentation_device: Literal["cpu", "gpu"] = "cpu"

    save_dir: Path | None = None

    @model_validator(mode="after")
    def _validate_normalization(self) -> Self:
        normalization = self.normalization
        if isinstance(normalization, list):
            if not normalization:
                msg = "normalization list must not be empty"
                raise ValueError(msg)
            for entry in normalization:
                _validate_normalization_entry(entry)
        elif isinstance(normalization, dict):
            _validate_normalization_entry(normalization)
        else:
            msg = "normalization must be a dict or a list of dicts"
            raise TypeError(msg)
        if self.bbox_parametrization == "center_extents" and self.dfl_reg_max > 1:
            msg = "bbox_parametrization='center_extents' requires dfl_reg_max=1"
            raise ValueError(msg)
        return self

    def uses_gpu_augmentations(self) -> bool:
        return any(
            placement == "gpu"
            for placement in (
                self.intensity_noise_augmentation_device,
                self.intensity_blur_augmentation_device,
                self.intensity_mult_brightness_augmentation_device,
                self.intensity_add_brightness_augmentation_device,
                self.intensity_contrast_augmentation_device,
                self.intensity_lowres_augmentation_device,
                self.intensity_gamma_augmentation_device,
                self.gin_augmentation_device,
                self.channel_dropout_augmentation_device,
                self.zoom_augmentation_device,
                self.rotation_augmentation_device,
                self.cutout_augmentation_device,
                self.mirroring_augmentation_device,
            )
        )


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
                    f"{self.nc}-class dataset requires class indices 0-{self.nc - 1}, but there are"
                    f" invalid class indices {min(self.names.keys())}-{max(self.names.keys())} "
                    "defined in dataset.yaml"
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
    model_state_dict: dict
    ema_updates: int
    optimizer_state_dict: dict
    scaler_state_dict: dict
    train_config: DefaultConfig
    train_metrics: dict
    date: datetime.datetime


def load_model_checkpoint(checkpoint: Path) -> ModelCheckpoint:
    checkpoint = torch.load(checkpoint, map_location="cpu")
    return ModelCheckpoint(
        epoch=checkpoint["epoch"],
        best_fitness=checkpoint["best_fitness"],
        model_state_dict=checkpoint["model_state_dict"],
        ema_updates=checkpoint["ema_updates"],
        optimizer_state_dict=checkpoint["optimizer_state_dict"],
        scaler_state_dict=checkpoint["scaler_state_dict"],
        train_config=DefaultConfig.model_validate_json(checkpoint["train_config"]),
        train_metrics=checkpoint["train_metrics"],
        date=datetime.datetime.fromisoformat(checkpoint["date"]),
    )


def save_model_checkpoint(  # noqa: PLR0913
    paths: list[Path],
    epoch: int,
    best_fitness: float,
    model: HardcodedModel,
    ema_updates: int,
    optimizer: torch.optim.Optimizer,
    scaler: torch.GradScaler,
    train_config: DefaultConfig,
    train_metrics: dict[str, float],
) -> None:
    checkpoint = ModelCheckpoint(
        epoch=epoch,
        best_fitness=best_fitness,
        model_state_dict=model.state_dict(),
        ema_updates=ema_updates,
        optimizer_state_dict=optimizer.state_dict(),
        scaler_state_dict=scaler.state_dict(),
        train_config=train_config.model_dump_json(),
        train_metrics=train_metrics,
        date=datetime.datetime.now(tz=datetime.UTC).isoformat(),
    )
    for path in paths:
        torch.save(checkpoint, path)


def get_cfg(config_path: str | Path, overrides: dict[str, Any] | None) -> DefaultConfig:
    with Path(config_path).open() as f:
        yaml_config = yaml.safe_load(f)
    if overrides:
        if "save_dir" not in yaml_config:
            overrides.pop("save_dir", None)
        yaml_config = {**yaml_config, **overrides}

    if yaml_config.get("experiment_name") == "model":
        yaml_config["experiment_name"] = yaml_config.get("model", "").split(".")[0]
        LOGGER.warning(
            "'experiment_name=model' automatically updated to 'experiment_name=%s'.",
            yaml_config["experiment_name"],
        )

    return DefaultConfig.model_validate(yaml_config)


def get_dataset_cfg(config_path: str | Path) -> DatasetConfig:
    with Path(config_path).open() as f:
        yaml_config = yaml.safe_load(f)
    return DatasetConfig.model_validate(yaml_config)
