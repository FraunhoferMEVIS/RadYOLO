from __future__ import annotations

import datetime
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, TypedDict

import numpy as np
import torch
import yaml
from pydantic import (
    BaseModel,
    Field,
    FilePath,
    NonNegativeInt,
    model_validator,
)

from radyolo.config.spatial import (  # noqa: TC001, needs to be imported for pydantic model
    PatchSize,
    VoxelSize,
)

if TYPE_CHECKING:
    from radyolo.models.model import HardcodedModel

LOGGER = logging.getLogger(__name__)

DEPRECATED_CONFIG_KEYS = {"deterministic", "num_classes", "nms_time_limit"}
PrototypeLayer = Literal["higher_resolution", "full_resolution", "half_resolution"]
ImageLabelHeadFeature = Literal["o10", "o16", "o19", "o22"]


@dataclass(frozen=True)
class DetectionPostprocessConfig:
    confidence_threshold: float
    nms_iou_threshold: float
    agnostic_nms: bool
    max_det: int
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
    # Data and output settings
    data: Path  # path to data folder containing dataset.yaml
    fold: int  # fold from data splits file to use for training
    split_file_name: str = "data_splits.json"  # split file name inside the dataset folder
    output_folder_base: str = ""  # base folder for experiment outputs
    experiment_name: int | str  # experiment name, results saved to 'output_folder_base/experiment_name'

    # General Settings
    device: str = ""  # device to run on, "" for automatic selection, "cpu" for cpu, "0" for cuda:0, "0,1" for multi-GPU training
    amp: bool = True  # Automatic Mixed Precision (AMP) training, True runs AMP check
    # Set torch.backends.cudnn.benchmark = True which tests several convolution algorithms and
    # chooses the best one
    use_cudnn_benchmark: bool = True
    workers: int = 6  # number of dataloader workers per GPU/process
    save: bool = True  # save train checkpoints
    save_period: int = -1  # Save checkpoint every x epochs (disabled if < 1)
    exist_ok: bool = True  # whether to overwrite existing experiment
    resume: bool | FilePath = True  # resume training from last checkpoint
    verbose: bool = True  # whether to print verbose output
    seed: int = 0  # random seed for reproducibility
    # Model Parameters
    input_channels: int  # number of input channels for model
    num_classes: int = 0  # deprecated in config.yaml; inferred from dataset.yaml names
    model_scale: Literal["n", "s", "m", "l", "x"]  # allowed values: (n, s, m, l, x)

    # Optional Segmentation Head
    segmentation_output: bool = True  # add segmentation output to model
    prototype_masks: int = 8  # number of prototype masks for segmentation output
    prototype_layer: PrototypeLayer = "higher_resolution"

    # Detection Head Options
    # if True, add a separate objectness output and use CCE loss for class logits
    separate_objectness_head: bool = False
    include_stride4_output: bool = False  # if True, add a detection head on stride 4 level

    # Global image-label head
    enable_image_label_head: bool = False
    image_label_head_feature_maps: list[ImageLabelHeadFeature] = Field(
        default_factory=lambda: ["o10", "o16", "o19", "o22"]
    )
    image_label_head_projection_channels: int = Field(128, ge=1)
    image_label_head_mlp_hidden_layers: int = Field(1, ge=0)
    image_label_head_mlp_hidden_channels: int = Field(256, ge=1)
    image_label_head_dropout: float = Field(0.1, ge=0.0, lt=1.0)
    image_label_multiclass_cce_gain: float = Field(1.0, ge=0.0)
    image_label_multiclass_cce_class_weights: dict[str, list[float]] = Field(
        default_factory=dict
    )
    image_label_survival_num_bins: int = Field(32, ge=1)
    image_label_survival_mtlr_gain: float = Field(1.0, ge=0.0)
    image_label_survival_cox_gain: float = Field(1.0, ge=0.0)

    # Hyperparameters
    epochs: int = 1000  # number of epochs to train for
    patience: int = (
        300  # epochs to wait for no observable improvement for early stopping of training
    )
    training_mode: Literal["patchwise", "wholeimage"] = (
        "wholeimage"  # whether training/validation sample image patches or use resized whole images
    )
    patch_size: PatchSize  # input patch size stored in x, y, z (width, height, depth) order
    voxel_size: VoxelSize | None = None  # optional inference/training resampling target in x, y, z
    # Normalization for image intensities. Can be {type: "none"/"zscore"/"255_to_01"/"ct"} to be
    # applied globally or a list of such dicts for each channel. For ct you have to define
    # mean, std, clip_min and clip_max in the dict.
    normalization: dict | list[dict] = {"type": "none"}
    apply_preprocessing: bool = True  # keep preprocessing settings in config for inference, and apply them while loading raw training data
    foreground_oversampling: float = Field(
        0.0,
        ge=0.0,
        lt=1.0,
    )  # extra train-epoch sample fraction to draw from foreground cases
    min_box_crop_fraction: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
    )  # minimum fraction of a box volume that must remain inside a sampled patch to keep it
    batch_size: int = 2  # number of images per GPU/process
    nominal_batch_size: int = 64  # nominal effective global batch size for gradient accumulation
    optimizer: Literal["SGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"] = (
        "Adam"  # optimizer to use
    )
    initial_learning_rate: float = 0.01  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    final_learning_rate_ratio: float = 0.01  # final learning rate ratio (initial_learning_rate * final_learning_rate_ratio)
    momentum: float = 0.937  # SGD momentum/Adam beta1
    weight_decay: float = 5e-4  # optimizer weight decay 5e-4
    warmup_epochs: float = 3.0  # warmup epochs
    warmup_momentum: float = 0.8  # warmup initial momentum
    warmup_bias_lr: float = 0.1  # warmup initial bias learning rate
    grad_clip_max_norm: float = 10.0  # max norm used for gradient clipping
    tal_topk: int = Field(
        10, ge=1
    )  # initial top-k anchors to keep per ground truth for task-aligned assignment
    tal_topk_final: int | None = Field(
        default=None, ge=1
    )  # final top-k anchors reached by the TAL schedule; defaults to tal_topk
    tal_topk_final_epoch: int | None = Field(
        default=None, ge=1
    )  # 1-based epoch at which tal_topk_final is reached; defaults to epochs
    tal_alpha: float = 0.5  # class score exponent in task-aligned assignment metric
    tal_beta: float = 1.0  # IoU exponent in task-aligned assignment metric
    ema_decay: float = 0.9999  # exponential moving average decay
    ema_tau: float = 400.0  # EMA ramp-up factor, epochs until full ema_decay is used
    objectness_loss: Literal["bce", "focal"] = (
        "bce"  # can be either "bce" (binary cross entropy) or "focal" (focal loss)
    )
    gamma: float = 2.0  # quality focal loss gamma (used when objectness_loss="focal")
    class_loss_gain: float = 0.5  # classification / objectness loss gain
    class_separate_loss_gain: float = 0.5  # class loss gain for separate softmax classification
    mask_loss_gain: float = 1.0  # mask loss gain
    box_loss_gain: float = 7.5  # box loss gain (based on complete IoU (CIoU) loss)
    dfl_loss_gain: float = 1.5  # DFL loss gain (based on DFL loss, or l1 loss if dfl_reg_max=1)
    dfl_reg_max: int = Field(16, ge=1)  # DFL histogram size; use 1 for direct box edge regression without DFL
    bbox_center_offset_limit: float = Field(
        4.0, gt=0.5
    )  # max absolute anchor-relative center offset for center_extents bbox prediction
    bbox_parametrization: Literal["ltrb", "center_extents"] = (
        "ltrb"  # bbox prediction format, choices=[ltrb, center_extents]
    )

    # Validation settings
    val: bool = True  # validate/test during training
    confidence_threshold_validation: float = (
        0.001  # object confidence threshold for validation/detection metrics
    )
    nms_iou_threshold: float = 0.2  # intersection over union (IoU) threshold for non-maxiumum suppression (NMS)
    agnostic_nms: bool = False  # class-agnostic NMS
    max_det: int = 300  # maximum number of detections per image
    clip_boxes_to_image: bool = True  # clip predicted 3D boxes to image bounds during validation/prediction
    nms_iou_threshold_candidates: list[float] = Field(
        default_factory=lambda: [0.02, 0.04, 0.06, 0.1, 0.15, 0.2]
    )  # NMS IoU thresholds evaluated after training for best/last checkpoints
    fitness_metric_weights: tuple[float, ...] = (
        0.4,
        0.4,
        0.2,
        0.0,
        0.0,
        0.0,
    )  # weights for [mAP10-95, mAP10-95Agn, dice, image_label_balanced_accuracy, image_label_multiclass_roc_auc, image_label_c_index] when computing validation fitness

    # Plot settings
    plots: bool = True  # save plots and images during train/val
    validation_plot_projection: Literal["max", "mean"] = (
        "max"  # projection reduction for validation plots, choices=[max, mean]
    )
    validation_plot_image_mode: Literal["default", "channel_difference"] = (
        "default"  # source image for validation plots, choices=[default, channel_difference]
    )
    validation_plot_difference_channels: tuple[NonNegativeInt, NonNegativeInt] = (
        2,
        0,
    )  # channels used for difference plots as [minuend, subtrahend]
    confidence_threshold_plotting: float = 0.1  # object confidence threshold for validation plots

    # Zoom Augmentation
    max_zoom: float = Field(1.5, ge=0)
    min_zoom: float = Field(0.7, ge=0)
    out_zoom_padding: Literal["constant", "random"] = "constant"
    prob_zoom: float = Field(0.4, ge=0, le=1.0)
    zoom_uniform: bool = False  # if false, enable anisotropic per-axis zoom factors
    zoom_augmentation_device: Literal["cpu", "gpu"] = "gpu"  # set to "cpu" or "gpu"

    # Mirroring Augmentation
    prob_mirroring: float = Field(0.5, ge=0, le=1.0)
    mirror_axes: list[Literal["x", "y", "z"]] = Field(default_factory=lambda: ["z", "y", "x"])
    mirroring_augmentation_device: Literal["cpu", "gpu"] = "gpu"  # set to "cpu" or "gpu"

    # Rotation augmentation
    prob_rotation: float = Field(0.2, ge=0, le=1.0)
    rotation_angle: float | None = None
    rotation_angle_x: tuple[float, float] = (-15.0, 15.0)
    rotation_angle_y: tuple[float, float] = (-15.0, 15.0)
    rotation_angle_z: tuple[float, float] = (-15.0, 15.0)
    rotation_keep_box_volume: bool = True
    rotation_augmentation_device: Literal["cpu", "gpu"] = "gpu"  # set to "cpu" or "gpu"

    # Cutout Augmentation
    prob_cutout: float = Field(0.1, ge=0, le=1.0)
    cutout_augmentation_device: Literal["cpu", "gpu"] = "gpu"  # set to "cpu" or "gpu"
    cutout_params: list[tuple[float, int]] = Field(
        default_factory=lambda: [
            (0.5, 1),
            (0.25, 2),
            (0.125, 4),
        ]
    )  # Cutout params (size_fraction, repeats)

    # Intensity augmentation parameters
    do_batchgenerators_intensity_augmentation: bool = True
    do_intensity_augmentations_per_channel: bool = False
    intensity_noise_prob: float = 0.1
    intensity_noise_variance: tuple[float, float] = (0.0, 0.1)
    intensity_noise_augmentation_device: Literal["cpu", "gpu"] = "gpu"  # set to "cpu" or "gpu"
    intensity_blur_prob: float = 0.2
    intensity_blur_sigma: tuple[float, float] = (0.5, 1.0)
    intensity_blur_augmentation_device: Literal["cpu", "gpu"] = "gpu"  # set to "cpu" or "gpu"
    intensity_mult_brightness_prob: float = 0.15
    intensity_mult_brightness_range: tuple[float, float] = (0.75, 1.25)
    intensity_mult_brightness_augmentation_device: Literal["cpu", "gpu"] = "gpu"  # set to "cpu" or "gpu"
    intensity_add_brightness_prob: float = 0.15
    intensity_add_brightness_mu: float = 0.0
    intensity_add_brightness_sigma: float = 0.4
    intensity_add_brightness_augmentation_device: Literal["cpu", "gpu"] = "gpu"  # set to "cpu" or "gpu"
    intensity_contrast_prob: float = 0.15
    intensity_contrast_range: tuple[float, float] = (0.75, 1.25)
    intensity_contrast_augmentation_device: Literal["cpu", "gpu"] = "gpu"  # set to "cpu" or "gpu"
    intensity_lowres_prob: float = 0.25
    intensity_lowres_scale: tuple[float, float] = (0.5, 1.0)
    intensity_lowres_augmentation_device: Literal["cpu", "gpu"] = "gpu"  # set to "cpu" or "gpu"
    intensity_gamma_prob: float = 0.4
    intensity_gamma_range: tuple[float, float] = (0.7, 1.5)
    intensity_gamma_p_invert: float = 0.25
    intensity_gamma_p_retain_stats: float = 1.0
    intensity_gamma_augmentation_device: Literal["cpu", "gpu"] = "gpu"  # set to "cpu" or "gpu"

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_prototype_layer_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "prototype_layer" in data:
            migrated = dict(data)
        else:
            legacy_modes = [
                ("full_resolution_prototypes", "full_resolution"),
                ("half_resolution_prototypes", "half_resolution"),
                ("higher_resolution_prototypes", "higher_resolution"),
            ]
            enabled_modes = [mode for key, mode in legacy_modes if data.get(key) is True]
            if len(enabled_modes) > 1:
                msg = (
                    "prototype selection fields are mutually exclusive; got multiple legacy "
                    f"values enabled: {enabled_modes}"
                )
                raise ValueError(msg)

            migrated = dict(data)
            migrated["prototype_layer"] = (
                enabled_modes[0] if enabled_modes else "higher_resolution"
            )
            for key, _ in legacy_modes:
                migrated.pop(key, None)

        if "rotation_angle" in migrated:
            rotation_angle = migrated["rotation_angle"]
            if rotation_angle is not None:
                if isinstance(rotation_angle, bool) or not isinstance(rotation_angle, (int, float)):
                    msg = "rotation_angle must be a numeric scalar when provided"
                    raise TypeError(msg)
                angle_magnitude = float(rotation_angle)
                angle_range = (-abs(angle_magnitude), abs(angle_magnitude))
                for axis_key in ("rotation_angle_x", "rotation_angle_y", "rotation_angle_z"):
                    migrated.setdefault(axis_key, angle_range)

        return migrated

    @model_validator(mode="after")
    def validate_nms_optimization_candidates(self) -> Self:
        candidates = [float(candidate) for candidate in self.nms_iou_threshold_candidates]
        if not candidates:
            msg = "nms_iou_threshold_candidates must contain at least one value"
            raise ValueError(msg)
        invalid_candidates = [candidate for candidate in candidates if not 0.0 <= candidate <= 1.0]
        if invalid_candidates:
            msg = (
                "nms_iou_threshold_candidates must be between 0.0 and 1.0 inclusive; "
                f"got {invalid_candidates}"
            )
            raise ValueError(msg)
        self.nms_iou_threshold_candidates = sorted(set(candidates))
        return self

    # GIN augmentation
    gin_layers: int = 4
    gin_channels: int = 2
    gin_prob_kernel_size_one: float = 0.8
    gin_prob: float = 0.3
    gin_augmentation_device: Literal["cpu", "gpu"] = "gpu"  # set to "cpu" or "gpu"

    # Channel dropout augmentation (drop at most one channel)
    channel_dropout_probabilities: tuple[float, ...] = Field(
        default_factory=tuple
    )  # must be empty or match channel count
    channel_dropout_fill_value: float = 0.0  # ignored if channel_dropout_use_mean=true
    channel_dropout_use_mean: bool = False  # if true, replace dropped channel with its mean
    channel_dropout_augmentation_device: Literal["cpu", "gpu"] = "gpu"  # set to "cpu" or "gpu"

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
        if not self.channel_dropout_probabilities:
            self.channel_dropout_probabilities = (0.0,) * self.input_channels
        elif len(self.channel_dropout_probabilities) != self.input_channels:
            msg = (
                "channel_dropout_probabilities length must match input_channels, got "
                f"{len(self.channel_dropout_probabilities)} and {self.input_channels}"
            )
            raise ValueError(msg)
        if self.num_classes < 1:
            msg = "num_classes must be inferred from dataset.yaml names and be at least 1"
            raise ValueError(msg)
        if self.bbox_parametrization == "center_extents" and self.dfl_reg_max > 1:
            msg = "bbox_parametrization='center_extents' requires dfl_reg_max=1"
            raise ValueError(msg)
        if self.tal_topk_final is None:
            self.tal_topk_final = self.tal_topk
        if self.tal_topk_final_epoch is None:
            self.tal_topk_final_epoch = self.epochs
        if len(self.fitness_metric_weights) not in {3, 6}:
            msg = "fitness_metric_weights must contain exactly three legacy entries or six extended entries"
            raise ValueError(msg)
        if any(weight < 0 for weight in self.fitness_metric_weights):
            msg = "fitness_metric_weights entries must be non-negative"
            raise ValueError(msg)
        if sum(self.fitness_metric_weights) <= 0:
            msg = "fitness_metric_weights must sum to a positive value"
            raise ValueError(msg)
        if self.enable_image_label_head and not self.image_label_head_feature_maps:
            msg = "image_label_head_feature_maps must contain at least one feature map when enabled"
            raise ValueError(msg)
        if not np.isfinite(self.image_label_survival_mtlr_gain):
            msg = "image_label_survival_mtlr_gain must be finite"
            raise ValueError(msg)
        if not np.isfinite(self.image_label_survival_cox_gain):
            msg = "image_label_survival_cox_gain must be finite"
            raise ValueError(msg)
        for task_name, class_weights in self.image_label_multiclass_cce_class_weights.items():
            if not task_name:
                msg = "image_label_multiclass_cce_class_weights keys must be non-empty"
                raise ValueError(msg)
            if not class_weights:
                msg = (
                    "image_label_multiclass_cce_class_weights entries must contain at least "
                    f"one weight, but '{task_name}' is empty"
                )
                raise ValueError(msg)
            if any(not np.isfinite(weight) for weight in class_weights):
                msg = (
                    "image_label_multiclass_cce_class_weights entries must be finite, but "
                    f"'{task_name}' contains {class_weights}"
                )
                raise ValueError(msg)
            if any(weight < 0.0 for weight in class_weights):
                msg = (
                    "image_label_multiclass_cce_class_weights entries must be non-negative, "
                    f"but '{task_name}' contains {class_weights}"
                )
                raise ValueError(msg)
            if not any(weight > 0.0 for weight in class_weights):
                msg = (
                    "image_label_multiclass_cce_class_weights entries must contain at least "
                    f"one positive weight, but '{task_name}' contains {class_weights}"
                )
                raise ValueError(msg)
        self.image_label_head_feature_maps = list(dict.fromkeys(self.image_label_head_feature_maps))
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


def _sanitize_checkpoint_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        items = ((key, _sanitize_checkpoint_value(item)) for key, item in value.items())
        try:
            return value.__class__(items)
        except TypeError:
            return dict(items)
    if isinstance(value, list):
        return [_sanitize_checkpoint_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_checkpoint_value(item) for item in value)
    return value


def load_model_checkpoint(checkpoint: Path) -> ModelCheckpoint:
    checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
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
        torch.save(_sanitize_checkpoint_value(checkpoint), path)


def get_dataset_config(dataset_folder: str | Path) -> DatasetConfig:
    return get_dataset_cfg(Path(dataset_folder) / "dataset.yaml")


def _num_classes_from_dataset(yaml_config: dict[str, Any]) -> int:
    dataset_root = yaml_config.get("data")
    if dataset_root is None:
        return int(yaml_config.get("num_classes", 0))
    dataset_cfg = get_dataset_config(dataset_root)
    if dataset_cfg.names is None:
        msg = "dataset.yaml must define 'names' so num_classes can be inferred"
        raise ValueError(msg)
    return len(dataset_cfg.names)


def get_cfg(config_path: str | Path, overrides: dict[str, Any] | None) -> DefaultConfig:
    config_path = Path(config_path)
    with config_path.open() as f:
        yaml_config = yaml.safe_load(f) or {}
    deprecated_config_keys = DEPRECATED_CONFIG_KEYS.intersection(yaml_config)
    if overrides:
        deprecated_config_keys |= DEPRECATED_CONFIG_KEYS.intersection(overrides)
        yaml_config = {**yaml_config, **overrides}

    yaml_config.pop("deterministic", None)

    yaml_config["num_classes"] = _num_classes_from_dataset(yaml_config)
    if "deterministic" in deprecated_config_keys:
        LOGGER.warning(
            "Config field 'deterministic' is deprecated and ignored."
        )
    if "num_classes" in deprecated_config_keys:
        LOGGER.warning(
            (
                "Config field 'num_classes' is deprecated and ignored; "
                "using %d classes from %s instead."
            ),
            yaml_config["num_classes"],
            Path(yaml_config["data"]) / "dataset.yaml",
        )
    if "nms_time_limit" in deprecated_config_keys:
        LOGGER.warning("Config field 'nms_time_limit' is deprecated and ignored.")

    return DefaultConfig.model_validate(yaml_config)


def get_dataset_cfg(config_path: str | Path) -> DatasetConfig:
    with Path(config_path).open() as f:
        yaml_config = yaml.safe_load(f)
    return DatasetConfig.model_validate(yaml_config)
