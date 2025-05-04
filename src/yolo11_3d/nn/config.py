from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypedDict, assert_never

import annotated_types
import torch
import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    DirectoryPath,
    Field,
    FilePath,
    NewPath,
    NonNegativeInt,
    field_serializer,
    model_validator,
)

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

    task: str
    mode: str

    # Train settings
    model: FilePath | NewPath  # path to model file
    data: FilePath  # path to data file
    epochs: int = 100  # number of epochs to train for
    time: float | None = None  # number of hours to train for, overrides epochs if supplied
    patience: int = (
        100  # epochs to wait for no observable improvement for early stopping of training
    )
    batch: int = 16  # number of images per batch (-1 for AutoBatch)
    imgsz: DimList  # input images size as int for train and val modes, or list[h,w] for predict and export modes
    save: bool = True  # save train checkpoints and predict results
    save_period: int = -1  # Save checkpoint every x epochs (disabled if < 1)
    cache: bool = False  # True/ram, disk or False. use cache for data loading
    device: str = ""  # device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
    workers: int = 8  # number of worker threads for data loading (per RANK if DDP)
    project: str = ""  # project name
    name: int | str  # experiment name, results saved to 'project/name' directory
    exist_ok: bool = False  # whether to overwrite existing experiment
    pretrained: bool | FilePath = (
        True  # whether to use a pretrained model (bool) or a model to load weights from (Path)
    )
    optimizer: Literal["SGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"] = (
        "auto"  # optimizer to use
    )
    verbose: bool = True  # whether to print verbose output
    seed: int = 0  # random seed for reproducibility
    deterministic: bool = True  # whether to enable deterministic mode
    single_cls: bool = False  # train multi-class data as single-class
    rect: bool = (
        False  # rectangular training if mode='train' or rectangular validation if mode='val'
    )
    cos_lr: bool = False  # use cosine learning rate scheduler
    close_mosaic: int = 10  # disable mosaic augmentation for final epochs (0 to disable)
    resume: bool | FilePath = False  # resume training from last checkpoint
    amp: bool = True  # Automatic Mixed Precision (AMP) training, True runs AMP check
    fraction: float = 1.0  # dataset fraction to train on
    profile: bool = False  # profile ONNX and TensorRT speeds during training for loggers
    freeze: int | list[int] | None = (
        None  # freeze first n layers, or freeze list of layer indices during training
    )
    multi_scale: bool = False  # Whether to use multiscale during training
    # Segmentation
    # overlap_mask: bool = (
    #     True  # merge object masks into a single image mask during training (segment train only)
    # )
    # mask_ratio: int = 4 # mask downsample ratio (segment train only)
    # Classification
    # dropout: float = 0.0 # use dropout regularization

    # Val/Test Settings
    val: bool = True  # validate/test during training
    split: str = "val"  # dataset split to use for validation, i.e. 'val', 'test' or 'train'
    save_json: bool = False  # save results to JSON file
    save_hybrid: bool = False  # save hybrid version of labels (labels+additional predictions)
    conf: float | None = (
        None  # object confidence threshold for detection (default 0.25 predict, 0.001 val)
    )
    iou: float = 0.7  # intersection over union (IoU) threshold for NMS
    max_det: int = 300  # maximum number of detections per image
    half: bool = False  # use half precision (FP16)
    dnn: bool = False  # use OpenCV DNN for ONNX inference
    plots: bool = True  # save plots and images during train/val

    # Predict Settings
    source: DirectoryPath | None = None  # source directory for images or videos
    # vid_stride: int = 1 # video frame-rate stride
    # stream_buffer: bool = False # buffer all streaming frames or return the most recent frame
    visualize: bool = False  # visualize model features
    augment: bool = False  # apply image augmentation to prediction sources
    agnostic_nms: bool = False  # class-agnostic NMS
    classes: int | list[int] | None = (
        None  # filter results by class, i.e. classes=0, or classes=[0,2,3]
    )
    retina_masks: bool = False  # use high-resolution segmentation masks
    embed: list[int] | None = None  # return feature vectors/embeddings from given layers

    # Visualize Settings
    show: bool = False  # show predicted images and videos if environment allows
    # save_frames: bool = False  # save predicted individual video frames
    save_txt: bool = False  # save results as .txt file
    save_conf: bool = False  # save results with confidence scores
    save_crop: bool = False  # save cropped images with results
    show_labels: bool = True  # show prediction labels, i.e. 'person'
    show_conf: bool = True  # show prediction confidence, i.e. '0.99'
    show_boxes: bool = True  # show prediction boxes
    line_width: int | None = None  # line width of the bounding boxes. Scaled to image size if None

    # Export Settings
    format: str = "torchscript"  # format to export to
    keras: bool = False  # use Keras
    optimize: bool = False  # TorchScript: optimize for mobile
    int8: bool = False  # CoreML/TF INT8 quantization
    dynamic: bool = False  # ONNX/TF/TensorRT: dynamic axes
    simplify: bool = False  # ONNX: simplify model using `onnxslim`
    opset: int | None = None  # ONNX: opset version
    workspace: float | None = (
        None  # TensorRT: workspace size (GiB), `None` will let TensorRT auto-allocate memory
    )
    nms: bool = False  # CoreML: add NMS

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
    # pose: float = 12.0  # pose loss gain
    # kobj: float = 1.0  # keypoint obj loss gain
    nbs: int = 64  # nominal batch size
    # hsv_h: float = 0.015  # image HSV-Hue augmentation
    # hsv_s: float = 0.7  # image HSV-Saturation augmentation
    # hsv_v: float = 0.4  # image HSV-Value augmentation
    degrees: float = Field(0.0, le=180, ge=0)  # image roation (+/- deg)
    translate: float = 0.1  # image translation
    scale: float = 0.5  # image scale
    shear: float = 0.0  # image shear
    perspective: float = Field(0.0, ge=0, le=0.001)  # image perspective
    flipud: float = Field(0.0, ge=0, le=1)  # image flip up/down
    fliplr: float = Field(0.5, ge=0, le=1)  # image flip left/right
    # bgr: float = Field(0.0, ge=0, le=1)  # image channel bgr
    mosaic: float = Field(1.0, ge=0, le=1)  # image mosaic
    mixup: float = Field(0.0, ge=0, le=1)  # image mixup
    # copy_paste: float = Field(0.0, ge=0, le=1)  # segment copu-paste
    # copy_paste_mode: Literal["flip", "mixup"] = "flip"  # the method to do copy_paste augmentation
    # auto_augment: Literal["randaugment", "autoaugment", "augmix"] = (
    #     "randaugment"  # auto augmentation policy for classification
    # )
    erasing: float = Field(
        0.4, ge=0, le=0.9
    )  # probability of random erasing during classification training, 0 means no erasing.
    crop_fraction: float = Field(
        1.0, ge=0.1, le=1
    )  # image crop fraction for classification, 1.0 means no crop.

    cfg: FilePath | None = None  # for overriding defaults.yaml

    save_dir: Path | None = None


class DatasetConfig(BaseModel):
    path: Path
    train: Path | list[Path]
    val: Path | list[Path] | None = Field(
        validation_alias=AliasChoices("val", "validation"), default=None
    )
    test: Path | list[Path] | None = None
    names: dict[NonNegativeInt, str] | None = None
    nc: NonNegativeInt = 0
    download: str | None = None

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        self.path = (DATASETS_DIR / self.path).resolve()
        if not self.path.is_dir():
            raise ValueError
        for k, split in (
            ("train", self.train),
            ("val", self.val),
            ("test", self.test),
        ):
            if split is None:
                continue
            if isinstance(split, Path):
                x = (self.path / split).resolve()
                if not x.exists() and split.parts[0] == "..":
                    x = (self.path / Path(*split.parts[1:])).resolve()
                setattr(self, k, x)
            elif isinstance(split, list):
                setattr(self, k, [(self.path / path_).resolve() for path_ in split])
            else:
                assert_never(split)
        return self

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
