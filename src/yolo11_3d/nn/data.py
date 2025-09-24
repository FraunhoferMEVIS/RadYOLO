from __future__ import annotations

import json
import logging
import os
from collections.abc import Sized
from pathlib import Path
from typing import Literal, TypedDict, cast, NotRequired

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from yolo11_3d.nn.augmentations import (
    batch_generators_intensity_augmentations,
    enlarge_bounding_boxes,
    random_mirroring,
    random_zoom,
    tensor_cutout,
    random_rotation,
    random_channel_dropout,
    gin_augmentation,
)
from yolo11_3d.nn.config import DefaultConfig
from yolo11_3d.ultralytics.utils.ops import xyzwhd2xyzxyz, xyzxyz2xyzwhd

LOGGER = logging.getLogger(__name__)


class Sample(TypedDict):
    image: torch.Tensor
    shape: tuple[int, int, int]
    original_shape: tuple[int, int, int]
    ratio_pad: (
        tuple[tuple[float, float, float], tuple[int, int, int]]
        | None
    )
    classes: torch.Tensor
    bboxes: torch.Tensor
    masks: torch.Tensor | None
    case_name: str

class Batch(TypedDict):
    img: torch.Tensor
    ori_shape: tuple[tuple[int, int, int], ...]
    ratio_pad: tuple[
        tuple[
            tuple[float, float, float], tuple[int, int, int]
        ]
        | None,
        ...,
    ]
    shape: tuple[tuple[int, int, int], ...]
    cls: torch.Tensor
    bboxes: torch.Tensor
    batch_idx: torch.Tensor
    masks: tuple[torch.Tensor | None, ...]
    case_names: tuple[str]

class NormalizationConfig(TypedDict):
    type: Literal["none", "zscore", "ct", "255_to_01"]
    mean: NotRequired[float]
    std: NotRequired[float]
    clip_min: NotRequired[float]
    clip_max: NotRequired[float]

class NIfTIDatasetWithMasks(Dataset[Sample], Sized):
    def __init__(
        self,
        src_path: Path,
        imgsz: tuple[int, int, int],
        augment: bool = False,
        min_stride: int = 8,
        pad: float = 0.5,
        fold: int = 0,
        group: Literal["train", "val", "test"] = "train",
        normalization: NormalizationConfig | list[NormalizationConfig] = {"type": "none"},
        load_masks: bool = True,
        config: DefaultConfig | None = None,
    ) -> None:
        super().__init__()
        self.src_path = src_path
        self.imgsz = imgsz
        self.augment = augment
        self.min_stride = min_stride
        self.pad = pad
        self.fold = fold
        self.group = group
        self.normalization = normalization
        self.config = config
        self.load_masks = load_masks
        splits_file = os.path.join(src_path, "data_splits.json")
        with open(splits_file, "r") as file:
            splits = json.load(file)
        if fold >= len(splits):
            raise Exception(f"There are only {len(splits)} folds defined, but fold index {fold} was selected.")
        self.cases: list[str] = splits[fold][group]

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> Sample:
        with torch.no_grad():
            return self.augmentation(self.load_image_label_and_mask(index))

    def augmentation(self, sample: Sample) -> Sample:
        if not self.augment:
            return sample

        scale_tensor = torch.as_tensor(self.imgsz)[[2, 1, 0, 2, 1, 0]]

        image = sample["image"]
        classes = sample["classes"]
        bboxes = xyzwhd2xyzxyz(sample["bboxes"])
        masks = sample["masks"]
        bboxes = bboxes * scale_tensor

        labels = torch.cat((classes, bboxes), dim=1)

        image, labels, masks = random_zoom(
            image,
            labels,
            masks,
            self.config.max_zoom,
            self.config.min_zoom,
            self.config.prob_zoom,
            self.config.out_zoom_padding,
            self.config.zoom_uniform,
        )

        image, labels, masks = random_rotation(
            image,
            labels,
            masks,
            angle_ranges=(
                self.config.rotation_angle_x,
                self.config.rotation_angle_y,
                self.config.rotation_angle_z,
            ),
            p=self.config.prob_rotation,
            keep_box_volume=self.config.rotation_keep_box_volume,
        )

        image, labels, masks = tensor_cutout(image, labels, masks, self.config.cutout_params, self.config.prob_cutout)
        nl = len(labels)
        if nl:
            labels[:, 1:] = xyzxyz2xyzwhd(labels[:, 1:])

        image, labels, masks = enlarge_bounding_boxes(image, labels, masks, self.imgsz, self.min_stride)
        
        if nl:
            labels[:, 1:] = labels[:, 1:] / scale_tensor

        image, labels, masks = random_mirroring(
            image, labels, masks, self.config.mirror_axes, self.config.prob_mirroring
        )


        if self.config.do_batchgenerators_intensity_augmentation:
            image = batch_generators_intensity_augmentations(
                image,
                self.config.do_intensity_augmentations_per_channel,
                noise_prob=self.config.intensity_noise_prob,
                noise_variance=self.config.intensity_noise_variance,
                blur_prob=self.config.intensity_blur_prob,
                blur_sigma=self.config.intensity_blur_sigma,
                mult_brightness_prob=self.config.intensity_mult_brightness_prob,
                mult_brightness_range=self.config.intensity_mult_brightness_range,
                add_brightness_prob=self.config.intensity_add_brightness_prob,
                add_brightness_mu=self.config.intensity_add_brightness_mu,
                add_brightness_sigma=self.config.intensity_add_brightness_sigma,
                contrast_prob=self.config.intensity_contrast_prob,
                contrast_range=self.config.intensity_contrast_range,
                lowres_prob=self.config.intensity_lowres_prob,
                lowres_scale=self.config.intensity_lowres_scale,
                gamma_prob=self.config.intensity_gamma_prob,
                gamma_range=self.config.intensity_gamma_range,
                gamma_p_invert=self.config.intensity_gamma_p_invert,
                gamma_p_retain_stats=self.config.intensity_gamma_p_retain_stats,
            )
        
        image, labels, masks = gin_augmentation(
            image,
            labels,
            masks,
            self.config.gin_layers,
            self.config.gin_channels,
            self.config.gin_prob_kernel_size_one,
            self.config.gin_prob,
        )

        image, labels, masks = random_channel_dropout(
            image,
            labels,
            masks,
            drop_probabilities=self.config.channel_dropout_probabilities,
            fill_value=None if self.config.channel_dropout_use_mean else self.config.channel_dropout_fill_value,
        )

        sample["image"] = image
        sample["classes"] = labels[:, 0:1]
        sample["bboxes"] = labels[:, 1:]
        sample["masks"] = masks

        return sample

    def load_image_label_and_mask(self, index: int) -> Sample:
        case_name = self.cases[index]

        image = self.load_image(case_name)
        image, original_shape, resized_shape = self.resample_and_normalize(image)
        ratio = (
            resized_shape[0] / original_shape[0],
            resized_shape[1] / original_shape[1],
            resized_shape[2] / original_shape[2],
        )
        pad = (0, 0, 0)
        classes, bboxes, masks = self.load_labels_and_masks(case_name)

        return Sample(
            image=image,
            shape=resized_shape,
            original_shape=original_shape,
            ratio_pad=(ratio, pad),
            classes=classes,
            bboxes=bboxes,
            masks=masks,
            case_name=case_name,
        )

    def load_image(self, case_name: str) -> torch.Tensor:
        image_path = os.path.join(self.src_path, "images", f"{case_name}.nii.gz")
        if not os.path.exists(image_path):
            raise IOError(f"Image path does not exists: {image_path}. Is your dataset format correct?")
        nifti_image = nib.loadsave.load(image_path)
        if not isinstance(nifti_image, nib.nifti1.Nifti1Image):
            raise TypeError
        image = np.asanyarray(nifti_image.dataobj, dtype=np.float32)
        image_tensor = torch.from_numpy(image)
        image_tensor = image_tensor.permute(*reversed(range(image_tensor.ndim)))
        return image_tensor

    def resample_and_normalize(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int, int], tuple[int, int, int]]:
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image_resampled = cast(
            "torch.Tensor", F.interpolate(image.unsqueeze(0), self.imgsz, mode="trilinear")
        ).squeeze_(0)

        image_resampled = self._apply_normalization(image_resampled)

        return image_resampled, image.shape[1:], image_resampled.shape[1:]

    def _apply_normalization(self, image: torch.Tensor) -> torch.Tensor:
        """Apply per-channel normalization according to self.normalization.

        Supports:
        - NormalizationConfig: {type: "ct"|"zScore"|"255_to_01"|"none", ...params}
        - list[NormalizationConfig]: list of entries above (NormalizationConfig) for each input channel
        """
        if image.ndim != 4:
            return image
        C = image.shape[0]
        normalization_cfg = self.normalization

        eps = 1e-6
        if isinstance(normalization_cfg, list):
            if len(normalization_cfg) != C:
                raise ValueError(
                    f"Normalization config length ({len(normalization_cfg)}) does not match number of channels ({C})."
                )
            for c in range(C):
                entry = normalization_cfg[c]
                if not isinstance(entry, dict) or "type" not in entry:
                    raise ValueError("Invalid normalization entry; expected dict with 'type'")

                normalization_scheme = str(entry.get("type", "none"))

                x = image[c]
                if normalization_scheme == "none":
                    pass
                elif normalization_scheme == "255_to_01":
                    x = x.div(255.0).clamp_(0.0, 1.0)
                    image[c] = x
                elif normalization_scheme == "zscore":
                    mean = torch.mean(x)
                    std = torch.std(x, unbiased=False).clamp_min(eps)
                    image[c] = (x - mean) / std
                elif normalization_scheme == "ct":
                    clip_min = float(entry.get("clip_min", -1024.0))
                    clip_max = float(entry.get("clip_max", 3071.0))
                    mean = float(entry.get("mean", 0.0))
                    std = float(entry.get("std", 1.0))
                    std = max(std, eps)
                    x = x.clamp(clip_min, clip_max)
                    image[c] = (x - mean) / std
                else:
                    raise NotImplementedError(f"Unknown normalization scheme '{normalization_scheme}' for channel {c}")
        elif isinstance(normalization_cfg, dict):
            normalization_scheme = normalization_cfg["type"]
            if normalization_scheme == "none":
                pass
            elif normalization_scheme == "255_to_01":
                image = image.div(255.0).clamp_(0.0, 1.0)
            elif normalization_scheme == "zscore":
                mean = torch.mean(image)
                std = torch.std(image, unbiased=False).clamp_min(eps)
                image = (image - mean) / std
            elif normalization_scheme == "ct":
                clip_min = float(normalization_cfg.get("clip_min", -1024.0))
                clip_max = float(normalization_cfg.get("clip_max", 3071.0))
                mean = float(normalization_cfg.get("mean", 0.0))
                std = float(normalization_cfg.get("std", 1.0))
                std = max(std, eps)
                image = image.clamp(clip_min, clip_max)
                image = (image - mean) / std
            else:
                raise NotImplementedError(f"Unknown normalization scheme '{normalization_cfg}'")
        else:
            raise ValueError(f"Invalid normalization config; expected string or dict but got {normalization_cfg}")

        return image
    
    def load_labels_and_masks(self, case_name: str) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor | None]]:
        label_path = os.path.join(self.src_path, "labels", f"{case_name}.txt")
        with open(label_path, "r") as file:
            lines = file.read().strip().splitlines()
        if lines:
            string_labels = [line.split() for line in lines if len(line)]
            if any(len(label) not in (0, 7) for label in string_labels):
                raise Exception(f"Labels in wrong fomat in file {label_path}")
            labels = np.array(string_labels, dtype=np.float32)
        else:
            labels = np.zeros((0, 7))
        classes = torch.from_numpy(labels[:, 0:1])
        bboxes = torch.from_numpy(labels[:, 1:])

        mask_path = os.path.join(self.src_path, "masks", f"{case_name}.nii.gz")
        if self.load_masks and os.path.exists(mask_path):
            nifti_image = nib.loadsave.load(mask_path)
            if not isinstance(nifti_image, nib.nifti1.Nifti1Image):
                raise TypeError
            mask_numpy = np.asanyarray(nifti_image.dataobj, dtype=np.uint8)
            mask_tensor = torch.from_numpy(mask_numpy)
            mask_tensor = mask_tensor.permute(*reversed(range(mask_tensor.ndim)))
            mask_tensor = mask_tensor.unsqueeze(0)
            if len(mask_tensor.shape) == 4:
                mask_tensor = mask_tensor.unsqueeze(0)
            mask_tensor_resampled = cast(
                "torch.Tensor", F.interpolate(mask_tensor, self.imgsz, mode="nearest-exact")
            ).squeeze_(0)
        else:
            mask_tensor_resampled = None

        return classes, bboxes, mask_tensor_resampled

    @staticmethod
    def collate_fn(batch: list[Sample]) -> Batch:
        with torch.no_grad():
            return {
                "img": torch.stack([sample["image"] for sample in batch]),
                "ori_shape": tuple(sample["original_shape"] for sample in batch),
                "ratio_pad": tuple(sample["ratio_pad"] for sample in batch),
                "shape": tuple(sample["shape"] for sample in batch),
                "cls": torch.cat([sample["classes"] for sample in batch], 0),
                "bboxes": torch.cat([sample["bboxes"] for sample in batch], 0),
                "batch_idx": torch.cat(
                    [torch.full(sample["classes"].size()[:1], idx) for idx, sample in enumerate(batch)],
                    0,
                ),
                "masks": tuple(sample["masks"] for sample in batch),
                "case_names": tuple(sample["case_name"] for sample in batch),
            }


def build_dataloader(
    dataset: NIfTIDatasetWithMasks,
    batch_size: int,
    workers: int,
    shuffle: bool = True,
    device: torch.device | str = "cpu",
) -> DataLoader:
    """Return an DataLoader for training or validation set."""
    batch_size = min(batch_size, len(dataset))
    number_workers = min(os.cpu_count() or 1, workers)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=number_workers,
        pin_memory=(device.type if isinstance(device, torch.device) else device) != "cpu",
        collate_fn=getattr(dataset, "collate_fn", None),
        persistent_workers=True
    )
