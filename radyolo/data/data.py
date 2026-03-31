from __future__ import annotations

import json
import logging
import os
from collections.abc import Sized
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from radyolo.data.augmentations import (
    batch_generators_intensity_augmentations,
    ensure_non_zero_box_extents,
    gin_augmentation,
    random_channel_dropout,
    random_mirroring,
    random_rotation,
    random_zoom,
    tensor_cutout,
)
from radyolo.data.preprocessing import preprocess_image, resample_mask
from radyolo.utils.ops import xyzwhd2xyzxyz, xyzxyz2xyzwhd

if TYPE_CHECKING:
    from radyolo.config.config import DefaultConfig
    from radyolo.config.spatial import PatchSize
    from radyolo.data.normalization import NormalizationConfig

LOGGER = logging.getLogger(__name__)


def _has_intensity_augmentations_on_device(
    config: DefaultConfig,
    placement: Literal["cpu", "gpu"],
) -> bool:
    return any(
        augmentation_device == placement
        for augmentation_device in (
            config.intensity_noise_augmentation_device,
            config.intensity_blur_augmentation_device,
            config.intensity_mult_brightness_augmentation_device,
            config.intensity_add_brightness_augmentation_device,
            config.intensity_contrast_augmentation_device,
            config.intensity_lowres_augmentation_device,
            config.intensity_gamma_augmentation_device,
        )
    )


def _apply_configured_image_augmentations(
    image: torch.Tensor,
    config: DefaultConfig,
    placement: Literal["cpu", "gpu"],
) -> torch.Tensor:
    if config.do_batchgenerators_intensity_augmentation and _has_intensity_augmentations_on_device(
        config,
        placement,
    ):
        image = batch_generators_intensity_augmentations(
            image,
            config.do_intensity_augmentations_per_channel,
            noise_prob=config.intensity_noise_prob,
            noise_variance=config.intensity_noise_variance,
            noise_augmentation_device=config.intensity_noise_augmentation_device,
            blur_prob=config.intensity_blur_prob,
            blur_sigma=config.intensity_blur_sigma,
            blur_augmentation_device=config.intensity_blur_augmentation_device,
            mult_brightness_prob=config.intensity_mult_brightness_prob,
            mult_brightness_range=config.intensity_mult_brightness_range,
            mult_brightness_augmentation_device=config.intensity_mult_brightness_augmentation_device,
            add_brightness_prob=config.intensity_add_brightness_prob,
            add_brightness_mu=config.intensity_add_brightness_mu,
            add_brightness_sigma=config.intensity_add_brightness_sigma,
            add_brightness_augmentation_device=config.intensity_add_brightness_augmentation_device,
            contrast_prob=config.intensity_contrast_prob,
            contrast_range=config.intensity_contrast_range,
            contrast_augmentation_device=config.intensity_contrast_augmentation_device,
            lowres_prob=config.intensity_lowres_prob,
            lowres_scale=config.intensity_lowres_scale,
            lowres_augmentation_device=config.intensity_lowres_augmentation_device,
            gamma_prob=config.intensity_gamma_prob,
            gamma_range=config.intensity_gamma_range,
            gamma_p_invert=config.intensity_gamma_p_invert,
            gamma_p_retain_stats=config.intensity_gamma_p_retain_stats,
            gamma_augmentation_device=config.intensity_gamma_augmentation_device,
            placement=placement,
        )

    if config.gin_augmentation_device == placement:
        image = gin_augmentation(
            image,
            config.gin_layers,
            config.gin_channels,
            config.gin_prob_kernel_size_one,
            config.gin_prob,
        )

    if config.channel_dropout_augmentation_device == placement:
        image = random_channel_dropout(
            image,
            drop_probabilities=config.channel_dropout_probabilities,
            fill_value=None
            if config.channel_dropout_use_mean
            else config.channel_dropout_fill_value,
        )

    return image


def _has_spatial_augmentations_on_device(
    config: DefaultConfig,
    placement: Literal["cpu", "gpu"],
) -> bool:
    return any(
        augmentation_device == placement
        for augmentation_device in (
            config.zoom_augmentation_device,
            config.rotation_augmentation_device,
            config.cutout_augmentation_device,
            config.mirroring_augmentation_device,
        )
    )


def _apply_configured_augmentations_to_sample(
    image: torch.Tensor,
    classes: torch.Tensor,
    bboxes: torch.Tensor,
    masks: torch.Tensor | None,
    config: DefaultConfig,
    patch_size: PatchSize,
    placement: Literal["cpu", "gpu"],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if _has_spatial_augmentations_on_device(config, placement):
        image, classes, bboxes, masks = _apply_configured_spatial_augmentations_to_sample(
            image,
            classes,
            bboxes,
            masks,
            config,
            patch_size,
            placement,
        )
    image = _apply_configured_image_augmentations(image, config, placement)
    return image, classes, bboxes, masks


def _apply_configured_spatial_augmentations_to_sample(
    image: torch.Tensor,
    classes: torch.Tensor,
    bboxes: torch.Tensor,
    masks: torch.Tensor | None,
    config: DefaultConfig,
    patch_size: PatchSize,
    placement: Literal["cpu", "gpu"],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    labels = torch.cat((classes, bboxes), dim=1)
    scale_tensor = torch.tensor(
        patch_size.get_zyx(),
        device=labels.device,
        dtype=labels.dtype,
    )[[2, 1, 0, 2, 1, 0]]
    labels[:, 1:] = xyzwhd2xyzxyz(labels[:, 1:])
    scale_tensor = scale_tensor.to(device=labels.device, dtype=labels.dtype)
    labels[:, 1:] = labels[:, 1:] * scale_tensor

    fill_value = float(torch.min(image).item())

    if config.zoom_augmentation_device == placement:
        image, labels, masks = random_zoom(
            image,
            labels,
            masks,
            config.max_zoom,
            config.min_zoom,
            config.prob_zoom,
            config.out_zoom_padding,
            fill_value,
            config.zoom_uniform,
        )

    if config.rotation_augmentation_device == placement:
        image, labels, masks = random_rotation(
            image,
            labels,
            masks,
            angle_ranges=(
                config.rotation_angle_x,
                config.rotation_angle_y,
                config.rotation_angle_z,
            ),
            p=config.prob_rotation,
            keep_box_volume=config.rotation_keep_box_volume,
            fill_value=fill_value,
        )

    if config.cutout_augmentation_device == placement:
        image, labels, masks = tensor_cutout(
            image,
            labels,
            masks,
            config.cutout_params,
            config.prob_cutout,
        )

    if labels.shape[0] > 0:
        labels[:, 1:] = xyzxyz2xyzwhd(labels[:, 1:])
        labels = ensure_non_zero_box_extents(labels)
        scale_tensor = scale_tensor.to(device=labels.device, dtype=labels.dtype)
        labels[:, 1:] = labels[:, 1:] / scale_tensor

    if config.mirroring_augmentation_device == placement:
        image, labels, masks = random_mirroring(
            image,
            labels,
            masks,
            config.mirror_axes,
            config.prob_mirroring,
        )

    return image, labels[:, 0:1], labels[:, 1:], masks


def apply_configured_augmentations_to_batch(
    batch: Batch,
    config: DefaultConfig,
    patch_size: PatchSize,
    placement: Literal["cpu", "gpu"],
) -> Batch:
    images = batch["img"]
    batch_device = images.device
    classes_out: list[torch.Tensor] = []
    bboxes_out: list[torch.Tensor] = []
    batch_idx_out: list[torch.Tensor] = []
    masks_out: list[torch.Tensor | None] = []

    for sample_idx in range(images.shape[0]):
        sample_mask = batch["batch_idx"] == sample_idx
        sample_classes = batch["cls"][sample_mask]
        sample_bboxes = batch["bboxes"][sample_mask]
        sample_masks = batch["masks"][sample_idx]

        image_aug, classes_aug, bboxes_aug, masks_aug = _apply_configured_augmentations_to_sample(
            images[sample_idx],
            sample_classes,
            sample_bboxes,
            sample_masks,
            config,
            patch_size,
            placement,
        )
        image_aug = image_aug.to(device=batch_device)
        classes_aug = classes_aug.to(device=batch_device)
        bboxes_aug = bboxes_aug.to(device=batch_device)
        if masks_aug is not None:
            masks_aug = masks_aug.to(device=batch_device)
        images[sample_idx] = image_aug
        masks_out.append(masks_aug)

        if classes_aug.shape[0]:
            classes_out.append(classes_aug)
            bboxes_out.append(bboxes_aug)
            batch_idx_out.append(
                torch.full(
                    (classes_aug.shape[0],),
                    sample_idx,
                    dtype=batch["batch_idx"].dtype,
                    device=batch["batch_idx"].device,
                )
            )

    batch["img"] = images
    batch["masks"] = tuple(masks_out)
    if classes_out:
        batch["cls"] = torch.cat(classes_out, 0)
        batch["bboxes"] = torch.cat(bboxes_out, 0)
        batch["batch_idx"] = torch.cat(batch_idx_out, 0)
    else:
        batch["cls"] = batch["cls"].new_empty((0, 1))
        batch["bboxes"] = batch["bboxes"].new_empty((0, 6))
        batch["batch_idx"] = batch["batch_idx"].new_empty((0,), dtype=batch["batch_idx"].dtype)
    return batch


def apply_configured_spatial_augmentations_to_batch(
    batch: Batch,
    config: DefaultConfig,
    patch_size: PatchSize,
    placement: Literal["cpu", "gpu"],
) -> Batch:
    return apply_configured_augmentations_to_batch(batch, config, patch_size, placement)


class Sample(TypedDict):
    image: torch.Tensor
    shape: tuple[int, int, int]
    original_shape: tuple[int, int, int]
    ratio_pad: tuple[tuple[float, float, float], tuple[int, int, int]] | None
    classes: torch.Tensor
    bboxes: torch.Tensor
    masks: torch.Tensor | None
    case_name: str


class Batch(TypedDict):
    img: torch.Tensor
    ori_shape: tuple[tuple[int, int, int], ...]
    ratio_pad: tuple[
        tuple[tuple[float, float, float], tuple[int, int, int]] | None,
        ...,
    ]
    shape: tuple[tuple[int, int, int], ...]
    cls: torch.Tensor
    bboxes: torch.Tensor
    batch_idx: torch.Tensor
    masks: tuple[torch.Tensor | None, ...]
    case_names: tuple[str, ...]


class NIfTIDatasetWithMasks(Dataset[Sample], Sized):
    def __init__(
        self,
        src_path: Path,
        patch_size: PatchSize,
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
        self.patch_size = patch_size
        self.augment = augment
        self.min_stride = min_stride
        self.pad = pad
        self.fold = fold
        self.group = group
        self.normalization = normalization
        self.config = config
        self.load_masks = load_masks
        splits_file = Path(src_path) / "data_splits.json"
        with splits_file.open("r") as file:
            splits = json.load(file)
        if fold >= len(splits):
            msg = f"There are only {len(splits)} folds defined, but fold index {fold} was selected."
            raise ValueError(msg)
        self.cases: list[str] = splits[fold][group]
        self.image_format = self._detect_image_format()

    def _detect_image_format(self) -> Literal["nifti", "numpy"]:
        for case_name in self.cases:
            nifti_path = Path(self.src_path) / "images" / f"{case_name}.nii.gz"
            if nifti_path.exists():
                return "nifti"
            numpy_path = Path(self.src_path) / "images" / f"{case_name}.npy"
            if numpy_path.exists():
                return "numpy"
        msg = (
            "Could not detect dataset image format. Expected files in images/ as "
            "<case>.nii.gz or <case>.npy for the configured split."
        )
        raise FileNotFoundError(msg)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> Sample:
        with torch.no_grad():
            return self.augmentation(self.load_image_label_and_mask(index))

    def augmentation(self, sample: Sample) -> Sample:
        if not self.augment:
            return sample

        image, classes, bboxes, masks = (
            sample["image"],
            sample["classes"],
            sample["bboxes"],
            sample["masks"],
        )

        image, classes, bboxes, masks = _apply_configured_augmentations_to_sample(
            image,
            classes,
            bboxes,
            masks,
            self.config,
            self.patch_size,
            placement="cpu",
        )

        sample["image"] = image
        sample["classes"] = classes
        sample["bboxes"] = bboxes
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
        if self.image_format == "nifti":
            image_path = Path(self.src_path) / "images" / f"{case_name}.nii.gz"
            if not image_path.exists():
                msg = f"Image path does not exists: {image_path}. Is your dataset format correct?"
                raise OSError(msg)
            nifti_image = nib.loadsave.load(image_path)
            if not isinstance(nifti_image, nib.nifti1.Nifti1Image):
                raise TypeError
            image = np.asanyarray(nifti_image.dataobj, dtype=np.float32)
        else:
            image_path = Path(self.src_path) / "images" / f"{case_name}.npy"
            if not image_path.exists():
                msg = f"Image path does not exists: {image_path}. Is your dataset format correct?"
                raise OSError(msg)
            image = np.load(image_path, allow_pickle=False).astype(np.float32, copy=False)

        image_tensor = torch.from_numpy(image)
        return image_tensor.permute(*reversed(range(image_tensor.ndim)))

    def resample_and_normalize(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int, int], tuple[int, int, int]]:
        image_resampled_normalized, original_shape, resized_shape = preprocess_image(
            image,
            self.normalization,
            target_shape=self.patch_size,
            output_dtype=torch.float32,
        )
        return image_resampled_normalized, original_shape, resized_shape

    def load_labels_and_masks(
        self, case_name: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        label_path = Path(self.src_path) / "labels" / f"{case_name}.txt"
        with label_path.open("r") as file:
            lines = file.read().strip().splitlines()
        if lines:
            string_labels = [line.split() for line in lines if len(line)]
            if any(len(label) not in (0, 7) for label in string_labels):
                msg = f"Labels in wrong fomat in file {label_path}"
                raise ValueError(msg)
            labels = np.array(string_labels, dtype=np.float32)
        else:
            labels = np.zeros((0, 7))
        classes = torch.from_numpy(labels[:, 0:1])
        bboxes = torch.from_numpy(labels[:, 1:])

        mask_nifti_path = Path(self.src_path) / "masks" / f"{case_name}.nii.gz"
        mask_npz_path = Path(self.src_path) / "masks" / f"{case_name}.npz"
        if self.load_masks and (mask_nifti_path.exists() or mask_npz_path.exists()):
            if mask_nifti_path.exists():
                nifti_image = nib.loadsave.load(mask_nifti_path)
                if not isinstance(nifti_image, nib.nifti1.Nifti1Image):
                    raise TypeError
                mask_numpy = np.asanyarray(nifti_image.dataobj, dtype=np.uint8)
            else:
                with np.load(mask_npz_path) as mask_archive:
                    if "mask" not in mask_archive.files:
                        msg = f"Compressed mask archive missing 'mask' array: {mask_npz_path}"
                        raise KeyError(msg)
                    mask_numpy = np.array(mask_archive["mask"], dtype=np.uint8)
            mask_tensor = torch.from_numpy(mask_numpy)
            mask_tensor = mask_tensor.permute(*reversed(range(mask_tensor.ndim)))
            mask_tensor_resampled = resample_mask(mask_tensor, self.patch_size)
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
                    [
                        torch.full(sample["classes"].size()[:1], idx)
                        for idx, sample in enumerate(batch)
                    ],
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
        persistent_workers=number_workers > 0,
    )
