from __future__ import annotations

import json
import logging
import math
import os
import random
from collections.abc import Sized
from typing import TYPE_CHECKING, Literal, TypedDict

import blosc2
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from radyolo.config.spatial import PatchSize
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
from radyolo.data.data_splits import DEFAULT_SPLIT_FILE_NAME, get_split_file_path
from radyolo.data.preprocessing import preprocess_image, resample_mask
from radyolo.utils.ops import xyzwhd2xyzxyz, xyzxyz2xyzwhd

if TYPE_CHECKING:
    from pathlib import Path

    from radyolo.config.config import DefaultConfig
    from radyolo.data.normalization import NormalizationConfig

LOGGER = logging.getLogger(__name__)
PATCHWISE_TRAIN_EXTRACTION_SCALE = 1.2


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

def _get_other_augmentation_placement(
    placement: Literal["cpu", "gpu"],
) -> Literal["cpu", "gpu"]:
    return "gpu" if placement == "cpu" else "cpu"


def _get_tensor_spatial_shape_xyz(tensor: torch.Tensor) -> tuple[int, int, int]:
    return tuple(int(dim) for dim in reversed(tensor.shape[-3:]))


def _center_crop_tensor_spatial(
    tensor: torch.Tensor,
    crop_origin_xyz: tuple[int, int, int],
    crop_shape_xyz: tuple[int, int, int],
) -> torch.Tensor:
    crop_origin_zyx = tuple(reversed(crop_origin_xyz))
    crop_shape_zyx = tuple(reversed(crop_shape_xyz))
    slices = [slice(None)] * tensor.ndim
    for axis, (origin, size) in enumerate(zip(crop_origin_zyx, crop_shape_zyx, strict=True)):
        slices[tensor.ndim - 3 + axis] = slice(origin, origin + size)
    return tensor[tuple(slices)]


def _center_crop_sample_to_patch(
    image: torch.Tensor,
    classes: torch.Tensor,
    bboxes: torch.Tensor,
    masks: torch.Tensor | None,
    target_patch_shape_xyz: tuple[int, int, int],
    *,
    min_box_crop_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    current_patch_shape_xyz = _get_tensor_spatial_shape_xyz(image)
    if current_patch_shape_xyz == target_patch_shape_xyz:
        return image, classes, bboxes, masks

    if any(
        target_dim > current_dim
        for target_dim, current_dim in zip(
            target_patch_shape_xyz,
            current_patch_shape_xyz,
            strict=True,
        )
    ):
        msg = (
            "Center crop target patch size must not exceed current patch size, got "
            f"current={current_patch_shape_xyz} and target={target_patch_shape_xyz}."
        )
        raise ValueError(msg)

    crop_origin_xyz = tuple(
        (current_dim - target_dim) // 2
        for current_dim, target_dim in zip(
            current_patch_shape_xyz,
            target_patch_shape_xyz,
            strict=True,
        )
    )
    image = _center_crop_tensor_spatial(image, crop_origin_xyz, target_patch_shape_xyz)
    if masks is not None:
        masks = _center_crop_tensor_spatial(masks, crop_origin_xyz, target_patch_shape_xyz)

    classes, bboxes, keep = _crop_boxes_to_patch_with_keep(
        classes,
        bboxes,
        current_patch_shape_xyz,
        crop_origin_xyz,
        target_patch_shape_xyz,
        min_box_crop_fraction=min_box_crop_fraction,
    )
    masks = _filter_instance_masks(masks, keep)
    return image, classes, bboxes, masks


def _should_center_crop_after_augmentations(
    config: DefaultConfig,
    placement: Literal["cpu", "gpu"],
) -> bool:
    if placement == "gpu":
        return True
    return not _has_spatial_augmentations_on_device(
        config,
        _get_other_augmentation_placement(placement),
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
    if _should_center_crop_after_augmentations(config, placement):
        image, classes, bboxes, masks = _center_crop_sample_to_patch(
            image,
            classes,
            bboxes,
            masks,
            patch_size.get_xyz(),
            min_box_crop_fraction=config.min_box_crop_fraction,
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
    images_out: list[torch.Tensor] = []
    classes_out: list[torch.Tensor] = []
    bboxes_out: list[torch.Tensor] = []
    batch_idx_out: list[torch.Tensor] = []
    masks_out: list[torch.Tensor | None] = []
    shapes_out: list[tuple[int, int, int]] = []
    ratio_pad_out: list[tuple[tuple[float, float, float], tuple[int, int, int]] | None] = []

    for sample_idx in range(images.shape[0]):
        image_in = images[sample_idx]
        sample_mask = batch["batch_idx"] == sample_idx
        sample_classes = batch["cls"][sample_mask]
        sample_bboxes = batch["bboxes"][sample_mask]
        sample_masks = batch["masks"][sample_idx]

        image_aug, classes_aug, bboxes_aug, masks_aug = _apply_configured_augmentations_to_sample(
            image_in,
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
        images_out.append(image_aug)
        masks_out.append(masks_aug)
        shapes_out.append(tuple(int(dim) for dim in image_aug.shape[-3:]))

        ratio_pad = batch["ratio_pad"][sample_idx]
        if ratio_pad is not None:
            input_patch_shape_xyz = _get_tensor_spatial_shape_xyz(image_in)
            output_patch_shape_xyz = _get_tensor_spatial_shape_xyz(image_aug)
            crop_origin_xyz = tuple(
                (input_dim - output_dim) // 2
                for input_dim, output_dim in zip(
                    input_patch_shape_xyz,
                    output_patch_shape_xyz,
                    strict=True,
                )
            )
            ratio, pad = ratio_pad
            ratio_pad = (
                ratio,
                tuple(
                    int(axis_pad) - crop
                    for axis_pad, crop in zip(pad, crop_origin_xyz, strict=True)
                ),
            )
        ratio_pad_out.append(ratio_pad)

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

    batch["img"] = torch.stack(images_out)
    batch["shape"] = tuple(shapes_out)
    batch["ratio_pad"] = tuple(ratio_pad_out)
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


def _get_spatial_axis_offset(array_shape: tuple[int, ...]) -> int:
    return 1 if len(array_shape) == 4 and array_shape[0] == 1 else 0


def _get_spatial_shape(array_shape: tuple[int, ...]) -> tuple[int, int, int]:
    spatial_axis_offset = _get_spatial_axis_offset(array_shape)
    spatial_shape = array_shape[spatial_axis_offset : spatial_axis_offset + 3]
    if len(spatial_shape) != 3:
        msg = f"Expected at least three spatial dimensions, got shape {array_shape}"
        raise ValueError(msg)
    return tuple(int(dim) for dim in spatial_shape)


def _compute_patch_origin(
    patch_center_xyz: tuple[int, int, int],
    patch_shape_xyz: tuple[int, int, int],
) -> tuple[int, int, int]:
    return tuple(
        int(center) - int(size) // 2
        for center, size in zip(patch_center_xyz, patch_shape_xyz, strict=True)
    )


def _compute_patch_layout(
    patch_origin_xyz: tuple[int, int, int],
    patch_shape_xyz: tuple[int, int, int],
    image_shape_xyz: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    source_start_xyz = []
    source_end_xyz = []
    destination_start_xyz = []
    destination_end_xyz = []

    for origin, patch_dim, image_dim in zip(
        patch_origin_xyz,
        patch_shape_xyz,
        image_shape_xyz,
        strict=True,
    ):
        source_start = max(origin, 0)
        source_end = min(origin + patch_dim, image_dim)
        destination_start = max(-origin, 0)
        destination_end = destination_start + max(source_end - source_start, 0)

        source_start_xyz.append(source_start)
        source_end_xyz.append(source_end)
        destination_start_xyz.append(destination_start)
        destination_end_xyz.append(destination_end)

    return (
        tuple(source_start_xyz),
        tuple(source_end_xyz),
        tuple(destination_start_xyz),
        tuple(destination_end_xyz),
    )


def _extract_patch_from_array(
    array: np.ndarray,
    patch_origin_xyz: tuple[int, int, int],
    patch_shape_xyz: tuple[int, int, int],
    *,
    fill_value: float | None = None,
) -> np.ndarray:
    array_shape = tuple(int(dim) for dim in array.shape)
    spatial_axis_offset = _get_spatial_axis_offset(array_shape)
    image_shape_xyz = _get_spatial_shape(array_shape)
    source_start_xyz, source_end_xyz, destination_start_xyz, destination_end_xyz = (
        _compute_patch_layout(patch_origin_xyz, patch_shape_xyz, image_shape_xyz)
    )
    if fill_value is None:
        fill_value = float(array.min())

    patch_shape = list(array_shape)
    patch_shape[spatial_axis_offset : spatial_axis_offset + 3] = list(patch_shape_xyz)
    patch = np.full(tuple(patch_shape), fill_value=fill_value, dtype=array.dtype)

    source_slices = [slice(None)] * array.ndim
    destination_slices = [slice(None)] * patch.ndim
    for axis in range(3):
        source_slices[spatial_axis_offset + axis] = slice(
            source_start_xyz[axis],
            source_end_xyz[axis],
        )
        destination_slices[spatial_axis_offset + axis] = slice(
            destination_start_xyz[axis],
            destination_end_xyz[axis],
        )

    patch[tuple(destination_slices)] = array[tuple(source_slices)]
    return patch


def _resolve_normalization(
    normalization: NormalizationConfig | list[NormalizationConfig] | None,
) -> NormalizationConfig | list[NormalizationConfig]:
    return {"type": "none"} if normalization is None else normalization


def _load_split_cases(
    src_path: Path,
    fold: int,
    group: Literal["train", "val", "test"],
    split_file_name: str = DEFAULT_SPLIT_FILE_NAME,
) -> list[str]:
    splits_file = get_split_file_path(src_path, split_file_name)
    with splits_file.open("r") as file:
        splits = json.load(file)
    if fold >= len(splits):
        msg = f"There are only {len(splits)} folds defined, but fold index {fold} was selected."
        raise ValueError(msg)
    return list(splits[fold][group])


def _get_image_path(
    src_path: Path,
    case_name: str,
    image_format: Literal["nifti", "blosc2"],
) -> Path:
    suffixes = {
        "nifti": ".nii.gz",
        "blosc2": ".b2nd",
    }
    return src_path / "images" / f"{case_name}{suffixes[image_format]}"


def _get_mask_path(
    src_path: Path,
    case_name: str,
    image_format: Literal["nifti", "blosc2"],
) -> Path | None:
    preferred_suffixes = {
        "nifti": (".nii.gz", ".b2nd"),
        "blosc2": (".b2nd", ".nii.gz"),
    }
    masks_dir = src_path / "masks"
    for suffix in preferred_suffixes[image_format]:
        mask_path = masks_dir / f"{case_name}{suffix}"
        if mask_path.exists():
            return mask_path
    return None


def _load_nifti_array(image_path: Path) -> np.ndarray:
    nifti_image = nib.loadsave.load(image_path)
    if not isinstance(nifti_image, nib.nifti1.Nifti1Image):
        raise TypeError
    return np.asanyarray(nifti_image.dataobj, dtype=np.float32)


def _array_to_runtime_tensor(array: np.ndarray) -> torch.Tensor:
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    return tensor.permute(*reversed(range(tensor.ndim)))


def _load_image_array(
    src_path: Path,
    case_name: str,
    image_format: Literal["nifti", "blosc2"],
) -> np.ndarray:
    image_path = _get_image_path(src_path, case_name, image_format)
    if not image_path.exists():
        msg = f"Image path does not exists: {image_path}. Is your dataset format correct?"
        raise OSError(msg)

    if image_format == "nifti":
        return _load_nifti_array(image_path)

    blosc2.set_nthreads(1)
    return np.asarray(blosc2.open(str(image_path), mode="r")[:]).astype(np.float32, copy=False)


def _load_labels(src_path: Path, case_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    label_path = src_path / "labels" / f"{case_name}.txt"
    with label_path.open("r") as file:
        lines = file.read().strip().splitlines()
    if lines:
        string_labels = [line.split() for line in lines if len(line)]
        if any(len(label) not in (0, 7) for label in string_labels):
            msg = f"Labels in wrong fomat in file {label_path}"
            raise ValueError(msg)
        labels = np.array(string_labels, dtype=np.float32)
    else:
        labels = np.zeros((0, 7), dtype=np.float32)
    return torch.from_numpy(labels[:, 0:1]), torch.from_numpy(labels[:, 1:])


def _load_foreground_case_bboxes(
    src_path: Path,
    cases: list[str],
) -> dict[str, torch.Tensor]:
    foreground_case_bboxes: dict[str, torch.Tensor] = {}
    for case_name in cases:
        _classes, bboxes = _load_labels(src_path, case_name)
        if bboxes.shape[0] > 0:
            foreground_case_bboxes[case_name] = bboxes
    return foreground_case_bboxes


def _compute_foreground_oversample_count(
    num_cases: int,
    foreground_oversampling: float,
) -> int:
    if foreground_oversampling <= 0 or num_cases == 0:
        return 0
    return math.ceil((num_cases * foreground_oversampling) / (1.0 - foreground_oversampling))


def _validate_foreground_oversampling(foreground_oversampling: float) -> None:
    if not 0.0 <= foreground_oversampling < 1.0:
        msg = (
            "foreground_oversampling must be in the range [0, 1), "
            f"got {foreground_oversampling}."
        )
        raise ValueError(msg)


def _sample_patch_center_from_bbox(
    bbox: torch.Tensor,
    image_shape_xyz: tuple[int, int, int],
    rng: random.Random | None = None,
) -> tuple[int, int, int]:
    bbox_tensor = bbox.to(dtype=torch.float32).unsqueeze(0)
    image_scale = bbox_tensor.new_tensor((*image_shape_xyz, *image_shape_xyz))
    box_xyzxyz = xyzwhd2xyzxyz(bbox_tensor * image_scale)[0]
    random_generator = random if rng is None else rng

    center_xyz = []
    for axis, image_dim in enumerate(image_shape_xyz):
        lower = float(box_xyzxyz[axis].item())
        upper = float(box_xyzxyz[axis + 3].item())
        lower_idx = max(math.floor(lower), 0)
        upper_idx = min(math.ceil(upper) - 1, image_dim - 1)

        if upper_idx < lower_idx:
            sampled_position = round((lower + upper) / 2.0)
        else:
            sampled_position = random_generator.randint(lower_idx, upper_idx)
        center_xyz.append(min(max(sampled_position, 0), image_dim - 1))

    return tuple(center_xyz)


def _load_mask_array(
    src_path: Path,
    case_name: str,
    image_format: Literal["nifti", "blosc2"],
    *,
    load_masks: bool,
) -> np.ndarray | None:
    if not load_masks:
        return None

    mask_path = _get_mask_path(src_path, case_name, image_format)
    if mask_path is None:
        return None

    if mask_path.suffixes[-2:] == [".nii", ".gz"]:
        nifti_image = nib.loadsave.load(mask_path)
        if not isinstance(nifti_image, nib.nifti1.Nifti1Image):
            raise TypeError
        return np.squeeze(np.asanyarray(nifti_image.dataobj, dtype=np.uint8))

    if mask_path.suffix == ".b2nd":
        blosc2.set_nthreads(1)
        return np.asarray(blosc2.open(str(mask_path), mode="r")[:]).astype(np.uint8, copy=False)

    msg = f"Unsupported mask format: {mask_path}"
    raise ValueError(msg)


def _detect_image_format(
    src_path: Path,
    cases: list[str],
) -> Literal["nifti", "blosc2"]:
    for case_name in cases:
        nifti_path = src_path / "images" / f"{case_name}.nii.gz"
        if nifti_path.exists():
            return "nifti"
        blosc2_path = src_path / "images" / f"{case_name}.b2nd"
        if blosc2_path.exists():
            return "blosc2"
    msg = (
        "Could not detect dataset image format. Expected files in images/ as "
        "<case>.nii.gz or <case>.b2nd for the configured split."
    )
    raise FileNotFoundError(msg)


def _load_case_index(
    src_path: Path,
    fold: int,
    group: Literal["train", "val", "test"],
    split_file_name: str = DEFAULT_SPLIT_FILE_NAME,
) -> tuple[list[str], Literal["nifti", "blosc2"]]:
    cases = _load_split_cases(src_path, fold, group, split_file_name)
    return cases, _detect_image_format(src_path, cases)


def _augment_sample(
    sample: Sample,
    *,
    augment: bool,
    config: DefaultConfig | None,
    patch_size: PatchSize,
) -> Sample:
    if not augment:
        return sample

    original_patch_shape_xyz = _get_tensor_spatial_shape_xyz(sample["image"])
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
        config,
        patch_size,
        placement="cpu",
    )

    sample["image"] = image
    sample["classes"] = classes
    sample["bboxes"] = bboxes
    sample["masks"] = masks
    sample["shape"] = tuple(int(dim) for dim in image.shape[-3:])
    if sample["ratio_pad"] is not None:
        ratio, pad = sample["ratio_pad"]
        cropped_patch_shape_xyz = _get_tensor_spatial_shape_xyz(image)
        crop_origin_xyz = tuple(
            (input_dim - output_dim) // 2
            for input_dim, output_dim in zip(
                original_patch_shape_xyz,
                cropped_patch_shape_xyz,
                strict=True,
            )
        )
        sample["ratio_pad"] = (
            ratio,
            tuple(
                int(axis_pad) - crop
                for axis_pad, crop in zip(pad, crop_origin_xyz, strict=True)
            ),
        )
    return sample


def _load_labels_and_masks(
    src_path: Path,
    case_name: str,
    image_format: Literal["nifti", "blosc2"],
    *,
    load_masks: bool,
    patch_size: PatchSize,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    classes, bboxes = _load_labels(src_path, case_name)
    mask_numpy = _load_mask_array(
        src_path,
        case_name,
        image_format,
        load_masks=load_masks,
    )
    if mask_numpy is None:
        return classes, bboxes, None

    mask_tensor = _array_to_runtime_tensor(mask_numpy)
    return classes, bboxes, resample_mask(mask_tensor, patch_size)


def _collate_samples(batch: list[Sample]) -> Batch:
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


def _sample_patch_center_for_axis(
    image_dim: int,
    patch_dim: int,
    rng: random.Random | None = None,
    crop_dim: int | None = None,
    allow_overhang: bool = False,
) -> int:
    if patch_dim >= image_dim:
        return image_dim // 2

    if allow_overhang:
        if crop_dim is None:
            msg = "crop_dim must be provided when allow_overhang=True"
            raise ValueError(msg)
        if crop_dim > patch_dim:
            msg = f"crop_dim must be <= patch_dim, got crop_dim={crop_dim}, patch_dim={patch_dim}"
            raise ValueError(msg)

        lower_overhang = (patch_dim - crop_dim) // 2
        upper_overhang = patch_dim - crop_dim - lower_overhang
        lower_center = (patch_dim // 2) - lower_overhang
        upper_center = image_dim - (patch_dim - (patch_dim // 2)) + upper_overhang
    else:
        lower_center = patch_dim // 2
        upper_center = image_dim - (patch_dim - (patch_dim // 2))
    random_generator = random if rng is None else rng
    return random_generator.randint(lower_center, upper_center)


def _sample_patch_center(
    image_shape_xyz: tuple[int, int, int],
    patch_shape_xyz: tuple[int, int, int],
    rng: random.Random | None = None,
    crop_shape_xyz: tuple[int, int, int] | None = None,
    allow_overhang: bool = False,
) -> tuple[int, int, int]:
    if allow_overhang and crop_shape_xyz is None:
        msg = "crop_shape_xyz must be provided when allow_overhang=True"
        raise ValueError(msg)

    return tuple(
        _sample_patch_center_for_axis(
            image_dim,
            patch_dim,
            rng,
            crop_dim=None if crop_shape_xyz is None else crop_shape_xyz[axis],
            allow_overhang=allow_overhang,
        )
        for axis, (image_dim, patch_dim) in enumerate(
            zip(image_shape_xyz, patch_shape_xyz, strict=True)
        )
    )


def _get_chunked_blosc2_shape(store_path: Path) -> tuple[int, int, int]:
    blosc2.set_nthreads(1)
    return _get_spatial_shape(
        tuple(int(dim) for dim in blosc2.open(str(store_path), mode="r").shape)
    )


def _get_patchwise_training_extraction_size(patch_size: PatchSize) -> PatchSize:
    return PatchSize(
        x=math.ceil(patch_size.x * PATCHWISE_TRAIN_EXTRACTION_SCALE),
        y=math.ceil(patch_size.y * PATCHWISE_TRAIN_EXTRACTION_SCALE),
        z=math.ceil(patch_size.z * PATCHWISE_TRAIN_EXTRACTION_SCALE),
    )


def _load_chunked_blosc2_patch(
    store_path: Path,
    patch_origin_xyz: tuple[int, int, int],
    patch_shape_xyz: tuple[int, int, int],
    *,
    dtype: np.dtype,
    fill_value: float | None = None,
) -> np.ndarray:
    blosc2.set_nthreads(1)
    store = blosc2.open(str(store_path), mode="r")

    store_shape = tuple(int(dim) for dim in store.shape)
    spatial_axis_offset = _get_spatial_axis_offset(store_shape)
    image_shape_xyz = _get_spatial_shape(store_shape)
    source_start_xyz, source_end_xyz, destination_start_xyz, destination_end_xyz = (
        _compute_patch_layout(patch_origin_xyz, patch_shape_xyz, image_shape_xyz)
    )

    patch_shape = list(store_shape)
    patch_shape[spatial_axis_offset : spatial_axis_offset + 3] = list(patch_shape_xyz)

    source_slices = [slice(None)] * len(store_shape)
    destination_slices = [slice(None)] * len(patch_shape)
    for axis in range(3):
        source_slices[spatial_axis_offset + axis] = slice(
            source_start_xyz[axis],
            source_end_xyz[axis],
        )
        destination_slices[spatial_axis_offset + axis] = slice(
            destination_start_xyz[axis],
            destination_end_xyz[axis],
        )

    source_array = np.asarray(store[tuple(source_slices)], dtype=dtype)
    if fill_value is None:
        fill_value = float(source_array.min())
    patch = np.full(tuple(patch_shape), fill_value=fill_value, dtype=dtype)

    patch[tuple(destination_slices)] = source_array
    return patch


def _crop_boxes_to_patch(
    classes: torch.Tensor,
    bboxes: torch.Tensor,
    image_shape_xyz: tuple[int, int, int],
    patch_origin_xyz: tuple[int, int, int],
    patch_shape_xyz: tuple[int, int, int],
    min_box_crop_fraction: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    cropped_classes, cropped_bboxes, _ = _crop_boxes_to_patch_with_keep(
        classes,
        bboxes,
        image_shape_xyz,
        patch_origin_xyz,
        patch_shape_xyz,
        min_box_crop_fraction=min_box_crop_fraction,
    )
    return cropped_classes, cropped_bboxes


def _crop_boxes_to_patch_with_keep(
    classes: torch.Tensor,
    bboxes: torch.Tensor,
    image_shape_xyz: tuple[int, int, int],
    patch_origin_xyz: tuple[int, int, int],
    patch_shape_xyz: tuple[int, int, int],
    min_box_crop_fraction: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if bboxes.numel() == 0:
        return classes, bboxes, torch.zeros((0,), dtype=torch.bool, device=classes.device)

    image_scale = bboxes.new_tensor((*image_shape_xyz, *image_shape_xyz))
    patch_scale = bboxes.new_tensor((*patch_shape_xyz, *patch_shape_xyz))
    patch_origin = bboxes.new_tensor((*patch_origin_xyz, *patch_origin_xyz))
    patch_upper = bboxes.new_tensor(patch_shape_xyz)

    boxes_xyzxyz = xyzwhd2xyzxyz(bboxes * image_scale)
    original_extents = (boxes_xyzxyz[:, 3:6] - boxes_xyzxyz[:, 0:3]).clamp(min=0)
    original_volume = torch.prod(original_extents, dim=1)

    boxes_xyzxyz = boxes_xyzxyz - patch_origin
    boxes_xyzxyz[:, 0:3] = boxes_xyzxyz[:, 0:3].clamp(min=0)
    boxes_xyzxyz[:, 3:6] = torch.minimum(boxes_xyzxyz[:, 3:6], patch_upper)

    cropped_extents = (boxes_xyzxyz[:, 3:6] - boxes_xyzxyz[:, 0:3]).clamp(min=0)
    cropped_volume = torch.prod(cropped_extents, dim=1)
    contained_fraction = cropped_volume / original_volume.clamp_min(torch.finfo(bboxes.dtype).eps)

    keep = torch.all(boxes_xyzxyz[:, 3:6] > boxes_xyzxyz[:, 0:3], dim=1)
    keep &= contained_fraction >= min_box_crop_fraction
    if not torch.any(keep):
        return classes.new_empty((0, 1)), bboxes.new_empty((0, 6)), keep

    cropped_boxes = xyzxyz2xyzwhd(boxes_xyzxyz) / patch_scale
    return classes[keep], cropped_boxes[keep], keep


def _filter_instance_masks(
    masks: torch.Tensor | None,
    keep: torch.Tensor,
) -> torch.Tensor | None:
    if masks is None:
        return None
    if masks.shape[0] != keep.shape[0]:
        return masks
    filtered_masks = masks[keep]
    if filtered_masks.shape[0] == 0:
        return None
    return filtered_masks


def _load_patch_mask(
    src_path: Path,
    case_name: str,
    image_format: Literal["nifti", "blosc2"],
    patch_origin_xyz: tuple[int, int, int],
    patch_shape_xyz: tuple[int, int, int],
    *,
    load_masks: bool,
) -> torch.Tensor | None:
    mask_path = _get_mask_path(src_path, case_name, image_format)
    if not load_masks or mask_path is None:
        return None

    if mask_path.suffix == ".b2nd":
        patch_mask = _load_chunked_blosc2_patch(
            mask_path,
            patch_origin_xyz,
            patch_shape_xyz,
            dtype=np.uint8,
        )
    else:
        mask_numpy = _load_mask_array(
            src_path,
            case_name,
            image_format,
            load_masks=load_masks,
        )
        if mask_numpy is None:
            return None
        patch_mask = _extract_patch_from_array(
            mask_numpy,
            patch_origin_xyz,
            patch_shape_xyz,
            fill_value=0,
        )

    mask_tensor = _array_to_runtime_tensor(patch_mask).to(dtype=torch.uint8)
    if mask_tensor.ndim == 3:
        mask_tensor = mask_tensor.unsqueeze(0)
    return mask_tensor


class WholeImageDataset(Dataset[Sample], Sized):
    def __init__(  # noqa: PLR0913
        self,
        src_path: Path,
        patch_size: PatchSize,
        augment: bool = False,
        min_stride: int = 8,
        fold: int = 0,
        group: Literal["train", "val", "test"] = "train",
        normalization: NormalizationConfig | list[NormalizationConfig] | None = None,
        apply_preprocessing: bool = True,
        load_masks: bool = True,
        foreground_oversampling: float = 0.0,
        min_box_crop_fraction: float = 0.5,
        split_file_name: str | None = None,
        config: DefaultConfig | None = None,
    ) -> None:
        super().__init__()
        _validate_foreground_oversampling(foreground_oversampling)
        self.src_path = src_path
        self.patch_size = patch_size
        self.augment = augment
        self.min_stride = min_stride
        self.fold = fold
        self.group = group
        self.normalization = _resolve_normalization(normalization)
        self.apply_preprocessing = apply_preprocessing
        self.config = config
        self.load_masks = load_masks
        self.foreground_oversampling = foreground_oversampling
        self.min_box_crop_fraction = min_box_crop_fraction
        self.split_file_name = (
            split_file_name
            or (config.split_file_name if config is not None else DEFAULT_SPLIT_FILE_NAME)
        )
        self.cases, self.image_format = _load_case_index(
            self.src_path,
            fold,
            group,
            self.split_file_name,
        )
        self.foreground_case_bboxes = _load_foreground_case_bboxes(self.src_path, self.cases)
        self.foreground_cases = tuple(self.foreground_case_bboxes)
        self.foreground_oversample_count = _compute_foreground_oversample_count(
            len(self.cases),
            foreground_oversampling if self.foreground_cases else 0.0,
        )
        if foreground_oversampling > 0 and not self.foreground_cases:
            LOGGER.warning(
                "WholeImageDataset foreground oversampling requested for split '%s', "
                "but no foreground labels were found. Falling back to one sample per case.",
                group,
            )

    def _get_image_path(self, case_name: str) -> Path:
        return _get_image_path(self.src_path, case_name, self.image_format)

    def _get_mask_path(self, case_name: str) -> Path | None:
        return _get_mask_path(self.src_path, case_name, self.image_format)

    @staticmethod
    def _load_nifti_array(image_path: Path) -> np.ndarray:
        return _load_nifti_array(image_path)

    @staticmethod
    def _array_to_runtime_tensor(array: np.ndarray) -> torch.Tensor:
        return _array_to_runtime_tensor(array)

    def _load_image_array(self, case_name: str) -> np.ndarray:
        return _load_image_array(self.src_path, case_name, self.image_format)

    def _load_labels(self, case_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        return _load_labels(self.src_path, case_name)

    def _load_mask_array(self, case_name: str) -> np.ndarray | None:
        return _load_mask_array(
            self.src_path,
            case_name,
            self.image_format,
            load_masks=self.load_masks,
        )

    def _detect_image_format(self) -> Literal["nifti", "blosc2"]:
        return _detect_image_format(self.src_path, self.cases)

    def __len__(self) -> int:
        return len(self.cases) + self.foreground_oversample_count

    def _sample_foreground_case(self) -> str:
        return random.choice(self.foreground_cases)

    def _resolve_case_name(self, index: int) -> str:
        dataset_length = len(self)
        if index < 0 or index >= dataset_length:
            raise IndexError(index)
        if index < len(self.cases):
            return self.cases[index]
        return self._sample_foreground_case()

    def __getitem__(self, index: int) -> Sample:
        with torch.no_grad():
            return self.augmentation(self.load_image_label_and_mask(index))

    def augmentation(self, sample: Sample) -> Sample:
        return _augment_sample(
            sample,
            augment=self.augment,
            config=self.config,
            patch_size=self.patch_size,
        )

    def load_image_label_and_mask(self, index: int) -> Sample:
        case_name = self._resolve_case_name(index)

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
        image = self._load_image_array(case_name)
        return _array_to_runtime_tensor(image)

    def resample_and_normalize(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int, int], tuple[int, int, int]]:
        image_resampled_normalized, original_shape, resized_shape = preprocess_image(
            image,
            self.normalization,
            target_shape=self.patch_size,
            apply_resampling=self.apply_preprocessing,
            apply_normalization=self.apply_preprocessing,
            output_dtype=torch.float32,
        )
        return image_resampled_normalized, original_shape, resized_shape

    def load_labels_and_masks(
        self, case_name: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return _load_labels_and_masks(
            self.src_path,
            case_name,
            self.image_format,
            load_masks=self.load_masks,
            patch_size=self.patch_size,
        )

    @staticmethod
    def collate_fn(batch: list[Sample]) -> Batch:
        return _collate_samples(batch)


class PatchWiseDataset(Dataset[Sample], Sized):
    def __init__(  # noqa: PLR0913
        self,
        src_path: Path,
        patch_size: PatchSize,
        augment: bool = False,
        min_stride: int = 8,
        fold: int = 0,
        group: Literal["train", "val", "test"] = "train",
        normalization: NormalizationConfig | list[NormalizationConfig] | None = None,
        apply_preprocessing: bool = True,
        load_masks: bool = True,
        foreground_oversampling: float = 0.33,
        min_box_crop_fraction: float = 0.5,
        split_file_name: str | None = None,
        config: DefaultConfig | None = None,
        deterministic_sampling: bool | None = None,
        sampling_seed: int | None = None,
    ) -> None:
        super().__init__()
        _validate_foreground_oversampling(foreground_oversampling)
        self.src_path = src_path
        self.patch_size = patch_size
        self.augment = augment
        self.min_stride = min_stride
        self.fold = fold
        self.group = group
        self.normalization = _resolve_normalization(normalization)
        self.apply_preprocessing = apply_preprocessing
        self.config = config
        self.load_masks = load_masks
        self.foreground_oversampling = foreground_oversampling
        self.min_box_crop_fraction = min_box_crop_fraction
        self.split_file_name = (
            split_file_name
            or (config.split_file_name if config is not None else DEFAULT_SPLIT_FILE_NAME)
        )
        self.extraction_patch_size = (
            _get_patchwise_training_extraction_size(self.patch_size)
            if self.augment
            else self.patch_size
        )
        self.deterministic_sampling = (
            group != "train" and not augment
            if deterministic_sampling is None
            else deterministic_sampling
        )
        self.sampling_seed = (
            int(config.seed)
            if sampling_seed is None and config is not None
            else int(sampling_seed or 0)
        )
        self.cases, self.image_format = _load_case_index(
            self.src_path,
            fold,
            group,
            self.split_file_name,
        )
        self.foreground_case_bboxes = _load_foreground_case_bboxes(self.src_path, self.cases)
        self.foreground_cases = tuple(self.foreground_case_bboxes)
        self.foreground_oversample_count = _compute_foreground_oversample_count(
            len(self.cases),
            foreground_oversampling if self.foreground_cases else 0.0,
        )
        if foreground_oversampling > 0 and not self.foreground_cases:
            LOGGER.warning(
                "PatchWiseDataset foreground oversampling requested for split '%s', "
                "but no foreground labels were found. Falling back to one sample per case.",
                group,
            )

    def __len__(self) -> int:
        return len(self.cases) + self.foreground_oversample_count

    def __getitem__(self, index: int) -> Sample:
        with torch.no_grad():
            return self.augmentation(self.load_image_label_and_mask(index))

    def augmentation(self, sample: Sample) -> Sample:
        return _augment_sample(
            sample,
            augment=self.augment,
            config=self.config,
            patch_size=self.patch_size,
        )

    @staticmethod
    def collate_fn(batch: list[Sample]) -> Batch:
        return _collate_samples(batch)

    def _sample_patch_center(
        self,
        image_shape_xyz: tuple[int, int, int],
        patch_shape_xyz: tuple[int, int, int],
        rng: random.Random | None = None,
    ) -> tuple[int, int, int]:
        return _sample_patch_center(
            image_shape_xyz,
            patch_shape_xyz,
            rng,
            crop_shape_xyz=self.patch_size.get_xyz(),
            allow_overhang=self.augment,
        )

    @staticmethod
    def _sample_patch_center_from_bbox(
        bbox: torch.Tensor,
        image_shape_xyz: tuple[int, int, int],
        rng: random.Random | None = None,
    ) -> tuple[int, int, int]:
        return _sample_patch_center_from_bbox(bbox, image_shape_xyz, rng)

    def _get_index_rng(self, index: int) -> random.Random | None:
        if not self.deterministic_sampling:
            return None

        group_offsets = {"train": 0, "val": 1, "test": 2}
        seed = (
            self.sampling_seed
            + (self.fold * 1_000_003)
            + (group_offsets[self.group] * 10_000_019)
            + (index * 1_000_000_007)
        )
        return random.Random(seed)

    def _sample_foreground_case_and_bbox(
        self,
        rng: random.Random | None = None,
    ) -> tuple[str, torch.Tensor]:
        random_generator = random if rng is None else rng
        case_name = random_generator.choice(self.foreground_cases)
        case_bboxes = self.foreground_case_bboxes[case_name]
        bbox_index = random_generator.randrange(case_bboxes.shape[0])
        return case_name, case_bboxes[bbox_index]

    def _resolve_sampling_target(
        self,
        index: int,
        rng: random.Random | None = None,
    ) -> tuple[str, torch.Tensor | None]:
        dataset_length = len(self)
        if index < 0 or index >= dataset_length:
            raise IndexError(index)
        if index < len(self.cases):
            return self.cases[index], None
        return self._sample_foreground_case_and_bbox(rng)

    def _sample_patch_center_for_index(
        self,
        image_shape_xyz: tuple[int, int, int],
        patch_shape_xyz: tuple[int, int, int],
        rng: random.Random | None,
    ) -> tuple[int, int, int]:
        if rng is None:
            return self._sample_patch_center(image_shape_xyz, patch_shape_xyz)
        return self._sample_patch_center(image_shape_xyz, patch_shape_xyz, rng)

    def _sample_patch_center_from_bbox_for_index(
        self,
        bbox: torch.Tensor,
        image_shape_xyz: tuple[int, int, int],
        rng: random.Random | None,
    ) -> tuple[int, int, int]:
        if rng is None:
            return self._sample_patch_center_from_bbox(bbox, image_shape_xyz)
        return self._sample_patch_center_from_bbox(bbox, image_shape_xyz, rng)

    def _get_chunked_image_shape(self, case_name: str) -> tuple[int, int, int]:
        return _get_chunked_blosc2_shape(
            _get_image_path(self.src_path, case_name, self.image_format)
        )

    @staticmethod
    def _load_chunked_blosc2_patch(
        store_path: Path,
        patch_origin_xyz: tuple[int, int, int],
        patch_shape_xyz: tuple[int, int, int],
        *,
        dtype: np.dtype,
    ) -> np.ndarray:
        return _load_chunked_blosc2_patch(
            store_path,
            patch_origin_xyz,
            patch_shape_xyz,
            dtype=dtype,
            fill_value=None if np.issubdtype(dtype, np.floating) else 0,
        )

    def _load_chunked_image_patch(
        self,
        case_name: str,
        patch_origin_xyz: tuple[int, int, int],
        patch_shape_xyz: tuple[int, int, int],
    ) -> np.ndarray:
        return self._load_chunked_blosc2_patch(
            _get_image_path(self.src_path, case_name, self.image_format),
            patch_origin_xyz,
            patch_shape_xyz,
            dtype=np.float32,
        )

    def _load_chunked_mask_patch(
        self,
        mask_path: Path,
        patch_origin_xyz: tuple[int, int, int],
        patch_shape_xyz: tuple[int, int, int],
    ) -> np.ndarray:
        return self._load_chunked_blosc2_patch(
            mask_path,
            patch_origin_xyz,
            patch_shape_xyz,
            dtype=np.uint8,
            fill_value=0,
        )

    def _crop_boxes_to_patch(
        self,
        classes: torch.Tensor,
        bboxes: torch.Tensor,
        image_shape_xyz: tuple[int, int, int],
        patch_origin_xyz: tuple[int, int, int],
        patch_shape_xyz: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _crop_boxes_to_patch(
            classes,
            bboxes,
            image_shape_xyz,
            patch_origin_xyz,
            patch_shape_xyz,
            min_box_crop_fraction=self.min_box_crop_fraction,
        )

    def _crop_boxes_and_masks_to_patch(
        self,
        classes: torch.Tensor,
        bboxes: torch.Tensor,
        masks: torch.Tensor | None,
        image_shape_xyz: tuple[int, int, int],
        patch_origin_xyz: tuple[int, int, int],
        patch_shape_xyz: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        cropped_classes, cropped_bboxes, keep = _crop_boxes_to_patch_with_keep(
            classes,
            bboxes,
            image_shape_xyz,
            patch_origin_xyz,
            patch_shape_xyz,
            min_box_crop_fraction=self.min_box_crop_fraction,
        )
        return cropped_classes, cropped_bboxes, _filter_instance_masks(masks, keep)

    def _load_patch_mask(
        self,
        case_name: str,
        patch_origin_xyz: tuple[int, int, int],
        patch_shape_xyz: tuple[int, int, int],
    ) -> torch.Tensor | None:
        return _load_patch_mask(
            self.src_path,
            case_name,
            self.image_format,
            patch_origin_xyz,
            patch_shape_xyz,
            load_masks=self.load_masks,
        )

    def load_image_label_and_mask(self, index: int) -> Sample:
        rng = self._get_index_rng(index)
        case_name, foreground_bbox = self._resolve_sampling_target(index, rng)
        patch_shape_xyz = self.extraction_patch_size.get_xyz()

        if self.image_format == "nifti":
            image_array = _load_image_array(self.src_path, case_name, self.image_format)
            image_shape_xyz = _get_spatial_shape(tuple(int(dim) for dim in image_array.shape))
            if foreground_bbox is None:
                patch_center_xyz = self._sample_patch_center_for_index(
                    image_shape_xyz,
                    patch_shape_xyz,
                    rng,
                )
            else:
                patch_center_xyz = self._sample_patch_center_from_bbox_for_index(
                    foreground_bbox,
                    image_shape_xyz,
                    rng,
                )
            patch_origin_xyz = _compute_patch_origin(patch_center_xyz, patch_shape_xyz)
            image_patch = _extract_patch_from_array(
                image_array,
                patch_origin_xyz,
                patch_shape_xyz,
            )
        else:
            image_shape_xyz = self._get_chunked_image_shape(case_name)
            if foreground_bbox is None:
                patch_center_xyz = self._sample_patch_center_for_index(
                    image_shape_xyz,
                    patch_shape_xyz,
                    rng,
                )
            else:
                patch_center_xyz = self._sample_patch_center_from_bbox_for_index(
                    foreground_bbox,
                    image_shape_xyz,
                    rng,
                )
            patch_origin_xyz = _compute_patch_origin(patch_center_xyz, patch_shape_xyz)
            image_patch = self._load_chunked_image_patch(
                case_name,
                patch_origin_xyz,
                patch_shape_xyz,
            )

        image_tensor = _array_to_runtime_tensor(image_patch)
        image_tensor, _, resized_shape = preprocess_image(
            image_tensor,
            self.normalization,
            apply_normalization=self.apply_preprocessing,
            output_dtype=torch.float32,
        )
        classes, bboxes = _load_labels(self.src_path, case_name)
        masks = self._load_patch_mask(case_name, patch_origin_xyz, patch_shape_xyz)
        classes, bboxes, masks = self._crop_boxes_and_masks_to_patch(
            classes,
            bboxes,
            masks,
            image_shape_xyz,
            patch_origin_xyz,
            patch_shape_xyz,
        )

        original_shape = tuple(reversed(image_shape_xyz))
        ratio = (1.0, 1.0, 1.0)
        pad = tuple(-int(origin) for origin in patch_origin_xyz)

        return Sample(
            image=image_tensor,
            shape=resized_shape,
            original_shape=original_shape,
            ratio_pad=(ratio, pad),
            classes=classes,
            bboxes=bboxes,
            masks=masks,
            case_name=case_name,
        )


def build_dataloader(
    dataset: Dataset[Sample],
    batch_size: int,
    workers: int,
    shuffle: bool = True,
    device: torch.device | str = "cpu",
    sampler: Sampler | None = None,
) -> DataLoader:
    """Return an DataLoader for training or validation set."""
    batch_size = min(batch_size, len(dataset))
    number_workers = min(os.cpu_count() or 1, workers)
    multiprocessing_context = None
    if number_workers > 0 and os.name == "posix":
        multiprocessing_context = "spawn"
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=number_workers,
        pin_memory=(device.type if isinstance(device, torch.device) else device) != "cpu",
        collate_fn=getattr(dataset, "collate_fn", None),
        persistent_workers=number_workers > 0,
        multiprocessing_context=multiprocessing_context,
    )
