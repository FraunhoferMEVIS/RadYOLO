from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from radyolo.data.normalization import NormalizationConfig, normalize_image

if TYPE_CHECKING:
    from radyolo.config.spatial import (
        PatchSize,
        VoxelSize,
    )

ImageArrayDType = Literal["float32", "float16", "int16"]

_NUMPY_DTYPES: dict[ImageArrayDType, np.dtype] = {
    "float32": np.dtype(np.float32),
    "float16": np.dtype(np.float16),
    "int16": np.dtype(np.int16),
}
_TORCH_DTYPES: dict[ImageArrayDType, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "int16": torch.int16,
}


def resolve_numpy_dtype(dtype_name: ImageArrayDType | str) -> np.dtype:
    try:
        return _NUMPY_DTYPES[cast("ImageArrayDType", dtype_name.lower())]
    except KeyError as exc:
        msg = f"Unsupported image dtype '{dtype_name}'. Supported: {sorted(_NUMPY_DTYPES)}"
        raise ValueError(msg) from exc


def resolve_torch_dtype(dtype_name: ImageArrayDType | str) -> torch.dtype:
    try:
        return _TORCH_DTYPES[cast("ImageArrayDType", dtype_name.lower())]
    except KeyError as exc:
        msg = f"Unsupported image dtype '{dtype_name}'. Supported: {sorted(_TORCH_DTYPES)}"
        raise ValueError(msg) from exc


def resolve_target_shape(
    source_shape: tuple[int, int, int],
    *,
    target_shape: PatchSize | None = None,
    input_voxel_size: VoxelSize | None = None,
    target_voxel_size: VoxelSize | None = None,
) -> tuple[int, int, int]:
    if target_shape is not None and target_voxel_size is not None:
        msg = "Only one of target_shape or target_voxel_size can be specified."
        raise ValueError(msg)

    if target_shape is not None:
        return target_shape.get_zyx()

    if target_voxel_size is not None:
        if input_voxel_size is None:
            msg = "target_voxel_size requires input_voxel_size to be provided."
            raise ValueError(msg)
        return tuple(
            max(1, round(size * source_spacing / target_spacing))
            for size, source_spacing, target_spacing in zip(
                source_shape,
                input_voxel_size.get_zyx(),
                target_voxel_size.get_zyx(),
                strict=True,
            )
        )

    return source_shape


def preprocess_image(
    image: torch.Tensor,
    normalization: NormalizationConfig | list[NormalizationConfig] = {"type": "none"},
    *,
    target_shape: PatchSize | None = None,
    input_voxel_size: VoxelSize | None = None,
    target_voxel_size: VoxelSize | None = None,
    apply_resampling: bool = True,
    apply_normalization: bool = True,
    output_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, tuple[int, int, int], tuple[int, int, int]]:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        msg = f"Expected image with 3 or 4 dimensions, got shape {tuple(image.shape)}"
        raise ValueError(msg)

    source_shape = cast("tuple[int, int, int]", tuple(int(dim) for dim in image.shape[1:]))
    destination_shape = (
        resolve_target_shape(
            source_shape,
            target_shape=target_shape,
            input_voxel_size=input_voxel_size,
            target_voxel_size=target_voxel_size,
        )
        if apply_resampling
        else source_shape
    )

    image = image.to(dtype=torch.float32)
    if destination_shape != source_shape:
        image = cast(
            "torch.Tensor",
            F.interpolate(
                image.unsqueeze(0),
                size=destination_shape,
                mode="trilinear",
                align_corners=False,
            ),
        ).squeeze(0)

    if apply_normalization:
        image = normalize_image(image, normalization)
    if image.dtype != output_dtype:
        image = image.to(dtype=output_dtype)

    return image, source_shape, destination_shape


def resample_mask(mask: torch.Tensor, target_shape: PatchSize) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if mask.ndim != 4:
        msg = f"Expected mask with 3 or 4 dimensions, got shape {tuple(mask.shape)}"
        raise ValueError(msg)

    target_shape_zyx = target_shape.get_zyx()
    source_shape = tuple(int(dim) for dim in mask.shape[1:])
    if source_shape == target_shape_zyx:
        return mask

    resized_mask = cast(
        "torch.Tensor",
        F.interpolate(
            mask.unsqueeze(0),
            size=target_shape_zyx,
            mode="nearest-exact",
        ),
    ).squeeze(0)
    return resized_mask.to(dtype=mask.dtype)
