from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from radyolo.data.normalization import NormalizationConfig, normalize_image

if TYPE_CHECKING:
    from collections.abc import Sequence

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


# Every spatial input dimension of the model has to be a multiple of its coarsest stride.
# `HardcodedModel.size_divisor` is derived from the built model and has to equal this value.
INPUT_SIZE_DIVISOR = 32


def round_up(value: int, multiple: int) -> int:
    """Return the smallest multiple of ``multiple`` that is greater than or equal to ``value``."""
    return math.ceil(value / multiple) * multiple


def round_up_to_multiple(
    shape: Sequence[int],
    multiple: int = INPUT_SIZE_DIVISOR,
) -> tuple[int, ...]:
    """Round each dimension of ``shape`` up to the next multiple of ``multiple``."""
    return tuple(round_up(int(dim), multiple) for dim in shape)


def pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int = INPUT_SIZE_DIVISOR,
    *,
    value: float = 0.0,
    target_shape: Sequence[int] | None = None,
) -> torch.Tensor:
    """Pad the last three (spatial) dimensions at their high end.

    The tensor is padded to ``target_shape`` if given, otherwise to the next multiple of
    ``multiple``. Padding only ever appends voxels behind the existing content, so voxel
    coordinates of the original content (and therefore box coordinates) stay unchanged.
    """
    spatial_shape = tuple(int(dim) for dim in tensor.shape[-3:])
    padded_shape = (
        tuple(int(dim) for dim in target_shape)
        if target_shape is not None
        else round_up_to_multiple(spatial_shape, multiple)
    )
    if any(padded < current for padded, current in zip(padded_shape, spatial_shape, strict=True)):
        msg = f"Cannot pad shape {spatial_shape} to smaller shape {padded_shape}."
        raise ValueError(msg)
    if padded_shape == spatial_shape:
        return tensor
    # F.pad expects (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
    pad = []
    for padded, current in zip(reversed(padded_shape), reversed(spatial_shape), strict=True):
        pad.extend((0, padded - current))
    return F.pad(tensor, tuple(pad), mode="constant", value=value)


def fit_shape_within(
    source_shape: Sequence[int],
    max_shape: Sequence[int],
) -> tuple[int, int, int]:
    """Return ``source_shape`` or, if it exceeds ``max_shape``, a uniformly downscaled version."""
    scale = min(1.0, *(limit / dim for dim, limit in zip(source_shape, max_shape, strict=True)))
    if scale >= 1.0:
        return cast("tuple[int, int, int]", tuple(int(dim) for dim in source_shape))
    return cast(
        "tuple[int, int, int]",
        tuple(
            max(1, min(int(limit), round(dim * scale)))
            for dim, limit in zip(source_shape, max_shape, strict=True)
        ),
    )


def resolve_numpy_dtype(dtype_name: ImageArrayDType | str) -> np.dtype:
    """Return a NumPy dtype for a configured image dtype name."""
    try:
        return _NUMPY_DTYPES[cast("ImageArrayDType", dtype_name.lower())]
    except KeyError as exc:
        msg = f"Unsupported image dtype '{dtype_name}'. Supported: {sorted(_NUMPY_DTYPES)}"
        raise ValueError(msg) from exc


def resolve_torch_dtype(dtype_name: ImageArrayDType | str) -> torch.dtype:
    """Return a torch dtype for a configured image dtype name."""
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
    max_shape: PatchSize | None = None,
) -> tuple[int, int, int]:
    """Resolve output ZYX shape from an explicit target shape or voxel spacing.

    ``max_shape`` (variable input size) is applied last: the resolved shape is kept unless it
    exceeds ``max_shape``, in which case it is uniformly downscaled to fit.
    """
    if target_shape is not None and (target_voxel_size is not None or max_shape is not None):
        msg = "target_shape cannot be combined with target_voxel_size or max_shape."
        raise ValueError(msg)

    if target_shape is not None:
        return target_shape.get_zyx()

    if target_voxel_size is not None and max_shape is not None:
        resolved = resolve_target_shape(
            source_shape,
            input_voxel_size=input_voxel_size,
            target_voxel_size=target_voxel_size,
        )
        return fit_shape_within(resolved, max_shape.get_zyx())

    if max_shape is not None:
        return fit_shape_within(source_shape, max_shape.get_zyx())

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
    max_shape: PatchSize | None = None,
    apply_resampling: bool = True,
    apply_normalization: bool = True,
    output_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, tuple[int, int, int], tuple[int, int, int]]:
    """Optionally resample and normalize an image, returning source and destination shapes."""
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
            max_shape=max_shape,
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
    """Resample a segmentation mask to target ZYX shape with nearest-neighbor interpolation."""
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
