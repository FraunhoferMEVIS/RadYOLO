from __future__ import annotations

from yolo11_3d.ultralytics.utils.ops import xyzwhd2xyzxyz

__author__ = "kgeissler"

import math
import numpy as np
import random
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from batchgeneratorsv2.transforms.intensity.brightness import (
    MultiplicativeBrightnessTransform, BrightnessAdditiveTransform
)
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform

if TYPE_CHECKING:
    from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


def bbox_iov(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Returns the intersection over box2 volume given box1, box2. Boxes are z1x1y1z2x2y2.

    box1:       np.array of shape(6)
    box2:       np.array of shape(nx6)
    returns:    np.array of shape(n)
    """
    box2 = box2.T

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_z1, b1_x2, b1_y2, b1_z2 = box1[0], box1[1], box1[2], box1[3], box1[4], box1[5]
    b2_x1, b2_y1, b2_z1, b2_x2, b2_y2, b2_z2 = box2[0], box2[1], box2[2], box2[3], box2[4], box2[5]

    # Intersection volume
    inter_vol = (
        (torch.minimum(b1_x2, b2_x2) - torch.maximum(b1_x1, b2_x1)).clip(0)
        * (torch.minimum(b1_y2, b2_y2) - torch.maximum(b1_y1, b2_y1)).clip(0)
        * (torch.minimum(b1_z2, b2_z2) - torch.maximum(b1_z1, b2_z1)).clip(0)
    )

    # box2 vol
    box2_vol = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) * (b2_z2 - b2_z1) + eps

    # Intersection over box2 vol
    return inter_vol.div(box2_vol)


def tensor_cutout(
    im: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor | None,
    cutout_params: list[tuple[float, int]],
    p: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor | None] | None]:
    """Applies image cutout augmentation https://arxiv.org/abs/1708.04552.

    Args:
        im (torch.Tensor): 3-D tensor to be augmented.
        labels (np.array): Med YOLO labels corresponding to im. class z1 x1 y1 z2 x2 y2
        cutout_params (list[tuple[float, int]]): a list of ordered pairs that set sizes and numbers of cutout boxes.
                                           the maximum extent of the boxes is set by the first element of the ordered pair
                                           the number of boxes with that maximum extent is set by the second element of the ordered pair
        p (float, optional): probability of performing cutout augmentation. Defaults to 0.5.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple of the augmented image and labels.
    """
    if random.random() >= p:
        return im, labels, masks

    d, h, w = im.shape[1:]

    image_min = torch.min(im).item()
    image_max = torch.max(im).item()
    for scale in (scale for scale, num_repeats in cutout_params for _ in range(num_repeats)):
        mask_d = random.randint(1, int(d * scale))  # create random masks
        mask_h = random.randint(1, int(h * scale))
        mask_w = random.randint(1, int(w * scale))

        # box
        zmin = max(0, random.randint(0, d) - mask_d // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        zmax = min(d, zmin + mask_d)
        ymax = min(h, ymin + mask_h)
        xmax = min(w, xmin + mask_w)

        # apply random greyscale mask
        # images scaled between 0 and 1 after being returned by the dataset
        im[:, zmin:zmax, ymin:ymax, xmin:xmax] = random.uniform(image_min, image_max)

        # remove obscured labels
        if len(labels) and scale > 0.03:
            box = torch.as_tensor(
                [xmin, ymin, zmin, xmax, ymax, zmax], dtype=labels.dtype, device=labels.device
            )
            iov = bbox_iov(box, labels[:, 1:7])  # intersection over volume
            keep_mask = iov < 0.60
            labels = labels[keep_mask]  # remove >60% obscured labels
            if masks is not None:
                masks = masks[keep_mask]

    return im, labels, masks

def random_zoom(
    image: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor | None,
    max_zoom: float = 1.5,
    min_zoom: float = 0.7,
    p: float = 0.5,
    out_zoom_padding: str = "random",
    uniform_zoom: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor | None] | None]:
    """Randomly zooms in or out of a random part of the input image.

    Args:
        image (torch.Tensor): 3-D tensor to be augmented.
        labels (List[float]): MedYOLO labels corresponding to image. class z1 x1 y1 z2 x2 y2
        max_zoom (float, optional): maximum edge length multiplier. Defaults to 1.5.
        min_zoom (float, optional): minimum edge length multiplier. Defaults to 0.7.
        p (float, optional): probability of zooming the input. Defaults to 0.5.
        out_zoom_padding (str): Padding strategy when zooming out ("random" or "constant").
        uniform_zoom (bool): If True (default) a single zoom factor is sampled and applied to all
            spatial dimensions. If False, an independent zoom factor is sampled for each spatial
            dimension (depth, height, width) uniformly from [min_zoom, max_zoom].

    Returns:
        image: Augmented tensor.
        augmented_labels: Adjusted labels.
    """
    if random.random() >= p:
        return image, labels, masks

    augmented_labels = labels.clone()

    # retrieve original image shape (this is resized to imgsz_z x imgsz_y x imgsz_x by this point in the dataloader)
    channels, d, h, w = image.shape
    if uniform_zoom:
        zoom_factors = (
            random.uniform(min_zoom, max_zoom),
        ) * 3  # (d, h, w) identical factors
    else:
        zoom_factors = (
            random.uniform(min_zoom, max_zoom),  # depth
            random.uniform(min_zoom, max_zoom),  # height
            random.uniform(min_zoom, max_zoom),  # width
        )

    # add batch dimension for functional interpolate
    image = torch.nn.functional.interpolate(
        image.unsqueeze(0), scale_factor=zoom_factors, mode="trilinear", align_corners=False
    ).squeeze_(0)

    if masks is not None:
        masks = masks.unsqueeze(0)
        masks = F.interpolate(masks, scale_factor=zoom_factors, mode="nearest-exact").squeeze(0)

    # retrieve new image shape
    new_d, new_h, new_w = image.shape[1:]

    # scale labels per dimension (labels are in xyzxyz format after concatenation)
    # order: x1 y1 z1 x2 y2 z2
    zoom_w, zoom_h, zoom_d = zoom_factors[2], zoom_factors[1], zoom_factors[0]
    augmented_labels[:, [1, 4]] *= zoom_w  # x coordinates
    augmented_labels[:, [2, 5]] *= zoom_h  # y coordinates
    augmented_labels[:, [3, 6]] *= zoom_d  # z coordinates

    # Generic crop/pad handling per dimension
    orig_sizes = (d, h, w)
    new_sizes = (new_d, new_h, new_w)
    # For each dimension build slicing / padding indices
    crop_starts: list[int] = []  # start index in resized image to take (if cropping)
    pad_starts: list[int] = []   # start index in final canvas to place resized (if padding)
    final_indices: list[slice] = []
    cropped_dims = []  # keep track which dims were cropped (for filtering labels later)
    for dim, (orig, new) in enumerate(zip(orig_sizes, new_sizes)):
        if new >= orig:  # crop
            max_start = new - orig
            start = 0 if max_start == 0 else random.randint(0, max_start)
            crop_starts.append(start)
            pad_starts.append(0)
            final_indices.append(slice(start, start + orig))
            cropped_dims.append(dim)
        else:  # pad
            # choose random placement within original canvas
            max_start = orig - new
            start = 0 if max_start == 0 else random.randint(0, max_start)
            crop_starts.append(0)
            pad_starts.append(start)
            final_indices.append(slice(0, new))

    # Apply cropping
    image_cropped = image[:, final_indices[0], final_indices[1], final_indices[2]]
    if masks is not None:
        masks_cropped = masks[:, final_indices[0], final_indices[1], final_indices[2]]
    else:
        masks_cropped = None

    # If any padding needed (i.e., any new < orig), create canvas and place
    if any(new < orig for orig, new in zip(orig_sizes, new_sizes)):
        # prepare canvas
        if out_zoom_padding == "random":
            canvas = torch.rand(channels, d, h, w) * (torch.max(image_cropped) - torch.min(image_cropped)) + torch.min(image_cropped)
        elif out_zoom_padding == "constant":
            canvas = torch.zeros(channels, d, h, w) + torch.min(image_cropped)
        else:
            raise ValueError(f"Unknown out zoom padding mode {out_zoom_padding}")
        z_start, y_start, x_start = pad_starts
        z_slice = slice(z_start, z_start + image_cropped.shape[1])
        y_slice = slice(y_start, y_start + image_cropped.shape[2])
        x_slice = slice(x_start, x_start + image_cropped.shape[3])
        canvas[:, z_slice, y_slice, x_slice] = image_cropped
        image = canvas
        if masks_cropped is not None:
            mask_canvas = torch.zeros(masks_cropped.shape[0], d, h, w, dtype=masks_cropped.dtype, device=masks_cropped.device)
            mask_canvas[:, z_slice, y_slice, x_slice] = masks_cropped
            masks = mask_canvas
        else:
            masks = None
    else:
        # Only cropping, no padding
        image = image_cropped
        masks = masks_cropped

    # Adjust labels: subtract crop_starts for cropped dims, add pad_starts for padded dims
    # crop_starts list contains either crop start (if cropped) else 0; pad_starts has placement start (0 if cropped)
    # mapping dim indices: 0->z,1->h(y),2->w(x) ; label order x,y,z
    # First subtract crop starts for each axis where cropping occurred
    z_crop, y_crop, x_crop = crop_starts
    z_pad, y_pad, x_pad = pad_starts
    # subtract crop
    augmented_labels[:, [3, 6]] -= z_crop  # z1,z2 at indices 3,6? actually order x1(1) y1(2) z1(3) x2(4) y2(5) z2(6)
    augmented_labels[:, [2, 5]] -= y_crop
    augmented_labels[:, [1, 4]] -= x_crop
    # add padding offsets
    augmented_labels[:, [3, 6]] += z_pad
    augmented_labels[:, [2, 5]] += y_pad
    augmented_labels[:, [1, 4]] += x_pad

    # Filter labels whose centers fall far outside (only necessary for cropped dimensions)
    if augmented_labels.shape[0]:
        label_center_x = (augmented_labels[:, 1] + augmented_labels[:, 4]) / 2
        label_center_y = (augmented_labels[:, 2] + augmented_labels[:, 5]) / 2
        label_center_z = (augmented_labels[:, 3] + augmented_labels[:, 6]) / 2
        # base valid range
        valid = (
            (label_center_x >= -8) & (label_center_x <= w + 8) &
            (label_center_y >= -8) & (label_center_y <= h + 8) &
            (label_center_z >= -8) & (label_center_z <= d + 8)
        )
        augmented_labels = augmented_labels[valid]
        if masks is not None:
            masks = masks[valid]

        # Clamp to image bounds
        augmented_labels[:, 1].clamp_(0, w)  # x1
        augmented_labels[:, 2].clamp_(0, h)  # y1
        augmented_labels[:, 3].clamp_(0, d)  # z1
        augmented_labels[:, 4].clamp_(0, w)  # x2
        augmented_labels[:, 5].clamp_(0, h)  # y2
        augmented_labels[:, 6].clamp_(0, d)  # z2

    return image, augmented_labels, masks


def random_rotation(
    image: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor | None,
    angle_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = ((-20.0, 20.0), (-20.0, 20.0), (-20.0, 20.0)),
    p: float = 0.5,
    mode: str = "bilinear",
    keep_box_volume: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Randomly rotate the 3D volume about its center with independent angles per axis.

    Args:
        image: Tensor (C, D, H, W)
        labels: Tensor (N, 7) with columns [cls, x1, y1, z1, x2, y2, z2] in absolute voxel coords.
        masks: Optional tensor (M, D, H, W) of per-instance masks aligned with image.
        angle_ranges: ((min_x,max_x),(min_y,max_y),(min_z,max_z)) in degrees for rotations about x,y,z axes.
        p: Probability of applying the rotation.
        mode: Interpolation mode for image ("bilinear" or "nearest"). Masks always rotated with nearest.

    Returns:
        Rotated (image, labels, masks).
    """
    if random.random() >= p or (labels.numel() == 0 and masks is None):
        return image, labels, masks

    device = image.device
    dtype = image.dtype
    C, D, H, W = image.shape

    # Sample angles in radians
    ax = random.uniform(*angle_ranges[0]) * math.pi / 180.0
    ay = random.uniform(*angle_ranges[1]) * math.pi / 180.0
    az = random.uniform(*angle_ranges[2]) * math.pi / 180.0

    cx, cy, cz = (W - 1) / 2.0, (H - 1) / 2.0, (D - 1) / 2.0

    # Rotation matrices (x,y,z axes) operating on [x,y,z]
    Rx = torch.tensor([
        [1, 0, 0],
        [0, math.cos(ax), -math.sin(ax)],
        [0, math.sin(ax), math.cos(ax)],
    ], device=device, dtype=dtype)
    Ry = torch.tensor([
        [math.cos(ay), 0, math.sin(ay)],
        [0, 1, 0],
        [-math.sin(ay), 0, math.cos(ay)],
    ], device=device, dtype=dtype)
    Rz = torch.tensor([
        [math.cos(az), -math.sin(az), 0],
        [math.sin(az), math.cos(az), 0],
        [0, 0, 1],
    ], device=device, dtype=dtype)
    R = Rz @ Ry @ Rx  # combined forward rotation

    # For grid_sample we need inverse transform (maps output -> input)
    R_inv = R.T  # rotation matrices are orthonormal so inverse = transpose

    # Rotate image using affine_grid/grid_sample (normalized coordinates in [-1,1])
    affine = torch.zeros(1, 3, 4, device=device, dtype=dtype)
    affine[0, :, :3] = R_inv
    grid = F.affine_grid(affine, size=(1, C, D, H, W), align_corners=True)
    # grid coords order is (x,y,z); our rotation matrix acts about center (0,0,0) in normalized space already
    image_rot = F.grid_sample(
        image.unsqueeze(0),
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)

    masks_rot: torch.Tensor | None
    if masks is not None:
        masks_rot = F.grid_sample(
            masks.unsqueeze(0).float(),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)
        # ensure binary nature retained
        masks_rot = (masks_rot > 0.5).to(masks.dtype)
    else:
        masks_rot = None

    if labels.shape[0] == 0:
        return image_rot, labels, masks_rot

    # Rotate bounding boxes: compute 8 corners, rotate, get new AABB
    x1, y1, z1, x2, y2, z2 = (
        labels[:, 1],
        labels[:, 2],
        labels[:, 3],
        labels[:, 4],
        labels[:, 5],
        labels[:, 6],
    )
    # Corner combination indices
    corner_bits = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
        device=device,
        dtype=dtype,
    )  # (8,3)
    # Widths
    w = (x2 - x1).unsqueeze(1)
    h = (y2 - y1).unsqueeze(1)
    d_ = (z2 - z1).unsqueeze(1)
    # Build corner coords (N,8,3) in (x,y,z)
    base = torch.stack([x1, y1, z1], 1).unsqueeze(1)  # (N,1,3)
    extents = torch.stack([w, h, d_], 2)  # (N,1,3)
    corners = base + corner_bits.unsqueeze(0) * extents  # broadcast

    # Center, rotate, uncenter
    corners_centered = corners.clone()
    corners_centered[..., 0] -= cx
    corners_centered[..., 1] -= cy
    corners_centered[..., 2] -= cz
    # Apply forward rotation (row-vector formulation p' = p * R^T)
    rot_corners = torch.matmul(corners_centered, R.T)
    rot_corners[..., 0] += cx
    rot_corners[..., 1] += cy
    rot_corners[..., 2] += cz

    new_x1 = rot_corners[..., 0].min(dim=1).values.clamp_(0, W - 1)
    new_y1 = rot_corners[..., 1].min(dim=1).values.clamp_(0, H - 1)
    new_z1 = rot_corners[..., 2].min(dim=1).values.clamp_(0, D - 1)
    new_x2 = rot_corners[..., 0].max(dim=1).values.clamp_(0, W - 1)
    new_y2 = rot_corners[..., 1].max(dim=1).values.clamp_(0, H - 1)
    new_z2 = rot_corners[..., 2].max(dim=1).values.clamp_(0, D - 1)

    rotated_labels = labels.clone()
    rotated_labels[:, 1] = new_x1
    rotated_labels[:, 2] = new_y1
    rotated_labels[:, 3] = new_z1
    rotated_labels[:, 4] = new_x2
    rotated_labels[:, 5] = new_y2
    rotated_labels[:, 6] = new_z2

    if keep_box_volume and rotated_labels.shape[0]:
        # Original volumes
        orig_vol = (labels[:, 4] - labels[:, 1]) * (labels[:, 5] - labels[:, 2]) * (labels[:, 6] - labels[:, 3])
        new_vol = (rotated_labels[:, 4] - rotated_labels[:, 1]) * (rotated_labels[:, 5] - rotated_labels[:, 2]) * (rotated_labels[:, 6] - rotated_labels[:, 3])
        eps = 1e-6
        scale = ((orig_vol + eps) / (new_vol + eps)).clamp(min=0) ** (1.0 / 3.0)  # (N,)
        # centers and half-sizes
        cx_b = (rotated_labels[:, 1] + rotated_labels[:, 4]) / 2.0
        cy_b = (rotated_labels[:, 2] + rotated_labels[:, 5]) / 2.0
        cz_b = (rotated_labels[:, 3] + rotated_labels[:, 6]) / 2.0
        hx = (rotated_labels[:, 4] - rotated_labels[:, 1]) / 2.0 * scale
        hy = (rotated_labels[:, 5] - rotated_labels[:, 2]) / 2.0 * scale
        hz = (rotated_labels[:, 6] - rotated_labels[:, 3]) / 2.0 * scale
        # New coords
        nx1 = cx_b - hx
        nx2 = cx_b + hx
        ny1 = cy_b - hy
        ny2 = cy_b + hy
        nz1 = cz_b - hz
        nz2 = cz_b + hz
        # Clamp to image bounds
        nx1.clamp_(0, W - 1)
        nx2.clamp_(0, W - 1)
        ny1.clamp_(0, H - 1)
        ny2.clamp_(0, H - 1)
        nz1.clamp_(0, D - 1)
        nz2.clamp_(0, D - 1)
        # Write back
        rotated_labels[:, 1] = nx1
        rotated_labels[:, 2] = ny1
        rotated_labels[:, 3] = nz1
        rotated_labels[:, 4] = nx2
        rotated_labels[:, 5] = ny2
        rotated_labels[:, 6] = nz2

    # Remove degenerate boxes
    keep = (rotated_labels[:, 4] - rotated_labels[:, 1] > 1) & (rotated_labels[:, 5] - rotated_labels[:, 2] > 1) & (rotated_labels[:, 6] - rotated_labels[:, 3] > 1)
    rotated_labels = rotated_labels[keep]
    if masks_rot is not None:
        masks_rot = masks_rot[keep] if keep.numel() == masks_rot.shape[0] else masks_rot

    return image_rot, rotated_labels, masks_rot


def random_mirroring(
    image: torch.Tensor,
    labels: torch.Tensor,
    masks: list[torch.Tensor | None] | None,
    mirror_axes: list[str] = ["z", "y", "x"],
    p: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor | None] | None]:
    """Applies image mirroring augmentation.

    Args:
        image (torch.Tensor): 3-D tensor to be augmented.
        labels (np.ndarray): MedYOLO labels corresponding to image. class z1 x1 y1 z2 x2 y2
        mirror_axes (List[str]): list of axes to be randomly mirrored
        p (float, optional): probability of performing mirroring augmentation.

    Returns:
        image: augmented image
        labels: augmented labels
    """
    if random.random() >= p:
        return image, labels, masks

    flip_dims = []
    new_labels = labels.clone()

    for axis in mirror_axes:
        if random.random() < 0.5:
            idx = ["x", "y", "z"].index(axis) + 1
            flip_dims.append(4 - idx)
            new_labels[:, idx] = 1 - labels[:, idx]

    if flip_dims:
        image = torch.flip(image, dims=flip_dims)
        if masks is not None:
            masks = torch.flip(masks, dims=flip_dims)

    return image, new_labels, masks


def batch_generators_intensity_augmentations(
    image: torch.Tensor,
    do_intensity_augmentations_per_channel: bool = False,
    noise_prob: float = 0.1,
    noise_variance: tuple[float, float] = (0.0, 1.0),
    blur_prob: float = 0.2,
    blur_sigma: tuple[float, float] = (0.5, 1.0),
    mult_brightness_prob: float = 0.15,
    mult_brightness_range: tuple[float, float] = (0.75, 1.25),
    add_brightness_prob: float = 0.15,
    add_brightness_mu: float = 0.0,
    add_brightness_sigma: float = 2.0,
    contrast_prob: float = 0.15,
    contrast_range: tuple[float, float] = (0.75, 1.25),
    lowres_prob: float = 0.25,
    lowres_scale: tuple[float, float] = (0.5, 1.0),
    gamma_prob: float = 0.4,
    gamma_range: tuple[float, float] = (0.7, 1.5),
    gamma_p_invert: float = 0.25,
    gamma_p_retain_stats: float = 1.0,
) -> torch.Tensor:
    synchronize_channels = not do_intensity_augmentations_per_channel
    transforms: list[BasicTransform] = []
    if noise_prob > 0:
        transforms.append(
            RandomTransform(
                GaussianNoiseTransform(
                    noise_variance=noise_variance, synchronize_channels=synchronize_channels
                ),
                apply_probability=noise_prob,
            )
        )
    if blur_prob > 0:
        transforms.append(
            RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=blur_sigma,
                    synchronize_channels=synchronize_channels,
                    p_per_channel=1,
                ),
                apply_probability=blur_prob,
            )
        )
    if mult_brightness_prob > 0:
        transforms.append(
            RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=mult_brightness_range, synchronize_channels=synchronize_channels
                ),
                apply_probability=mult_brightness_prob,
            )
        )
    if add_brightness_prob > 0:
        transforms.append(
            RandomTransform(
                BrightnessAdditiveTransform(
                    mu=add_brightness_mu,
                    sigma=add_brightness_sigma,
                    per_channel=do_intensity_augmentations_per_channel,
                    p_per_channel=1,
                ),
                apply_probability=add_brightness_prob,
            )
        )
    if contrast_prob > 0:
        transforms.append(
            RandomTransform(
                ContrastTransform(
                    contrast_range=contrast_range,
                    preserve_range=False,
                    synchronize_channels=synchronize_channels,
                ),
                apply_probability=contrast_prob,
            )
        )
    if lowres_prob > 0:
        transforms.append(
            RandomTransform(
                SimulateLowResolutionTransform(
                    scale=lowres_scale,
                    synchronize_channels=synchronize_channels,
                    synchronize_axes=False,
                    ignore_axes=tuple(),
                    p_per_channel=1,
                ),
                apply_probability=lowres_prob,
            )
        )
    if gamma_prob > 0:
        transforms.append(
            RandomTransform(
                GammaTransform(
                    gamma=gamma_range,
                    p_invert_image=gamma_p_invert,
                    synchronize_channels=synchronize_channels,
                    p_per_channel=1,
                    p_retain_stats=gamma_p_retain_stats,
                ),
                apply_probability=gamma_prob,
            )
        )
    if not transforms:
        return image
    composed_transforms = ComposeTransforms(transforms)
    bg_result: dict[str, Any] = composed_transforms(image=image)
    augmented_image: torch.Tensor = bg_result["image"]
    return augmented_image

def enlarge_bounding_boxes(
    image: torch.Tensor,
    labels: torch.Tensor,
    masks: list[torch.Tensor | None] | None,
    imgsz: tuple[int, int, int],
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor | None] | None]:
    """Enlarges bounding box edges to stride size which would be smaller than the stride size.

    Args:
        image (torch.Tensor): 3-D tensor to be augmented.
        labels (torch.Tensor): bounding boxes. class x y z w h d
        imgsz (tuple[int, int, int]): target image size.
        stride (int): stride size.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: augmented image and labels
    """
    if labels.shape[0] == 0:
        return image, labels, masks

    # enlarge bounding boxes to stride size by replacing the w,h,d which are smaller than the stride
    # size with the stride size
    labels[:, 4:] = torch.where(labels[:, 4:] < stride, stride, labels[:, 4:])  # width

    # remove labels that are outside of the image
    # we build a mask that is True for labels that are inside the image
    # for that we convert the xyzwhd bounding boxes to xyzxyz using the function xyzwhd2xyzxyz
    # and then check if the coordinates are within the image size
    bboxes = xyzwhd2xyzxyz(labels[:, 1:7])
    filter = (
        (bboxes[:, 0] >= 0)
        & (bboxes[:, 1] >= 0)
        & (bboxes[:, 2] >= 0)
        & (bboxes[:, 3] < imgsz[2])
        & (bboxes[:, 4] < imgsz[1])
        & (bboxes[:, 5] < imgsz[0])
    )
    labels = labels[filter]
    if masks is not None:
        masks = masks[filter]

    return image, labels, masks


def random_channel_dropout(
    image: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor | None,
    drop_probabilities: list[float] | tuple[float, ...],
    fill_value: float | None = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Randomly drop (zero-out or constant-fill) at most one channel based on per-channel probabilities.

    Args:
        image: (C, D, H, W) tensor.
        labels: (N, 7) unchanged.
        masks: Optional (M, D, H, W) masks (unchanged).
        drop_probabilities: per-channel independent probabilities. Only one channel can be dropped;
            if multiple trigger, one is chosen proportionally to their probabilities.
        p: Global probability to attempt a dropout event at all.
        fill_value: Value to fill the dropped channel with. If None, use channel mean.

    Returns:
        (image, labels, masks) with possibly one channel dropped.
    """
    C = image.shape[0]
    if len(drop_probabilities) != C:
        raise ValueError(f"drop_probabilities length {len(drop_probabilities)} must match channel count {C}")

    # Determine candidate channels
    probs = torch.tensor(drop_probabilities, dtype=torch.float32)
    # Sample Bernoulli for each channel
    bern = torch.bernoulli(probs).bool()
    if not bern.any():
        return image, labels, masks
    # If multiple, choose one weighted by their probabilities
    candidate_indices = torch.nonzero(bern).flatten()
    if candidate_indices.numel() > 1:
        cand_probs = probs[candidate_indices]
        cand_probs = cand_probs / cand_probs.sum()
        chosen_idx = candidate_indices[torch.multinomial(cand_probs, 1)]
    else:
        chosen_idx = candidate_indices[0]

    # Apply dropout: fill with value or channel mean
    if fill_value is None:
        replacement = image[chosen_idx].mean()
    else:
        replacement = torch.as_tensor(fill_value, dtype=image.dtype, device=image.device)
    image[chosen_idx] = replacement
    return image, labels, masks


class GINNet(nn.Module):
    def __init__(
        self,
        hidden_layers: int,
        hidden_channels: int,
        p_kernel_size_1: float = 0.5,
    ):
        super(GINNet, self).__init__()

        self._p_kernel_size_1 = p_kernel_size_1

        self.layers_kernel_size_3 = nn.ModuleList()
        self.layers_kernel_size_1 = nn.ModuleList()
        self.layers_activation_functions = nn.ModuleList()

        layer_dims = (
            [1]
            + [hidden_channels] * (hidden_layers - 1)
            + [1]
        )

        for i in range(len(layer_dims) - 1):
            self.layers_kernel_size_3.append(
                nn.Conv3d(
                    in_channels=layer_dims[i],
                    out_channels=layer_dims[i + 1],
                    kernel_size=3,
                    padding=1,
                )
            )
            self.layers_kernel_size_1.append(
                nn.Conv3d(
                    in_channels=layer_dims[i],
                    out_channels=layer_dims[i + 1],
                    kernel_size=1,
                    padding=0,
                )
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        for layer in self.layers_kernel_size_3:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.normal_(layer.bias)
        for layer in self.layers_kernel_size_1:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.normal_(layer.bias)

        output = torch.zeros_like(input)
        for channel in range(input.shape[0]):
            input_i = x = input[channel].unsqueeze(0)
            for i in range(len(self.layers_kernel_size_3)):
                if random.random() <= self._p_kernel_size_1:
                    x = self.layers_kernel_size_1[i](x)
                else:
                    x = self.layers_kernel_size_3[i](x)
                x = F.leaky_relu(x)
                
            alpha = random.random()
            output_i = alpha * input_i + (1 - alpha) * x
            output_i = output_i * (torch.norm(input_i, p="fro") / (torch.norm(output_i, p="fro") + 1e-6))
            output[channel] = output_i.squeeze(0)

        return output
    
def gin_augmentation(
    image: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor | None,
    hidden_layers: int,
    hidden_channels: int,
    p_kernel_size_1: float = 0.5,
    p: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor | None] | None]:
    """
    Applies GIN (Global Intensity Non-linear) augumentation to input image.
    """
    if random.random() >= p:
        return image, labels, masks
    
    gin_model = GINNet(hidden_layers, hidden_channels, p_kernel_size_1)
    with torch.no_grad():
        augmented_image = gin_model.forward(image)

    return augmented_image, labels, masks