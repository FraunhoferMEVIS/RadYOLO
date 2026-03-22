from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn.functional as F  # noqa: N812
from batchgeneratorsv2.transforms.intensity.brightness import (
    BrightnessAdditiveTransform,
    MultiplicativeBrightnessTransform,
)
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from torch import nn

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
        labels (torch.Tensor): labels corresponding to image. class x1 y1 z1 y2 y2 z2
        masks (torch.Tensor | None): corresponding masks
        cutout_params (list[tuple[float, int]]): a list of ordered pairs that set sizes and numbers
            of cutout boxes. The maximum extent of the boxes is set by the first element of the
            ordered pair. The number of boxes with that maximum extent is set by the second element
            of the ordered pair
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
        if len(labels):
            box = torch.as_tensor(
                [xmin, ymin, zmin, xmax, ymax, zmax], dtype=labels.dtype, device=labels.device
            )
            intersection_over_volume = bbox_iov(box, labels[:, 1:7])
            keep_mask = intersection_over_volume < 0.60
            labels = labels[keep_mask]  # remove >60% obscured labels
            if masks is not None and keep_mask.logical_not().any():
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
    padding_fill_value: float = 0,
    uniform_zoom: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor | None] | None]:
    """Randomly zooms in or out of a random part of the input image.

    Args:
        image (torch.Tensor): 3-D tensor to be augmented.
        labels (torch.Tensor): Llbels corresponding to image. class x1 y1 z1 x2 y2 z2
        masks (torach.Tensor | None): corresponding masks
        max_zoom (float, optional): maximum edge length multiplier. Defaults to 1.5.
        min_zoom (float, optional): minimum edge length multiplier. Defaults to 0.7.
        p (float, optional): probability of zooming the input. Defaults to 0.5.
        out_zoom_padding (str): Padding strategy when zooming out ("random" or "constant").
        padding_fill_value (float): Fill value to use for constant padding.
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

    # get original image shape (resized to imgsz_z x imgsz_y x imgsz_x by this point in dataloader)
    channels, d, h, w = image.shape
    if uniform_zoom:
        zoom_factors = (random.uniform(min_zoom, max_zoom),) * 3  # (d, h, w) identical factors
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
    pad_starts: list[int] = []  # start index in final canvas to place resized (if padding)
    final_indices: list[slice] = []
    cropped_dims = []  # keep track which dims were cropped (for filtering labels later)
    for dim, (orig, new) in enumerate(zip(orig_sizes, new_sizes, strict=True)):
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
    if any(new < orig for orig, new in zip(orig_sizes, new_sizes, strict=True)):
        # prepare canvas
        if out_zoom_padding == "random":
            canvas = torch.rand(channels, d, h, w, device=image_cropped.device) * (
                torch.max(image_cropped) - torch.min(image_cropped)
            ) + torch.min(image_cropped)
        elif out_zoom_padding == "constant":
            canvas = (
                torch.zeros(channels, d, h, w, device=image_cropped.device)
                + padding_fill_value
            )
        else:
            msg = f"Unknown out zoom padding mode {out_zoom_padding}"
            raise ValueError(msg)
        z_start, y_start, x_start = pad_starts
        z_slice = slice(z_start, z_start + image_cropped.shape[1])
        y_slice = slice(y_start, y_start + image_cropped.shape[2])
        x_slice = slice(x_start, x_start + image_cropped.shape[3])
        canvas[:, z_slice, y_slice, x_slice] = image_cropped
        image = canvas
        if masks_cropped is not None:
            mask_canvas = torch.zeros(
                masks_cropped.shape[0],
                d,
                h,
                w,
                dtype=masks_cropped.dtype,
                device=masks_cropped.device,
            )
            mask_canvas[:, z_slice, y_slice, x_slice] = masks_cropped
            masks = mask_canvas
        else:
            masks = None
    else:
        # Only cropping, no padding
        image = image_cropped
        masks = masks_cropped

    # Adjust labels: subtract crop_starts for cropped dims, add pad_starts for padded dims
    # crop_starts list contains either crop start (if cropped) else 0;
    # pad_starts has placement start (0 if cropped)
    # mapping dim indices: 0->z/d, 1->h/y, 2->w/x ; label order x,y,z
    # First subtract crop starts for each axis where cropping occurred
    z_crop, y_crop, x_crop = crop_starts
    z_pad, y_pad, x_pad = pad_starts
    augmented_labels[:, [3, 6]] -= z_crop
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
        allowed_center_min = -8
        keep_masks = (
            (label_center_x >= allowed_center_min)
            & (label_center_x <= w + 8)
            & (label_center_y >= allowed_center_min)
            & (label_center_y <= h + 8)
            & (label_center_z >= allowed_center_min)
            & (label_center_z <= d + 8)
        )
        augmented_labels = augmented_labels[keep_masks]
        if masks is not None and keep_masks.logical_not().any():
            masks = masks[keep_masks]

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
    angle_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-20.0, 20.0),
        (-20.0, 20.0),
        (-20.0, 20.0),
    ),
    p: float = 0.5,
    keep_box_volume: bool = True,
    fill_value: float = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Randomly rotate the 3D volume around its center with independent angles per axis.

    Args:
        image: Tensor (C, D, H, W)
        labels: Tensor (N, 7) with columns [cls, x1, y1, z1, x2, y2, z2] in absolute voxel coords.
        masks: Optional tensor (M, D, H, W) of per-instance masks aligned with image.
        angle_ranges: ((min_x,max_x),(min_y,max_y),(min_z,max_z)) in degrees for rotations each axis
        p: Probability of applying the rotation.
        keep_box_volume (bool): Reduce box size after rotation to preserve box volume
        fill_value (float): Fill value for padded area.

    Returns:
        Rotated (image, labels, masks).
    """
    if random.random() >= p or (labels.numel() == 0 and masks is None):
        return image, labels, masks

    device = image.device
    dtype = image.dtype
    if labels.device != device:
        labels = labels.to(device=device)
    if masks is not None and masks.device != device:
        masks = masks.to(device=device)
    channels, depth, height, width = image.shape

    # Sample angles in radians
    ax = random.uniform(*angle_ranges[0]) * math.pi / 180.0
    ay = random.uniform(*angle_ranges[1]) * math.pi / 180.0
    az = random.uniform(*angle_ranges[2]) * math.pi / 180.0

    cx, cy, cz = (width - 1) / 2.0, (height - 1) / 2.0, (depth - 1) / 2.0

    # Rotation matrices (x,y,z axes) operating on [x,y,z]
    rotation_x = torch.tensor(
        [
            [1, 0, 0],
            [0, math.cos(ax), -math.sin(ax)],
            [0, math.sin(ax), math.cos(ax)],
        ],
        device=device,
        dtype=dtype,
    )
    rotation_y = torch.tensor(
        [
            [math.cos(ay), 0, math.sin(ay)],
            [0, 1, 0],
            [-math.sin(ay), 0, math.cos(ay)],
        ],
        device=device,
        dtype=dtype,
    )
    rotation_z = torch.tensor(
        [
            [math.cos(az), -math.sin(az), 0],
            [math.sin(az), math.cos(az), 0],
            [0, 0, 1],
        ],
        device=device,
        dtype=dtype,
    )
    rotation = rotation_z @ rotation_y @ rotation_x  # combined forward rotation

    # For grid_sample we need inverse transform (maps output -> input)
    rotation_inverse = rotation.T  # rotation matrices are orthonormal so inverse = transpose

    # Rotate image using affine_grid/grid_sample (normalized coordinates in [-1,1])
    affine = torch.zeros(1, 3, 4, device=device, dtype=dtype)
    affine[0, :, :3] = rotation_inverse
    grid = F.affine_grid(affine, size=(1, channels, depth, height, width), align_corners=True)
    if grid.device != device:
        grid = grid.to(device=device)
    # padding only possible with constant 0, so subtract fill value before rotation and add after
    image = image - fill_value
    image_rot = F.grid_sample(
        image.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)
    image_rot = image_rot + fill_value

    if masks is not None:
        masks_input = masks.unsqueeze(0).to(device=device, dtype=grid.dtype)
        masks_rot = F.grid_sample(
            masks_input,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)
        masks_rot_binary = (masks_rot > 0.5).to(masks.dtype)
    else:
        masks_rot_binary = None

    if labels.shape[0] == 0:
        return image_rot, labels, masks_rot_binary

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
    d = (z2 - z1).unsqueeze(1)
    # Build corner coords (N,8,3) in (x,y,z)
    base = torch.stack([x1, y1, z1], 1).unsqueeze(1)  # (N,1,3)
    extents = torch.stack([w, h, d], 2)  # (N,1,3)
    corners = base + corner_bits.unsqueeze(0) * extents  # broadcast

    # Center, rotate, uncenter
    corners_centered = corners.clone().to(dtype=dtype, device=device)
    corners_centered[..., 0] -= cx
    corners_centered[..., 1] -= cy
    corners_centered[..., 2] -= cz
    # Apply forward rotation (row-vector formulation p' = p * R^T)
    rot_corners = torch.matmul(corners_centered, rotation.T)
    rot_corners[..., 0] += cx
    rot_corners[..., 1] += cy
    rot_corners[..., 2] += cz

    new_x1 = rot_corners[..., 0].min(dim=1).values.clamp_(0, width - 1)
    new_y1 = rot_corners[..., 1].min(dim=1).values.clamp_(0, height - 1)
    new_z1 = rot_corners[..., 2].min(dim=1).values.clamp_(0, depth - 1)
    new_x2 = rot_corners[..., 0].max(dim=1).values.clamp_(0, width - 1)
    new_y2 = rot_corners[..., 1].max(dim=1).values.clamp_(0, height - 1)
    new_z2 = rot_corners[..., 2].max(dim=1).values.clamp_(0, depth - 1)

    rotated_labels = labels.clone()
    rotated_labels[:, 1] = new_x1
    rotated_labels[:, 2] = new_y1
    rotated_labels[:, 3] = new_z1
    rotated_labels[:, 4] = new_x2
    rotated_labels[:, 5] = new_y2
    rotated_labels[:, 6] = new_z2

    if keep_box_volume and rotated_labels.shape[0]:
        # Original volumes
        orig_vol = (
            (labels[:, 4] - labels[:, 1])
            * (labels[:, 5] - labels[:, 2])
            * (labels[:, 6] - labels[:, 3])
        )
        new_vol = (
            (rotated_labels[:, 4] - rotated_labels[:, 1])
            * (rotated_labels[:, 5] - rotated_labels[:, 2])
            * (rotated_labels[:, 6] - rotated_labels[:, 3])
        )
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
        nx1.clamp_(0, width - 1)
        nx2.clamp_(0, width - 1)
        ny1.clamp_(0, height - 1)
        ny2.clamp_(0, height - 1)
        nz1.clamp_(0, depth - 1)
        nz2.clamp_(0, depth - 1)
        # Write back
        rotated_labels[:, 1] = nx1
        rotated_labels[:, 2] = ny1
        rotated_labels[:, 3] = nz1
        rotated_labels[:, 4] = nx2
        rotated_labels[:, 5] = ny2
        rotated_labels[:, 6] = nz2

    # Remove degenerate boxes
    keep_masks = (
        (rotated_labels[:, 4] - rotated_labels[:, 1] > 1)
        & (rotated_labels[:, 5] - rotated_labels[:, 2] > 1)
        & (rotated_labels[:, 6] - rotated_labels[:, 3] > 1)
    )
    rotated_labels = rotated_labels[keep_masks]
    if masks_rot_binary is not None and keep_masks.logical_not().any():
        masks_rot_binary = masks_rot_binary[keep_masks]

    return image_rot, rotated_labels, masks_rot_binary


def random_mirroring(
    image: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor | None,
    mirror_axes: tuple[str] = ("z", "y", "x"),
    p: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor | None] | None]:
    """Applies image mirroring augmentation.

    Args:
        image (torch.Tensor): 3-D tensor to be augmented.
        labels (np.ndarray): Labels corresponding to image. class x y z w h d
        masks (torch.Tensor | None): masks corresponding to labels
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

    prob_per_axis = 0.5
    for axis in mirror_axes:
        if random.random() < prob_per_axis:
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
    noise_augmentation_device: Literal["cpu", "gpu"] = "cpu",
    blur_prob: float = 0.2,
    blur_sigma: tuple[float, float] = (0.5, 1.0),
    blur_augmentation_device: Literal["cpu", "gpu"] = "cpu",
    mult_brightness_prob: float = 0.15,
    mult_brightness_range: tuple[float, float] = (0.75, 1.25),
    mult_brightness_augmentation_device: Literal["cpu", "gpu"] = "cpu",
    add_brightness_prob: float = 0.15,
    add_brightness_mu: float = 0.0,
    add_brightness_sigma: float = 2.0,
    add_brightness_augmentation_device: Literal["cpu", "gpu"] = "cpu",
    contrast_prob: float = 0.15,
    contrast_range: tuple[float, float] = (0.75, 1.25),
    contrast_augmentation_device: Literal["cpu", "gpu"] = "cpu",
    lowres_prob: float = 0.25,
    lowres_scale: tuple[float, float] = (0.5, 1.0),
    lowres_augmentation_device: Literal["cpu", "gpu"] = "cpu",
    gamma_prob: float = 0.4,
    gamma_range: tuple[float, float] = (0.7, 1.5),
    gamma_p_invert: float = 0.25,
    gamma_p_retain_stats: float = 1.0,
    gamma_augmentation_device: Literal["cpu", "gpu"] = "cpu",
    placement: Literal["cpu", "gpu"] = "cpu",
) -> torch.Tensor:
    synchronize_channels = not do_intensity_augmentations_per_channel
    if noise_augmentation_device == placement and noise_prob > 0:
        image = gaussian_noise_augmentation(
            image,
            variance_range=noise_variance,
            p=noise_prob,
            synchronize_channels=synchronize_channels,
        )
    if blur_augmentation_device == placement and blur_prob > 0:
        image = gaussian_blur_augmentation(
            image,
            sigma_range=blur_sigma,
            p=blur_prob,
            synchronize_channels=synchronize_channels,
        )
    if mult_brightness_augmentation_device == placement and mult_brightness_prob > 0:
        image = multiplicative_brightness_augmentation(
            image,
            multiplier_range=mult_brightness_range,
            p=mult_brightness_prob,
            synchronize_channels=synchronize_channels,
        )
    if add_brightness_augmentation_device == placement and add_brightness_prob > 0:
        image = additive_brightness_augmentation(
            image,
            mu=add_brightness_mu,
            sigma=add_brightness_sigma,
            p=add_brightness_prob,
            per_channel=do_intensity_augmentations_per_channel,
        )
    if contrast_augmentation_device == placement and contrast_prob > 0:
        image = contrast_augmentation(
            image,
            contrast_range=contrast_range,
            p=contrast_prob,
            synchronize_channels=synchronize_channels,
        )
    if lowres_augmentation_device == placement and lowres_prob > 0:
        image = simulate_low_resolution_augmentation(
            image,
            scale_range=lowres_scale,
            p=lowres_prob,
            synchronize_channels=synchronize_channels,
        )
    if gamma_augmentation_device == placement and gamma_prob > 0:
        image = gamma_augmentation(
            image,
            gamma_range=gamma_range,
            p=gamma_prob,
            synchronize_channels=synchronize_channels,
            p_invert_image=gamma_p_invert,
            p_retain_stats=gamma_p_retain_stats,
        )
    return image


def _sample_scalar(value_range: tuple[float, float]) -> float:
    return random.uniform(*value_range)


def _apply_batchgenerators_transform(
    image: torch.Tensor,
    transform: BasicTransform,
    apply_probability: float,
) -> torch.Tensor:
    bg_result: dict[str, Any] = RandomTransform(
        transform,
        apply_probability=apply_probability,
    )(image=image)
    return bg_result["image"]


def _gaussian_kernel_1d(
    sigma: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor | None, int]:
    if sigma <= 0:
        return None, 0
    radius = max(1, math.ceil(sigma * 3))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel, radius


def _blur_along_dimension(
    image: torch.Tensor,
    sigma: float,
    dim: int,
) -> torch.Tensor:
    kernel_1d, radius = _gaussian_kernel_1d(sigma, image.dtype, image.device)
    if kernel_1d is None:
        return image
    channels = image.shape[0]
    if dim == 0:
        kernel = kernel_1d.view(1, 1, -1, 1, 1)
        padding = (radius, 0, 0)
    elif dim == 1:
        kernel = kernel_1d.view(1, 1, 1, -1, 1)
        padding = (0, radius, 0)
    else:
        kernel = kernel_1d.view(1, 1, 1, 1, -1)
        padding = (0, 0, radius)
    weight = kernel.expand(channels, 1, *kernel.shape[2:]).contiguous()
    return F.conv3d(image.unsqueeze(0), weight, padding=padding, groups=channels).squeeze(0)


def _gaussian_blur_single_channel(
    image: torch.Tensor,
    sigmas: tuple[float, float, float],
) -> torch.Tensor:
    blurred = image.unsqueeze(0)
    for dim, sigma in enumerate(sigmas):
        blurred = _blur_along_dimension(blurred, sigma, dim)
    return blurred.squeeze(0)


def gaussian_noise_augmentation(
    image: torch.Tensor,
    variance_range: tuple[float, float],
    p: float,
    synchronize_channels: bool,
) -> torch.Tensor:
    if random.random() >= p:
        return image
    if synchronize_channels:
        sigma = _sample_scalar(variance_range)
        noise = torch.randn((1, *image.shape[1:]), device=image.device, dtype=image.dtype) * sigma
        return image + noise
    noise_channels = []
    for _ in range(image.shape[0]):
        sigma = _sample_scalar(variance_range)
        noise_channels.append(torch.randn_like(image[0:1]) * sigma)
    return image + torch.cat(noise_channels, dim=0)


def gaussian_blur_augmentation(
    image: torch.Tensor,
    sigma_range: tuple[float, float],
    p: float,
    synchronize_channels: bool,
) -> torch.Tensor:
    if random.random() >= p:
        return image
    if synchronize_channels:
        sigmas = tuple(_sample_scalar(sigma_range) for _ in range(3))
        blurred = image
        for dim, sigma in enumerate(sigmas):
            blurred = _blur_along_dimension(blurred, sigma, dim)
        return blurred
    blurred_channels = []
    for channel in range(image.shape[0]):
        sigmas = tuple(_sample_scalar(sigma_range) for _ in range(3))
        blurred_channels.append(_gaussian_blur_single_channel(image[channel], sigmas).unsqueeze(0))
    return torch.cat(blurred_channels, dim=0)


def multiplicative_brightness_augmentation(
    image: torch.Tensor,
    multiplier_range: tuple[float, float],
    p: float,
    synchronize_channels: bool,
) -> torch.Tensor:
    return _apply_batchgenerators_transform(
        image,
        MultiplicativeBrightnessTransform(
            multiplier_range=multiplier_range,
            synchronize_channels=synchronize_channels,
        ),
        apply_probability=p,
    )


def additive_brightness_augmentation(
    image: torch.Tensor,
    mu: float,
    sigma: float,
    p: float,
    per_channel: bool,
) -> torch.Tensor:
    return _apply_batchgenerators_transform(
        image,
        BrightnessAdditiveTransform(
            mu=mu,
            sigma=sigma,
            per_channel=per_channel,
            p_per_channel=1,
        ),
        apply_probability=p,
    )


def contrast_augmentation(
    image: torch.Tensor,
    contrast_range: tuple[float, float],
    p: float,
    synchronize_channels: bool,
) -> torch.Tensor:
    return _apply_batchgenerators_transform(
        image,
        ContrastTransform(
            contrast_range=contrast_range,
            preserve_range=False,
            synchronize_channels=synchronize_channels,
        ),
        apply_probability=p,
    )


def simulate_low_resolution_augmentation(
    image: torch.Tensor,
    scale_range: tuple[float, float],
    p: float,
    synchronize_channels: bool,
) -> torch.Tensor:
    return _apply_batchgenerators_transform(
        image,
        SimulateLowResolutionTransform(
            scale=scale_range,
            synchronize_channels=synchronize_channels,
            synchronize_axes=False,
            ignore_axes=(),
            p_per_channel=1,
        ),
        apply_probability=p,
    )


def gamma_augmentation(
    image: torch.Tensor,
    gamma_range: tuple[float, float],
    p: float,
    synchronize_channels: bool,
    p_invert_image: float,
    p_retain_stats: float,
) -> torch.Tensor:
    return _apply_batchgenerators_transform(
        image,
        GammaTransform(
            gamma=gamma_range,
            p_invert_image=p_invert_image,
            synchronize_channels=synchronize_channels,
            p_per_channel=1,
            p_retain_stats=p_retain_stats,
        ),
        apply_probability=p,
    )


def ensure_non_zero_box_extents(
    labels: torch.Tensor,
) -> torch.Tensor:
    """Ensures bounding box extents are non-zero, which would cause issues in IoU computation.

    Args:
        labels (torch.Tensor): bounding boxes. class x y z w h d

    Returns:
        torch.Tensor: augmented labels
    """
    if labels.shape[0] == 0:
        return labels

    labels[:, 4:] = torch.where(labels[:, 4:] < 1, 1, labels[:, 4:])

    return labels


def random_channel_dropout(
    image: torch.Tensor,
    drop_probabilities: list[float] | tuple[float, ...],
    fill_value: float | None = 0.0,
) -> torch.Tensor:
    """Drop (zero-out or constant-fill) at most one channel based on per-channel probabilities.

    Args:
        image: (C, D, H, W) tensor.
        drop_probabilities: per-channel independent probabilities. Only one channel can be dropped;
            if multiple trigger, one is chosen proportionally to their probabilities.
        p: Global probability to attempt a dropout event at all.
        fill_value: Value to fill the dropped channel with. If None, use channel mean.

    Returns:
        image with possibly one channel dropped.
    """
    channels = image.shape[0]
    if len(drop_probabilities) != channels:
        msg = f"drop_probabilities length {len(drop_probabilities)} must match channels {channels}"
        raise ValueError(msg)

    # Determine candidate channels
    probs = torch.tensor(drop_probabilities, dtype=torch.float32)
    # Sample Bernoulli for each channel
    bern = torch.bernoulli(probs).bool()
    if not bern.any():
        return image
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
    return image


class GINNet(nn.Module):
    def __init__(
        self,
        hidden_layers: int,
        hidden_channels: int,
        p_kernel_size_1: float = 0.5,
    ) -> None:
        super().__init__()

        self._p_kernel_size_1 = p_kernel_size_1

        self.layers_kernel_size_3 = nn.ModuleList()
        self.layers_kernel_size_1 = nn.ModuleList()
        self.layers_activation_functions = nn.ModuleList()

        layer_dims = [1] + [hidden_channels] * (hidden_layers - 1) + [1]

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

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        for layer in self.layers_kernel_size_3:
            nn.init.normal_(layer.weight)
            nn.init.normal_(layer.bias)
        for layer in self.layers_kernel_size_1:
            nn.init.normal_(layer.weight)
            nn.init.normal_(layer.bias)

        output = torch.zeros_like(input_tensor)
        for channel in range(input_tensor.shape[0]):
            input_i = x = input_tensor[channel].unsqueeze(0)
            num_layers = len(self.layers_kernel_size_3)
            for i in range(num_layers):
                if random.random() <= self._p_kernel_size_1:
                    x = self.layers_kernel_size_1[i](x)
                else:
                    x = self.layers_kernel_size_3[i](x)
                if i < num_layers - 1:
                    x = F.leaky_relu(x)

            alpha = random.random()
            output_i = alpha * input_i + (1 - alpha) * x
            output_i = output_i * (
                torch.norm(input_i, p="fro") / (torch.norm(output_i, p="fro") + 1e-6)
            )
            output[channel] = output_i.squeeze(0)

        return output


def gin_augmentation(
    image: torch.Tensor,
    hidden_layers: int,
    hidden_channels: int,
    p_kernel_size_1: float = 0.5,
    p: float = 0.2,
) -> torch.Tensor:
    """Applies GIN (Global Intensity Non-linear) augumentation to input image."""
    if random.random() >= p:
        return image

    gin_model = GINNet(hidden_layers, hidden_channels, p_kernel_size_1).to(
        device=image.device,
        dtype=image.dtype,
    )
    with torch.no_grad():
        return gin_model.forward(image)

