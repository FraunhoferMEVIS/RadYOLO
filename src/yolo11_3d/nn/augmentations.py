from __future__ import annotations

from yolo11_3d.ultralytics.utils.ops import xyzwhd2xyzxyz

__author__ = "kgeissler"

import random
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
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
    image: torch.Tensor, do_intensity_augmentations_per_channel: bool
) -> torch.Tensor:
    synchronize_channels = not do_intensity_augmentations_per_channel
    transforms: list[BasicTransform] = [
        RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 1), synchronize_channels=synchronize_channels
            ),
            apply_probability=0.1,
        ),
        RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.0),
                synchronize_channels=synchronize_channels,
                p_per_channel=1,
            ),
            apply_probability=0.2,
        ),
        RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=(0.75, 1.25), synchronize_channels=synchronize_channels
            ),
            apply_probability=0.15,
        ),
        RandomTransform(
            BrightnessAdditiveTransform(
                mu=0,
                sigma=2,
                per_channel=do_intensity_augmentations_per_channel,
                p_per_channel=1
            ),
           apply_probability=0.15
        ),
        RandomTransform(
            ContrastTransform(
                contrast_range=(0.75, 1.25),
                preserve_range=False,
                synchronize_channels=synchronize_channels,
            ),
            apply_probability=0.15,
        ),
        RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=synchronize_channels,
                synchronize_axes=False,
                ignore_axes=tuple(),
                p_per_channel=1,
            ),
            apply_probability=0.25,
        ),
        RandomTransform(
            GammaTransform(
                gamma=(0.7, 1.5),
                p_invert_image=0.25,
                synchronize_channels=synchronize_channels,
                p_per_channel=1,
                p_retain_stats=1,
            ),
            apply_probability=0.4,
        ),
    ]
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