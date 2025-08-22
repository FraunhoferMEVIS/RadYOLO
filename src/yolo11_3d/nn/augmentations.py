from __future__ import annotations

from yolo11_3d.ultralytics.utils.ops import xyzwhd2xyzxyz

__author__ = "kgeissler"

import random
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from batchgeneratorsv2.transforms.intensity.brightness import (
    MultiplicativeBrightnessTransform,  # , BrightnessAdditiveTransform
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
    masks: list[torch.Tensor | None] | None,
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
                # Filter masks in sync with labels (do not modify mask tensors’ voxels for cutout)
                masks = [masks[i] for i, k in enumerate(keep_mask.tolist()) if k]

    return im, labels, masks

def random_zoom(
    image: torch.Tensor,
    labels: torch.Tensor,
    masks: list[torch.Tensor | None] | None,
    max_zoom: float = 1.5,
    min_zoom: float = 0.7,
    p: float = 0.5,
    out_zoom_padding: str = "random",
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor | None] | None]:
    """Randomly zooms in or out of a random part of the input image.

    Args:
        image (torch.Tensor): 3-D tensor to be augmented.
        labels (List[float]): MedYOLO labels corresponding to image. class z1 x1 y1 z2 x2 y2
        max_zoom (float, optional): maximum edge length multiplier. Defaults to 1.5.
        min_zoom (float, optional): minimum edge length multiplier. Defaults to 0.7.
        p (float, optional): probability of zooming the input. Defaults to 0.5.

    Returns:
        image: Augmented tensor.
        augmented_labels: Adjusted labels.
    """
    if random.random() >= p:
        return image, labels, masks

    augmented_labels = labels.clone()

    # retrieve original image shape (this is resized to imgsz_z x imgsz_y x imgsz_x by this point in the dataloader)
    channels, d, h, w = image.shape

    zoom_factor = random.uniform(min_zoom, max_zoom)

    # add batch dimension for functional interpolate
    image = torch.nn.functional.interpolate(
        image.unsqueeze(0), scale_factor=zoom_factor, mode="trilinear", align_corners=False
    ).squeeze_(0)
    # Scale masks (nearest)
    if masks is not None:
        scaled_masks: list[torch.Tensor | None] = []
        for m in masks:
            if m is None:
                scaled_masks.append(None)
                continue
            # m: (D,H,W) -> (1,1,D,H,W)
            m5 = m.unsqueeze(0).unsqueeze(0)
            m_scaled = F.interpolate(m5, scale_factor=zoom_factor, mode="nearest-exact").squeeze(0).squeeze(0)
            scaled_masks.append(m_scaled)
        masks = scaled_masks

    # remove batch dimension for compatibility with later code

    # retrieve new image shape
    new_d, new_h, new_w = image.shape[1:]

    # shrink/expand labels
    augmented_labels[:, 1:7] = augmented_labels[:, 1:7] * zoom_factor

    # crop/pad the zoomed image back to the input size and position it randomly relative to the new tensor
    if zoom_factor >= 1.0:
        # new side lengths longer than original side lengths
        # crop to original im.shape (center needs at least original_length/2 distance to each edge to preserve original image)
        zoom_center_d = random.randint(d // 2, new_d - d // 2)
        zoom_center_h = random.randint(h // 2, new_h - h // 2)
        zoom_center_w = random.randint(w // 2, new_w - w // 2)

        zmin = zoom_center_d - d // 2
        ymin = zoom_center_h - h // 2
        xmin = zoom_center_w - w // 2
        zmax = zmin + d
        ymax = ymin + h
        xmax = xmin + w

        image = image[:, zmin:zmax, ymin:ymax, xmin:xmax]

        if masks is not None:
            cropped_masks: list[torch.Tensor | None] = []
            for m in masks:
                if m is None:
                    cropped_masks.append(None)
                    continue
                cropped_masks.append(m[zmin:zmax, ymin:ymax, xmin:xmax])
            masks = cropped_masks

        # move labels to correspond to new center of zoom
        augmented_labels[:, [1, 4]] = augmented_labels[:, [1, 4]] - xmin
        # augmented_labels[:, 4] = augmented_labels[:, 4] - xmin
        augmented_labels[:, [2, 5]] = augmented_labels[:, [2, 5]] - ymin
        # augmented_labels[:, 5] = augmented_labels[:, 5] - ymin
        augmented_labels[:, [3, 6]] = augmented_labels[:, [3, 6]] - zmin
        # augmented_labels[:, 6] = augmented_labels[:, 6] - zmin

        # remove labels whose centers are 8 voxels outside of the new image region
        label_center_x = (augmented_labels[:, 1] + augmented_labels[:, 4]) / 2
        label_center_y = (augmented_labels[:, 2] + augmented_labels[:, 5]) / 2
        label_center_z = (augmented_labels[:, 3] + augmented_labels[:, 6]) / 2

        filter = ~(
            (label_center_x < xmin - 8)
            | (label_center_x > xmax + 8)
            | (label_center_y < ymin - 8)
            | (label_center_y > ymax + 8)
            | (label_center_z < zmin - 8)
            | (label_center_z > zmax + 8)
        )
        augmented_labels = augmented_labels[filter]
        if masks is not None:
            masks = [m for m, k in zip(masks, filter.tolist()) if k]

        # crop labels beyond bounds of new image
        # faster individually
        augmented_labels[:, 1].clamp_(0, w)  # x1
        augmented_labels[:, 2].clamp_(0, h)  # y1
        augmented_labels[:, 3].clamp_(0, d)  # z1
        augmented_labels[:, 4].clamp_(0, w)  # x2
        augmented_labels[:, 5].clamp_(0, h)  # y2
        augmented_labels[:, 6].clamp_(0, d)  # z2

    else:
        # new side lengths shorter than original side lengths
        # pad to original image shape
        zoom_center_d = random.randint(new_d // 2 + 1, d - new_d // 2 - 1)
        zoom_center_h = random.randint(new_h // 2 + 1, h - new_h // 2 - 1)
        zoom_center_w = random.randint(new_w // 2 + 1, w - new_w // 2 - 1)

        xmin = zoom_center_w - new_w // 2
        ymin = zoom_center_h - new_h // 2
        zmin = zoom_center_d - new_d // 2

        xmax = xmin + new_w
        ymax = ymin + new_h
        zmax = zmin + new_d

        # create a new tensor
        if out_zoom_padding == "random":
            new_image = torch.rand(channels, d, h, w) * (
                torch.max(image) - torch.min(image)
            ) + torch.min(image)
        elif out_zoom_padding == "constant":
            new_image = torch.zeros(channels, d, h, w) + torch.min(image)
        else:
            msg = f"Unknown out zoom padding mode {out_zoom_padding}"
            raise ValueError(msg)
        new_image[:, zmin:zmax, ymin:ymax, xmin:xmax] = image
        image = new_image
        del new_image

        if masks is not None:
            padded_masks: list[torch.Tensor | None] = []
            for m in masks:
                if m is None:
                    padded_masks.append(None)
                    continue
                    
                new_mask = torch.zeros(d, h, w, dtype=m.dtype, device=m.device)
                new_mask[zmin:zmax, ymin:ymax, xmin:xmax] = m
                padded_masks.append(new_mask)
            masks = padded_masks

        # move labels to correspond to new center of zoom
        augmented_labels[:, [1, 4]] = augmented_labels[:, [1, 4]] + xmin
        # augmented_labels[:, 4] = augmented_labels[:, 4] + xmin
        augmented_labels[:, [2, 5]] = augmented_labels[:, [2, 5]] + ymin
        # augmented_labels[:, 5] = augmented_labels[:, 5] + ymin
        augmented_labels[:, [3, 6]] = augmented_labels[:, [3, 6]] + zmin
        # augmented_labels[:, 6] = augmented_labels[:, 6] + zmin

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
            mdims = [d - 1 for d in flip_dims]  # map (C,D,H,W) dims -> (D,H,W) dims
            flipped_masks: list[torch.Tensor | None] = []
            for m in masks:
                if m is None:
                    flipped_masks.append(None)
                    continue
                flipped_masks.append(torch.flip(m, dims=mdims))
            masks = flipped_masks

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
        # RandomTransform(
        #   BrightnessAdditiveTransform(
        #    mu=0,
        #    sigma=10,
        #    per_channel=do_intensity_augmentations_per_channel,
        #    p_per_sample=0.2,
        #    p_per_channel=1
        #    ),
        #    apply_probability=0.15
        # ),
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
        masks = [m for m, k in zip(masks, filter.tolist()) if k]

    return image, labels, masks