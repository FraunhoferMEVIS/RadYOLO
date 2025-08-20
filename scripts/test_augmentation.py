from __future__ import annotations

import random
from typing import Any

import torch
from yolo11_3d.nn.augmentations import (
    random_mirroring,
    random_zoom,
    tensor_cutout,
)
from yolo11_3d.ultralytics.utils.ops import xyzxyz2xyzwhd
from yolo11_3d.ultralytics.utils.plotting import plot_images

if __name__ == "__main__":
    # create empty test image
    orig_img = torch.zeros((3, 256, 256, 256), dtype=torch.float32)

    # slice volume along each axis and set one channel to 1. This makes every 1/8 cube have a unique
    # color enabling fast visual identification during transformations.
    orig_img[0, 128:, :, :] = 1
    orig_img[1, :, 128:, :] = 1
    orig_img[2, :, :, 128:] = 1

    # make image size a tensor once for later use
    imgsz = torch.as_tensor(orig_img.shape[1:])
    scale_tensor = imgsz[[2, 1, 0, 2, 1, 0]]

    n_boxes = 6
    n_dims = orig_img.ndim - 1  # subtract channel dim

    # randomly sample left-top and right-bottom coordinates (normalized) for n_boxes of bounding
    # boxes
    lt: list[list[float]] = [
        [random.uniform(0.1, 0.9) for _ in range(n_dims)] for _ in range(n_boxes)
    ]
    rb: list[list[float]] = [
        [random.uniform(lt_coord, 0.9) for lt_coord in lt_coords] for lt_coords in lt
    ]

    # add a single box with a width of <8 pixels to test the behavior of the
    lt.append([0.1, 0.1, 0.1])
    rb.append([0.1 + 0.02, 0.1 + 0.02, 0.1 + 0.02])

    # assign a unique class id to every generated bounding box
    orig_cls = torch.arange(len(lt)).unsqueeze_(-1)
    # concatenate left-top and right-bottom coordinates to form xyzxyz (normalized) bounding box
    orig_labels = torch.cat(
        (
            torch.as_tensor(lt),
            torch.as_tensor(rb),
        ),
        dim=1,
    )

    # collect corresponding images, labels and 'paths' for easy plotting later on
    # note that 'paths' is actually used to plot a heading label for each image in the plot
    images: list[torch.Tensor] = [orig_img]
    labels: list[torch.Tensor] = [torch.cat((orig_cls, xyzxyz2xyzwhd(orig_labels)), dim=1)]
    paths: list[str] = ["original"]

    # loop to evaluate all methods that require labels to be in format xyzxyz (not normalized)
    for op, kwargs in (
        (tensor_cutout, {"cutout_params": [(0.5, 1), (0.25, 2), (0.125, 4), (0.0625, 8)]}),
        (tensor_cutout, {"cutout_params": [(0.5, 1), (0.25, 2), (0.125, 4), (0.0625, 8)]}),
        (tensor_cutout, {"cutout_params": [(0.5, 1), (0.25, 2), (0.125, 4), (0.0625, 8)]}),
        (random_zoom, {"out_zoom_padding": "constant"}),
        (random_zoom, {"out_zoom_padding": "constant"}),
        (random_zoom, {"out_zoom_padding": "constant"}),
    ):
        kwargs: dict[str, Any]

        # clone image tensor so that modifications do not affect original image and thus also
        # subsequent loop evaluations
        op_img = orig_img.clone()

        # scale labels with image size to denormalize
        # torch.cat already creates a new tensor so no cloning necessary
        op_labels = torch.cat((orig_cls, orig_labels * scale_tensor), dim=1)

        # apply operation
        op_img, op_labels = op(op_img, op_labels, p=1.0, **kwargs)

        # convert bounding box to xyzwhd and normalize representation for plotting
        op_labels[:, 1:7] = xyzxyz2xyzwhd(op_labels[:, 1:7] / scale_tensor)

        images.append(op_img)
        labels.append(op_labels)
        paths.append(op.__name__)

    # loop to evaluate all functions that require labels to be in format xyzwhd (normalized)
    for op, kwargs in (
        (random_mirroring, {}),
        (random_mirroring, {}),
    ):
        kwargs: dict[str, Any]

        # clone image tensor so that modifications do not affect original image and thus also
        # subsequent loop evaluations
        op_img = orig_img.clone()

        # convert labels to xyzwhd (values are already normalized)
        # torch.cat already creates a new tensor so no cloning necessary
        op_labels = torch.cat((orig_cls, xyzxyz2xyzwhd(orig_labels)), dim=1)

        # apply operation
        op_img, op_labels = op(op_img, op_labels, p=1.0, **kwargs)

        # labels are already in format required for plotting
        images.append(op_img)
        labels.append(op_labels)
        paths.append(op.__name__)

    imgs = torch.stack(images, dim=0)
    labelss = torch.cat(labels, dim=0)

    plot_images(
        images=imgs.amax(-3),
        batch_idx=torch.as_tensor(
            [i for i, labels_ in enumerate(labels) for _ in range(len(labels_))]
        ),
        cls=labelss[:, 0:1],
        bboxes=labelss[:, [1, 2, 4, 5]],
        paths=paths,
        fname="image_xy.jpg",
    )

    plot_images(
        images=imgs.amax(-2),
        batch_idx=torch.as_tensor(
            [i for i, labels_ in enumerate(labels) for _ in range(len(labels_))]
        ),
        cls=labelss[:, 0:1],
        bboxes=labelss[:, [1, 3, 4, 6]],
        paths=paths,
        fname="image_xz.jpg",
    )
