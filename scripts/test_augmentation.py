from __future__ import annotations

import random
from typing import Any

import torch

from yolo11_3d.nn.augmentations import random_mirroring, random_zoom, tensor_cutout
from yolo11_3d.ultralytics.utils.ops import xyzxyz2xyzwhd
from yolo11_3d.ultralytics.utils.plotting import plot_images

if __name__ == "__main__":
    orig_img = torch.zeros((3, 300, 300, 300), dtype=torch.float32)

    orig_img[0, 150:, :, :] = 1
    orig_img[1, :, 150:, :] = 1
    orig_img[2, :, :, 150:] = 1

    imgsz = torch.as_tensor(orig_img.shape[1:])

    lt = [[random.uniform(0.1, 0.9) for _ in range(3)] for _ in range(6)]
    rb = [[random.uniform(lt_, 0.9) for lt_ in lt_s] for lt_s in lt]

    # orig_labels = torch.cat(
    #     (
    #         torch.arange(len(lt))[:, None],
    #         torch.as_tensor(lt) * imgsz[[2, 1, 0]],
    #         torch.as_tensor(rb) * imgsz[[2, 1, 0]],
    #     ),
    #     dim=1,
    # )
    orig_cls = torch.arange(len(lt)).unsqueeze_(-1)
    orig_labels = torch.cat(
        (
            torch.as_tensor(lt),
            torch.as_tensor(rb),
        ),
        dim=1,
    )

    images: list[torch.Tensor] = [orig_img]
    labels: list[torch.Tensor] = [
        # torch.cat((orig_labels[:, 0:1], xyzxyz2xyzwhd(orig_labels.clone()[:, 1:7])), dim=1)
        torch.cat((orig_cls, xyzxyz2xyzwhd(orig_labels)), dim=1)
    ]
    paths: list[str] = ["original"]

    for op, kwargs in (
        (tensor_cutout, {"cutout_params": [(0.5, 1), (0.25, 2), (0.125, 4), (0.0625, 8)]}),
        (tensor_cutout, {"cutout_params": [(0.5, 1), (0.25, 2), (0.125, 4), (0.0625, 8)]}),
        (tensor_cutout, {"cutout_params": [(0.5, 1), (0.25, 2), (0.125, 4), (0.0625, 8)]}),
        # (tensor_cutout, {"cutout_params": [(0.5, 1), (0.25, 2), (0.125, 4), (0.0625, 8)]}),
        # (tensor_cutout, {"cutout_params": [(0.5, 1), (0.25, 2), (0.125, 4), (0.0625, 8)]}),
        # (tensor_cutout, {"cutout_params": [(0.5, 1), (0.25, 2), (0.125, 4), (0.0625, 8)]}),
        (random_zoom, {"out_zoom_padding": "constant"}),
        (random_zoom, {"out_zoom_padding": "constant"}),
        (random_zoom, {"out_zoom_padding": "constant"}),
    ):
        kwargs: dict[str, Any]

        op_img = orig_img.clone()
        op_labels = torch.cat((orig_cls, orig_labels * imgsz[[2, 1, 0, 2, 1, 0]]), dim=1)

        op_img, op_labels = op(op_img, op_labels, p=1.0, **kwargs)

        op_labels[:, 1:7] = xyzxyz2xyzwhd(
            op_labels[:, 1:7] / torch.as_tensor(op_img.shape[1:])[[2, 1, 0, 2, 1, 0]]
        )

        images.append(op_img)
        labels.append(op_labels)
        paths.append(op.__name__)

    for op, kwargs in (
        (random_mirroring, {}),
        (random_mirroring, {}),
    ):
        kwargs: dict[str, Any]

        op_img = orig_img.clone()
        op_labels = torch.cat((orig_cls, xyzxyz2xyzwhd(orig_labels)), dim=1)

        op_img, op_labels = op(op_img, op_labels, p=1.0, **kwargs)

        images.append(op_img)
        labels.append(op_labels)
        paths.append(op.__name__)

    # cutout_img, cutout_labels = tensor_cutout(
    #     img.clone(),
    #     labels.clone(),
    #     cutout_params=[(0.5, 1), (0.25, 2), (0.125, 4), (0.0625, 8)],
    #     p=1.0,
    # )
    # zoom_img, zoom_labels = random_zoom(img, labels.clone(), p=1.0, out_zoom_padding="constant")

    # mirror_labels = labels.clone()
    # mirror_labels[:, 1:7] = xyzxyz2xyzwhd(mirror_labels[:, 1:7])
    # mirror_img, mirror_labels = random_mirroring(img, mirror_labels, p=1.0)

    # labels[:, 1:7] = xyzxyz2xyzwhd(labels[:, 1:7])
    # cutout_labels[:, 1:7] = xyzxyz2xyzwhd(cutout_labels[:, 1:7])
    # zoom_labels[:, 1:7] = xyzxyz2xyzwhd(zoom_labels[:, 1:7])

    # imgs = torch.stack((img, cutout_img, zoom_img, mirror_img), dim=0)
    # labelss = torch.cat((labels, cutout_labels, zoom_labels, mirror_labels))
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
