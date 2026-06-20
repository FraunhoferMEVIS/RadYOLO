from __future__ import annotations

import numpy as np


def _sorted_instances(
    labels_tuples: list[tuple[int, float, float, float, float, float, float]],
) -> list[tuple[int, int, float, float, float, float, float, float]]:
    indexed_instances = [
        (index, cls, xc, yc, zc, sx, sy, sz)
        for index, (cls, xc, yc, zc, sx, sy, sz) in enumerate(labels_tuples)
    ]
    return sorted(
        indexed_instances,
        key=lambda item: (item[5] * item[6] * item[7], item[0]),
        reverse=True,
    )


def compose_labelmap_from_masks(
    mask_arr: np.ndarray,
    labels_tuples: list[tuple[int, float, float, float, float, float, float]],
) -> np.ndarray | None:
    mask_arr = np.squeeze(mask_arr)
    if not labels_tuples:
        if mask_arr.ndim == 3:
            return np.zeros(mask_arr.shape, dtype=np.uint16)
        return None

    sorted_instances = _sorted_instances(labels_tuples)
    labelmap: np.ndarray | None = None

    if mask_arr.ndim == 3:
        if len(sorted_instances) != 1:
            return None
        _, cls, *_ = sorted_instances[0]
        labelmap = np.zeros(mask_arr.shape, dtype=np.uint16)
        labelmap[mask_arr > 0] = np.uint16(cls + 1)
    elif mask_arr.ndim == 4:
        num_instances = len(labels_tuples)
        if mask_arr.shape[-1] == num_instances:
            labelmap = np.zeros(mask_arr.shape[:-1], dtype=np.uint16)
            for index, cls, *_ in sorted_instances:
                labelmap[mask_arr[..., index] > 0] = np.uint16(cls + 1)
        elif mask_arr.shape[0] == num_instances:
            labelmap = np.zeros(mask_arr.shape[1:], dtype=np.uint16)
            for index, cls, *_ in sorted_instances:
                labelmap[mask_arr[index] > 0] = np.uint16(cls + 1)

    return labelmap


def compose_labelmap_from_boxes(
    image_shape: tuple[int, int, int],
    labels_tuples: list[tuple[int, float, float, float, float, float, float]],
) -> np.ndarray:
    width, height, depth = image_shape
    out = np.zeros((width, height, depth), dtype=np.uint16)
    for _, cls, xc, yc, zc, sx, sy, sz in _sorted_instances(labels_tuples):
        cx = xc * width
        cy = yc * height
        cz = zc * depth
        rx = (sx * width) / 2.0
        ry = (sy * height) / 2.0
        rz = (sz * depth) / 2.0
        x0 = max(0, int(np.floor(cx - rx)))
        x1 = min(width, int(np.ceil(cx + rx)))
        y0 = max(0, int(np.floor(cy - ry)))
        y1 = min(height, int(np.ceil(cy + ry)))
        z0 = max(0, int(np.floor(cz - rz)))
        z1 = min(depth, int(np.ceil(cz + rz)))
        if x1 <= x0 or y1 <= y0 or z1 <= z0:
            continue
        out[x0:x1, y0:y1, z0:z1] = np.uint16(cls + 1)
    return out
