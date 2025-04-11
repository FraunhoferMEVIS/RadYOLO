# Ultralytics YOLO 🚀, AGPL-3.0 license

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def xyxy2xywh[T: torch.Tensor | npt.NDArray](x: T) -> T:
    """Convert bounding box coordinates.

    From (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the top-left
    corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2)
        format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xyzxyz2xyzwhd[T: torch.Tensor | npt.NDArray](x: T) -> T:
    """Convert bounding box coordinates.

    From (x1, y1, z1, x2, y2, z2) format to (x, y, z, width, height, depth) format where
    (x1, y1, z1) is the top-left corner and (x2, y2, z2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in
        (x1, y1, z1, x2, y2, z2) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in
        (x, y, z, width, height, depth) format.
    """
    assert x.shape[-1] == 6, f"input shape last dimension expected 6 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = (x[..., 2] + x[..., 4]) / 2  # z center
    y[..., 3] = x[..., 3] - x[..., 0]  # width
    y[..., 4] = x[..., 4] - x[..., 1]  # height
    y[..., 5] = x[..., 5] - x[..., 2]  # depth
    return y


def xywh2xyxy[T: torch.Tensor | npt.NDArray](x: T) -> T:
    """Convert bounding box coordinates.

    From (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the top-left
    corner and (x2, y2) is the bottom-right corner.

    Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height)
        format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y


def xyzwhd2xyzxyz[T: torch.Tensor | npt.NDArray](x: T) -> T:
    """Convert bounding box coordinates.

    From (x, y, z, width, height, depth) format to (x1, y1, z1, x2, y2, z2) format where
    (x1, y1, z1) is the top-left corner and (x2, y2, z2) is the bottom-right corner.

    Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in
        (x, y, z, width, height, depth) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, z1, x2, y2, z2)
        format.
    """
    assert x.shape[-1] == 6, f"input shape last dimension expected 6 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    xyz = x[..., :3]  # centers
    whd = x[..., 3:] / 2  # half width-height
    y[..., :3] = xyz - whd  # top left xy
    y[..., 3:] = xyz + whd  # bottom right xy
    return y


def empty_like[T: torch.Tensor | npt.NDArray](x: T) -> T:
    """Creates empty torch.Tensor or np.ndarray with same shape as input and float32 dtype."""
    return (
        torch.empty_like(x, dtype=torch.float32)
        if isinstance(x, torch.Tensor)
        else np.empty_like(x, dtype=np.float32)
    )
