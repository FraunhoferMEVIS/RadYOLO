# Ultralytics YOLO 🚀, AGPL-3.0 license

from __future__ import annotations

import logging
import math

import torch

LOGGER = logging.getLogger(__name__)


def check_imgsz(
    imgsz: int | list[int],
    stride: int | torch.Tensor = 32,
    min_dim: int = 1,
    max_dim: int = 2,
    floor: int = 0,
) -> list[int] | int:
    """Verify image size is a multiple of the given stride in each dimension. If the image size is not a multiple of the
    stride, update it to the nearest multiple of the stride that is greater than or equal to the given floor value.

    Args:
        imgsz (int | cList[int]): Image size.
        stride (int): Stride value.
        min_dim (int): Minimum number of dimensions.
        max_dim (int): Maximum number of dimensions.
        floor (int): Minimum allowed value for image size.

    Returns:
        (List[int]): Updated image size.
    """
    # Convert stride to integer if it is a tensor
    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)

    # Convert image size to list if it is an integer
    if isinstance(imgsz, int):
        imgsz = [imgsz]
    elif isinstance(imgsz, list | tuple):
        imgsz = list(imgsz)
    # elif isinstance(imgsz, str):  # i.e. '640' or '[640,640]'
    #     imgsz = [int(imgsz)] if imgsz.isnumeric() else eval(imgsz)
    else:
        msg = (
            f"'imgsz={imgsz}' is of invalid type {type(imgsz).__name__}. "
            f"Valid imgsz types are int i.e. 'imgsz=640' or list i.e. 'imgsz=[640,640]'"
        )
        raise TypeError(msg)

    # Apply max_dim
    if len(imgsz) > max_dim:
        msg = (
            "'train' and 'val' imgsz must be an integer, while 'predict' and 'export' imgsz may be a [h, w] list "
            "or an integer, i.e. 'yolo export imgsz=640,480' or 'yolo export imgsz=640'"
        )
        if max_dim != 1:
            raise ValueError(f"imgsz={imgsz} is not a valid image size. {msg}")
        LOGGER.warning(f"WARNING ⚠️ updating to 'imgsz={max(imgsz)}'. {msg}")
        imgsz = [max(imgsz)]
    # Make image size a multiple of the stride
    size = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]

    # Print warning message if image size was updated
    if size != imgsz:
        LOGGER.warning(
            f"WARNING ⚠️ imgsz={imgsz} must be multiple of max stride {stride}, updating to {size}"
        )

    # Add missing dimensions if necessary
    return (
        [size[0], size[0]]
        if min_dim == 2 and len(size) == 1
        else size[0]
        if min_dim == 1 and len(size) == 1
        else size
    )
