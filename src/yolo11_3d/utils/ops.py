# Ultralytics YOLO 🚀, AGPL-3.0 license

from __future__ import annotations

import logging
import time

import numpy as np
import numpy.typing as npt
import torch

from yolo11_3d.validation.metrics import bbox_iou

LOGGER = logging.getLogger(__name__)


def scale_boxes(
    img1_shape: tuple[int, ...],
    boxes: torch.Tensor,
    img0_shape: tuple[int, ...],
    ratio_pad: tuple[tuple[float, float, float], tuple[int, int, int]] | None = None,
    padding: bool = True,
    xyzwhd: bool = False,
) -> torch.Tensor:
    """Rescales bounding boxes (in the format of xyxy by default).

    From the shape of the image they were originally specified in (img1_shape) to the shape of a
    different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of
            (depth, width, height).
        boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of
            (x1, y1, z1, x2, y2, z2)
        img0_shape (tuple): the shape of the target image, in the format of (depth, width, height).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio
            and pad will be calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If
            False then do regular rescaling.
        xyzwhd (bool): The box format is xyzwhd or not, default=False.

    Returns:
        boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, z1, x2, y2, z2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = (
            img1_shape[2] / img0_shape[2],
            img1_shape[1] / img0_shape[1],
            img1_shape[0] / img0_shape[0],
        )
        pad = (
            round((img1_shape[2] - img0_shape[2] * gain[0]) / 2 - 0.1),
            round((img1_shape[1] - img0_shape[1] * gain[1]) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain[2]) / 2 - 0.1),
        )  # whd padding
    else:
        gain = ratio_pad[0][::-1]
        pad = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad[0]  # x padding
        boxes[..., 1] -= pad[1]  # y padding
        boxes[..., 2] -= pad[2]  # z padding
        if not xyzwhd:
            boxes[..., 3] -= pad[0]  # x padding
            boxes[..., 4] -= pad[1]  # y padding
            boxes[..., 5] -= pad[2]  # z padding
    boxes[..., :6] /= torch.as_tensor(gain, device=boxes.device).tile(
        1, 2
    )
    return clip_boxes(boxes, img0_shape)



def nms_(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    order = torch.argsort(-scores)
    indices = torch.arange(boxes.shape[0])
    keep = torch.ones_like(indices, dtype=torch.bool)

    for i in indices:
        if keep[i]:
            box = boxes[order[i]]
            iou = bbox_iou(
                box.unsqueeze(0),
                (boxes[order[i] + 1 :]) * keep[i + 1 :].unsqueeze(-1),
                xyzwhd=False,
            )
            overlapped = torch.nonzero(iou > iou_threshold)
            keep[overlapped + i + 1] = 0

    return order[keep]


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    order = torch.argsort(-scores)
    keep = []

    iter_count = 0
    num_indices = len(order)
    while len(order) > 0 and iter_count < num_indices:
        idx = order[0]

        keep.append(idx)

        order = order[1:]

        # sanity check
        if len(order) == 0:
            break

        # sanity check
        if iter_count > num_indices:
            break

        iou = bbox_iou(boxes[idx], boxes[order], xyzwhd=False).squeeze(-1)

        mask = iou <= iou_threshold
        order = order[mask]

    return torch.stack(keep, 0)


def non_max_suppression(
    prediction: list[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]],
    conf_thres: float = 0.1,
    iou_thres: float = 0.2,
    classes: list[int] | torch.Tensor | None = None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels: list[list[int | float | torch.Tensor]] | list[torch.Tensor] = (),
    max_det: int = 300,
    nc: int = 0,  # number of classes (optional)
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    in_place: bool = True,
) -> list[torch.Tensor]:
    """Perform non-maximum suppression (NMS) on a set of boxes.

    Supports masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels.
        in_place (bool): If True, the input prediction tensor will be modified in place.

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
    # import torchvision  # scope for faster 'import ultralytics'

    # Checks
    assert 0 <= conf_thres <= 1, (
        f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    )
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(
        prediction, (list, tuple)
    ):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 6)  # number of classes
    nm = prediction.shape[1] - nc - 6  # number of masks
    mi = 6 + nc  # mask start index
    xc = prediction[:, 6:mi].amax(1) > conf_thres  # candidates

    # Settings
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if in_place:
        prediction[..., :6] = xyzwhd2xyzxyz(prediction[..., :6])  # xyzwhd to xyzxyz
    else:
        prediction = torch.cat(
            (xyzwhd2xyzxyz(prediction[..., :6]), prediction[..., 6:]), dim=-1
        )  # xyzwhd to xyzxyz

    t = time.time()
    output = [torch.zeros((0, 8 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 6), device=x.device)
            v[:, :6] = xywh2xyxy(lb[:, 1:7])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyzxyz, conf, cls)
        box, cls, mask = x.split((6, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 6 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 7:8] == classes).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[
                x[:, 6].argsort(descending=True)[:max_nms]
            ]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 7:8] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 6]  # scores
        boxes = x[:, :6] + c  # boxes (offset by class)
        # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = nms(boxes, scores, iou_thres)
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output


def clip_boxes(boxes: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Takes a list of bounding boxes and a shape (depth, width, height) and clips the bounding boxes.

    Args:
        boxes (torch.Tensor): The bounding boxes to clip.
        shape (tuple): The shape of the image.

    Returns:
        (torch.Tensor | numpy.ndarray): The clipped boxes.
    """
    if isinstance(
        boxes, torch.Tensor
    ):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        boxes[..., 0] = boxes[..., 0].clamp(0, shape[2])  # x1
        boxes[..., 1] = boxes[..., 1].clamp(0, shape[1])  # y1
        boxes[..., 2] = boxes[..., 2].clamp(0, shape[0])  # z1
        boxes[..., 3] = boxes[..., 3].clamp(0, shape[2])  # x2
        boxes[..., 4] = boxes[..., 4].clamp(0, shape[1])  # y2
        boxes[..., 5] = boxes[..., 5].clamp(0, shape[0])  # z2
    else:  # np.array (faster grouped)
        boxes[..., [0, 3]] = boxes[..., [0, 3]].clip(0, shape[2])  # x1, x2
        boxes[..., [1, 4]] = boxes[..., [1, 4]].clip(0, shape[1])  # y1, y2
        boxes[..., [2, 5]] = boxes[..., [2, 5]].clip(0, shape[0])  # z1, z2
    return boxes


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
    y[..., 0] = (x[..., 0] + x[..., 3]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 4]) / 2  # y center
    y[..., 2] = (x[..., 2] + x[..., 5]) / 2  # z center
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
