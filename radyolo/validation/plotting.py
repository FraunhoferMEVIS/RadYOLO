# Ultralytics YOLO 🚀, AGPL-3.0 license

from __future__ import annotations

import contextlib
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont

from radyolo.utils import ops

if TYPE_CHECKING:
    from collections.abc import Callable

LOGGER = logging.getLogger(__name__)


class Colors:
    """Ultralytics color palette.

    https://docs.ultralytics.com/reference/utils/plotting/#ultralytics.utils.plotting.Colors.

    This class provides methods to work with the Ultralytics color palette, including converting
    hex color codes to RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
    """

    def __init__(self) -> None:
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "042AFF",
            "0BDBEB",
            "F3F3F3",
            "00DFB7",
            "111F68",
            "FF6FDD",
            "FF444F",
            "CCED00",
            "00F344",
            "BD00FF",
            "00B4FF",
            "DD00BA",
            "00FFFF",
            "26C000",
            "01FFB3",
            "7D24FF",
            "7B0068",
            "FF1B6C",
            "FC6D2F",
            "A2FF0B",
        )
        self.palette: list[tuple[int, int, int]] = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n: int = len(self.palette)

    def __call__(self, i: int | str, bgr: bool = False) -> tuple[int, int, int]:
        """Converts hex color codes to RGB values."""
        c: tuple[int, int, int] = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h: str) -> tuple[int, int, int]:
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


class Annotator:
    """Ultralytics Annotator for train/val mosaics and JPGs and predictions annotations.

    Attributes:
        image (Image.Image or numpy array): The image to annotate.
        pil (bool): Whether to use PIL or cv2 for drawing annotations.
        font (ImageFont.truetype or ImageFont.load_default): Font used for text annotations.
        line_width (float): Line width for drawing.
    """

    def __init__(
        self,
        im: Image.Image | npt.NDArray[np.uint8 | np.float32],
        line_width: int | None = None,
        font_size: float | None = None,
        font: str = "Arial.ttf",
        pil: bool = False,
    ) -> None:
        """Initialize Annotator class with image and line width along with color palette."""
        self.line_width: int = line_width or max(
            round(sum(im.size if isinstance(im, Image.Image) else im.shape) / 2 * 0.003), 2
        )
        self.image: Image.Image | npt.NDArray[np.uint8 | np.float32]
        if isinstance(im, Image.Image) or pil:  # use PIL
            self.pil = True
            self.image = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw: ImageDraw.ImageDraw = ImageDraw.Draw(self.image)
            try:
                size = font_size or max(round(sum(self.image.size) / 2 * 0.035), 12)
                self.font = ImageFont.truetype(str(font), size)
            except Exception:
                self.font = ImageFont.load_default()
            self.getsize: Callable[[str], tuple[int, int] | tuple[float, float]] = (
                lambda x: self.font.getbbox(x)[2:4]
            )
        else:  # use cv2
            self.pil = False
            assert im.data.contiguous, (
                "Image not contiguous. Apply np.ascontiguousarray(im) to Annotator input images."
            )
            self.image = im if im.flags.writeable else im.copy()
            self.font_thickness: int = max(self.line_width - 1, 1)
            self.font_scale: float = self.line_width / 3
        self.dark_colors = {
            (235, 219, 11),
            (243, 243, 243),
            (183, 223, 0),
            (221, 111, 255),
            (0, 237, 204),
            (68, 243, 0),
            (255, 255, 0),
            (179, 255, 1),
            (11, 255, 162),
        }
        self.light_colors = {
            (255, 42, 4),
            (79, 68, 255),
            (255, 0, 189),
            (255, 180, 0),
            (186, 0, 221),
            (0, 192, 38),
            (255, 36, 125),
            (104, 0, 123),
            (108, 27, 255),
            (47, 109, 252),
            (104, 31, 17),
        }

    def get_txt_color(
        self,
        color: tuple[int, int, int] = (128, 128, 128),
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ) -> (
        tuple[Literal[104], Literal[31], Literal[17]]
        | tuple[Literal[255], Literal[255], Literal[255]]
        | tuple[int, int, int]
    ):
        """Assign text color based on background color.

        Args:
            color (tuple, optional): The background color of the rectangle for text (B, G, R).
            txt_color (tuple, optional): The color of the text (R, G, B).

        Returns:
            txt_color (tuple): Text color for label
        """
        if color in self.dark_colors:
            return 104, 31, 17
        if color in self.light_colors:
            return 255, 255, 255
        return txt_color

    def text_label(
        self,
        box: tuple[float, float, float, float],
        label: str = "",
        color: tuple[int, int, int] = (128, 128, 128),
        txt_color: tuple[int, int, int] = (255, 255, 255),
        margin: int = 5,
    ) -> None:
        """Draws a label with a background rectangle centered within a given bounding box.

        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            label (str): The text label to be displayed.
            color (tuple, optional): The background color of the rectangle (B, G, R).
            txt_color (tuple, optional): The color of the text (R, G, B).
            margin (int, optional): The margin between the text and the rectangle border.
        """
        # Calculate the center of the bounding box
        x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        # Get the size of the text
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale - 0.1, self.font_thickness
        )[0]
        # Calculate the top-left corner of the text (to center it)
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2
        # Calculate the coordinates of the background rectangle
        rect_x1 = text_x - margin
        rect_y1 = text_y - text_size[1] - margin
        rect_x2 = text_x + text_size[0] + margin
        rect_y2 = text_y + margin
        # Draw the background rectangle
        cv2.rectangle(self.image, (rect_x1, rect_y1), (rect_x2, rect_y2), color, -1)
        # Draw the text on top of the rectangle
        cv2.putText(
            self.image,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale - 0.1,
            self.get_txt_color(color, txt_color),
            self.font_thickness,
            lineType=cv2.LINE_AA,
        )

    def box_label(
        self,
        box: tuple[float, float, float, float] | torch.Tensor,
        label: str = "",
        color: tuple[int, int, int] = (128, 128, 128),
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """Draws a bounding box to image with label.

        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            label (str): The text label to be displayed.
            color (tuple, optional): The background color of the rectangle (B, G, R).
            txt_color (tuple, optional): The color of the text (R, G, B).
        """
        txt_color = self.get_txt_color(color, txt_color)
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        if self.pil:
            p1 = (box[0], box[1])
            self.draw.rectangle(box, width=self.line_width, outline=color)  # box
            if label:
                text_width, text_height = self.getsize(label)  # text width, height
                outside = p1[1] >= text_height  # label fits outside box
                if (
                    p1[0] > self.image.size[0] - text_width
                ):  # size is (w, h), check if label extend beyond right side of image
                    p1 = self.image.size[0] - text_width, p1[1]
                self.draw.rectangle(
                    (
                        p1[0],
                        p1[1] - text_height if outside else p1[1],
                        p1[0] + text_width + 1,
                        p1[1] + 1 if outside else p1[1] + text_height + 1,
                    ),
                    fill=color,
                )
                self.draw.text(
                    (p1[0], p1[1] - text_height if outside else p1[1]),
                    label,
                    fill=txt_color,
                    font=self.font,
                )
        else:  # cv2
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(
                self.image, p1, p2, color, thickness=self.line_width, lineType=cv2.LINE_AA
            )
            if label:
                text_width, text_height = cv2.getTextSize(
                    label, 0, fontScale=self.font_scale, thickness=self.font_thickness
                )[0]
                text_height += 3  # add pixels to pad text
                outside = p1[1] >= text_height  # label fits outside box
                if (
                    p1[0] > self.image.shape[1] - text_width
                ):  # check if label extend beyond right image side
                    p1 = self.image.shape[1] - text_width, p1[1]
                p2 = p1[0] + text_width, p1[1] - text_height if outside else p1[1] + text_height
                cv2.rectangle(self.image, p1, p2, color, -1, cv2.LINE_AA)
                cv2.putText(
                    self.image,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + text_height - 1),
                    0,
                    self.font_scale,
                    txt_color,
                    thickness=self.font_thickness,
                    lineType=cv2.LINE_AA,
                )

    def rectangle(self, xy, fill=None, outline=None, width: int = 1) -> None:
        """Add rectangle to image (PIL-only)."""
        self.draw.rectangle(xy, fill, outline, width)

    def text(
        self,
        xy,
        text: str,
        txt_color: tuple[int, int, int] = (255, 255, 255),
        anchor: Literal["top", "bottom"] = "top",
        box_style: bool = False,
    ) -> None:
        """Adds text to an image using PIL or cv2."""
        if anchor == "bottom":  # start y from font bottom
            text_width, text_height = self.getsize(text)
            xy[1] += 1 - text_height
        if self.pil:
            if box_style:
                text_width, text_height = self.getsize(text)
                self.draw.rectangle(
                    (xy[0], xy[1], xy[0] + text_width + 1, xy[1] + text_height + 1), fill=txt_color
                )
                # Using `txt_color` for background and draw fg with white color
                txt_color = (255, 255, 255)
            if "\n" in text:
                lines = text.split("\n")
                _, text_height = self.getsize(text)
                for line in lines:
                    self.draw.text(xy, line, fill=txt_color, font=self.font)
                    xy[1] += text_height
            else:
                self.draw.text(xy, text, fill=txt_color, font=self.font)
        else:
            if box_style:
                text_width, text_height = cv2.getTextSize(
                    text, 0, fontScale=self.font_scale, thickness=self.font_thickness
                )[0]  # text width, height
                text_height += 3  # add pixels to pad text
                outside = xy[1] >= text_height  # label fits outside box
                p2 = xy[0] + text_width, xy[1] - text_height if outside else xy[1] + text_height
                cv2.rectangle(self.image, xy, p2, txt_color, -1, cv2.LINE_AA)  # filled
                # Using `txt_color` for background and draw fg with white color
                txt_color = (255, 255, 255)
            cv2.putText(
                self.image,
                text,
                xy,
                0,
                self.font_scale,
                txt_color,
                thickness=self.font_thickness,
                lineType=cv2.LINE_AA,
            )

    def fromarray(self, im: npt.NDArray[np.uint8 | np.float32]) -> None:
        """Update self.im from a numpy array."""
        self.image = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.image)

    def result(self) -> npt.NDArray:
        """Return annotated image as array."""
        return np.asarray(self.image)

    def show(self, title: str | None = None) -> None:
        """Show the annotated image."""
        im = Image.fromarray(
            np.asarray(self.image)[..., ::-1]
        )  # Convert numpy array to PIL Image with RGB to BGR
        im.show(title=title)

    def save(self, filename: str = "image.jpg") -> None:
        """Save the annotated image to 'filename'."""
        cv2.imwrite(filename, np.asarray(self.image))

    def get_bbox_dimension(
        self, bbox: tuple[float, float, float, float]
    ) -> tuple[float, float, float]:
        """Calculate the area of a bounding box.

        Args:
            bbox (tuple): Bounding box coordinates in the format (x_min, y_min, x_max, y_max).

        Returns:
            width (float): Width of the bounding box.
            height (float): Height of the bounding box.
            area (float): Area enclosed by the bounding box.
        """
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        return width, height, width * height


def plot_images(
    images: torch.Tensor | np.ndarray,
    batch_idx: torch.Tensor | np.ndarray,
    cls: torch.Tensor | np.ndarray = np.zeros(0, dtype=np.float32),
    bboxes: torch.Tensor | np.ndarray = np.zeros(0, dtype=np.float32),
    confs: torch.Tensor | np.ndarray | None = None,
    masks: torch.Tensor | np.ndarray = np.zeros(0, dtype=np.uint8),
    fname: str = "images.jpg",
    names: dict[int, str] | None = None,
    max_size: int = 1920,
    max_subplots: int = 16,
    save: bool = True,
    conf_thres: float = 0.1,
) -> np.ndarray | None:
    """Plot image grid with labels, bounding boxes, masks, and keypoints.

    Args:
        images: Batch of images to plot. Shape: (batch_size, channels, height, width).
        batch_idx: Batch indices for each detection. Shape: (num_detections,).
        cls: Class labels for each detection. Shape: (num_detections,).
        bboxes: Bounding boxes for each detection. Shape: (num_detections, 4)
        confs: Confidence scores for each detection. Shape: (num_detections,).
        masks: Instance segmentation masks. Shape: (num_detections, height, width)
            or (1, height, width).
        fname: Output filename for the plotted image grid.
        names: Dictionary mapping class indices to class names.
        max_size: Maximum size of the output image grid.
        max_subplots: Maximum number of subplots in the image grid.
        save: Whether to save the plotted image grid to a file.
        conf_thres: Confidence threshold for displaying detections.

    Returns:
        np.ndarray: Plotted image grid as a numpy array if save is False, None otherwise.

    Note:
        This function supports both tensor and numpy array inputs. It will automatically
        convert tensor inputs to numpy arrays for processing.
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(cls, torch.Tensor):
        cls = cls.cpu().numpy()
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy().astype(int)
    if isinstance(batch_idx, torch.Tensor):
        batch_idx = batch_idx.cpu().numpy()

    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs**0.5)  # number of subplots (square)
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        mosaic[y : y + h, x : x + w, :] = images[i].transpose(1, 2, 0)

    # Resize (optional)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    font_size = int((h + w) * ns * 0.01)
    annotator = Annotator(
        mosaic,
        line_width=round(font_size / 10),
        font_size=font_size,
        pil=True,
    )
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if len(cls) > 0:
            idx = batch_idx == i
            classes = cls[idx].astype("int")
            labels = confs is None

            if len(bboxes):
                boxes = bboxes[idx]
                conf = (
                    confs[idx] if confs is not None else None
                )  # check for confidence presence (label vs pred)
                if len(boxes):
                    if boxes[:, :4].max() <= 1.1:  # if normalized with tolerance 0.1
                        boxes[..., [0, 2]] *= w  # scale to pixels
                        boxes[..., [1, 3]] *= h
                    elif scale < 1:  # absolute coords need scale if image scales
                        boxes[..., :4] *= scale
                boxes[..., 0] += x
                boxes[..., 1] += y
                boxes = ops.xywh2xyxy(boxes)
                for j, box in enumerate(boxes.astype(np.int64).tolist()):
                    c = classes[j]
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    if labels or conf[j] > conf_thres:
                        label = f"{c}" if labels else f"{c} {conf[j]:.1f}"
                        annotator.box_label(box, label, color=color)

            elif len(classes):
                for c in classes:
                    color = colors(c)
                    class_text = names.get(c, c) if names else c
                    annotator.text((x, y), f"{class_text}", txt_color=color, box_style=True)

            # Plot masks
            if len(masks):
                if idx.shape[0] == masks.shape[0]:  # overlap_masks=False
                    image_masks = masks[idx]
                else:  # overlap_masks=True
                    image_masks = masks[[i]]  # (1, 640, 640)
                    nl = idx.sum()
                    index = np.arange(nl).reshape((nl, 1, 1)) + 1
                    image_masks = np.repeat(image_masks, nl, axis=0)
                    image_masks = np.where(image_masks == index, 1.0, 0.0)

                im = np.asarray(annotator.image).copy()
                for j in range(len(image_masks)):
                    if labels or conf[j] > conf_thres:
                        color = colors(classes[j])
                        mh, mw = image_masks[j].shape
                        if mh != h or mw != w:
                            mask = image_masks[j].astype(np.uint8)
                            mask = cv2.resize(mask, (w, h))
                            mask = mask.astype(bool)
                        else:
                            mask = image_masks[j].astype(bool)
                        with contextlib.suppress(Exception):
                            im[y : y + h, x : x + w, :][mask] = (
                                im[y : y + h, x : x + w, :][mask] * 0.4 + np.array(color) * 0.6
                            )
                annotator.fromarray(im)
    if not save:
        return np.asarray(annotator.image)
    annotator.image.save(fname)
    return None


def _get_case_base_volume(image: torch.Tensor | npt.NDArray[np.float32]) -> torch.Tensor:
    return _get_case_plot_volume(image, image_mode="default")


def _get_case_plot_volume(
    image: torch.Tensor | npt.NDArray[np.float32],
    image_mode: Literal["default", "channel_difference"] = "default",
    difference_channels: tuple[int, int] = (2, 0),
) -> torch.Tensor:
    image_tensor = torch.as_tensor(image, dtype=torch.float32)
    if image_tensor.ndim != 4:
        msg = f"Expected case image with shape (C, D, H, W), got {tuple(image_tensor.shape)}"
        raise ValueError(msg)
    if image_mode == "channel_difference":
        minuend_channel, subtrahend_channel = difference_channels
        if max(minuend_channel, subtrahend_channel) >= image_tensor.shape[0]:
            LOGGER.warning(
                "Validation plot difference channels %s requested, but only %d channel(s) are "
                "available. Falling back to the default plot image.",
                difference_channels,
                image_tensor.shape[0],
            )
        else:
            return image_tensor[minuend_channel] - image_tensor[subtrahend_channel]
    if image_tensor.shape[0] > 1:
        return image_tensor[1] - image_tensor[0]
    return image_tensor[0]


def _project_case_volume(
    image: torch.Tensor | npt.NDArray[np.float32],
    plane: Literal["xy", "xz", "yz"],
    projection_mode: Literal["max", "mean"],
    image_mode: Literal["default", "channel_difference"] = "default",
    difference_channels: tuple[int, int] = (2, 0),
    upper_quantile: float = 0.995,
) -> npt.NDArray[np.float32]:
    volume = _get_case_plot_volume(
        image,
        image_mode=image_mode,
        difference_channels=difference_channels,
    )
    axis = {"xy": 0, "xz": 1, "yz": 2}[plane]
    projection = volume.amax(dim=axis) if projection_mode == "max" else volume.mean(dim=axis)
    projection = projection.to(torch.float32)
    finite_values = projection[torch.isfinite(projection)]
    if finite_values.numel() == 0:
        return torch.zeros_like(projection).cpu().numpy()

    scale = torch.quantile(finite_values, upper_quantile)
    if not torch.isfinite(scale) or scale <= 0:
        scale = finite_values.max()
    if not torch.isfinite(scale) or scale <= 0:
        scale = torch.tensor(1.0, device=projection.device)

    return projection.div(scale).clamp(0, 1).cpu().numpy()


def _project_boxes_for_plane(
    boxes_xyzxyz: torch.Tensor | npt.NDArray[np.float32],
    plane: Literal["xy", "xz", "yz"],
) -> npt.NDArray[np.float32]:
    if isinstance(boxes_xyzxyz, torch.Tensor):
        if boxes_xyzxyz.numel() == 0:
            return np.zeros((0, 4), dtype=np.float32)
        boxes = boxes_xyzxyz.detach().to(torch.float32).cpu().numpy()
    else:
        boxes = np.asarray(boxes_xyzxyz, dtype=np.float32)
        if boxes.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
    indices = {"xy": (0, 1, 3, 4), "xz": (0, 2, 3, 5), "yz": (1, 2, 4, 5)}[plane]
    return boxes[:, indices]


def _annotate_projected_boxes(
    axis: plt.Axes,
    boxes_xyzxyz: torch.Tensor | npt.NDArray[np.float32],
    classes: torch.Tensor | npt.NDArray[np.float32],
    plane: Literal["xy", "xz", "yz"],
    names: dict[int, str] | None,
    confs: torch.Tensor | npt.NDArray[np.float32] | None = None,
    conf_thres: float = 0.1,
) -> None:
    boxes_2d = _project_boxes_for_plane(boxes_xyzxyz, plane)
    if isinstance(classes, torch.Tensor):
        classes_np = classes.detach().to(torch.int64).cpu().numpy()
    else:
        classes_np = np.asarray(classes, dtype=np.int64)
    if confs is None:
        confs_np = None
    elif isinstance(confs, torch.Tensor):
        confs_np = confs.detach().to(torch.float32).cpu().numpy()
    else:
        confs_np = np.asarray(confs, dtype=np.float32)

    for index, box in enumerate(boxes_2d):
        if confs_np is not None and confs_np[index] <= conf_thres:
            continue
        class_index = int(classes_np[index])
        color = np.array(colors(class_index), dtype=np.float32) / 255.0
        label = names.get(class_index, str(class_index)) if names else str(class_index)
        if confs_np is not None:
            label = f"{label} {confs_np[index]:.2f}"
        x1, y1, x2, y2 = box.tolist()
        axis.add_patch(
            Rectangle(
                (x1, y1),
                max(x2 - x1, 1.0),
                max(y2 - y1, 1.0),
                fill=False,
                edgecolor=color,
                linewidth=1.5,
            )
        )
        axis.text(
            x1,
            max(y1 - 2.0, 0.0),
            label,
            color="white",
            fontsize=8,
            va="bottom",
            bbox={"facecolor": color, "edgecolor": "none", "pad": 1.5},
        )


def plot_validation_case(
    image: torch.Tensor | npt.NDArray[np.float32],
    reference_boxes: torch.Tensor | npt.NDArray[np.float32],
    reference_classes: torch.Tensor | npt.NDArray[np.float32],
    predicted_boxes: torch.Tensor | npt.NDArray[np.float32],
    predicted_classes: torch.Tensor | npt.NDArray[np.float32],
    fname: str | Path,
    names: dict[int, str] | None = None,
    case_name: str | None = None,
    prediction_confs: torch.Tensor | npt.NDArray[np.float32] | None = None,
    projection_mode: Literal["max", "mean"] = "max",
    image_mode: Literal["default", "channel_difference"] = "default",
    difference_channels: tuple[int, int] = (2, 0),
    conf_thres: float = 0.1,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), tight_layout=True)
    plane_titles = ("xy", "xz", "yz")

    for row_index, (row_title, boxes, classes, confs) in enumerate(
        (
            ("Reference", reference_boxes, reference_classes, None),
            ("Prediction", predicted_boxes, predicted_classes, prediction_confs),
        )
    ):
        for col_index, plane in enumerate(plane_titles):
            axis = axes[row_index, col_index]
            axis.imshow(
                _project_case_volume(
                    image,
                    plane,
                    projection_mode,
                    image_mode=image_mode,
                    difference_channels=difference_channels,
                ),
                cmap="gray",
                interpolation="lanczos",
                resample=True,
            )
            _annotate_projected_boxes(
                axis,
                boxes,
                classes,
                plane,
                names,
                confs=confs,
                conf_thres=conf_thres,
            )
            axis.set_title(f"{row_title} {plane.upper()}")
            axis.set_xticks([])
            axis.set_yticks([])

    if case_name:
        image_mode_suffix = ""
        if image_mode == "channel_difference":
            image_mode_suffix = f", ch{difference_channels[0]}-ch{difference_channels[1]}"
        elif image_mode != "default":
            image_mode_suffix = f", {image_mode}"
        figure.suptitle(f"{case_name} ({projection_mode} projection{image_mode_suffix})")
        figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(figure)


def plot_results(
    file: Path | str | None = "path/to/results.csv",
    directory: Path | str = "",
) -> None:
    """Plot training results from a results CSV file.

    Plots are saved as 'results.png' in the directory where the CSV is located.

    Args:
        file (str, optional): Path to the CSV file containing the training results. Defaults to
            'path/to/results.csv'.
        directory (str, optional): Directory where CSV file is located if 'file' is not provided.
            Defaults to ''.

    Example:
        ```python
        from ultralytics.utils.plotting import plot_results

        plot_results("path/to/results.csv", segment=True)
        ```
    """
    save_dir = Path(file).parent if file else Path(directory)
    fig, ax = plt.subplots(3, 5, figsize=(16, 7), tight_layout=True, sharex="col")
    indices = [2, 3, 4, 5, 6, 12, 13, 14, 15, 16, 7, 8, 9, 10, 11]

    ax = ax.ravel()
    files = list(save_dir.glob("results*.csv"))
    assert files, f"No results.csv files found in {save_dir.resolve()}, nothing to plot."
    for f in files:
        data = pd.read_csv(f)
        column_names = [name.strip() for name in data.columns]
        epochs = data.to_numpy()[:, 0]
        for plot_index, csv_index in enumerate(indices):
            try:
                y = data.to_numpy()[:, csv_index].astype("float")
                ax[plot_index].plot(epochs, y, label=f.stem, linewidth=2)
                ax[plot_index].set_ylabel(column_names[csv_index])
                if "metrics" in column_names[csv_index]:
                    ax[plot_index].set_xlabel("Epochs")
                if "loss" in column_names[csv_index]:
                    clean = y[np.isfinite(y)]
                    if clean.size > 0:
                        ax[plot_index].set_ylim(
                            np.min(clean) * 0.95,
                            np.percentile(clean, 95) * 1.1,
                        )
            except Exception as e:
                LOGGER.warning(f"WARNING: Plotting error for {f} ({column_names[csv_index]}): {e}")
        ax[0].legend()
        fname = save_dir / "results.png"
        fig.savefig(fname, dpi=200)
        plt.close()


def output_to_target(
    output: list[torch.Tensor], max_det: int = 300, axis: int = 2
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Convert model output to plotting target format [batch_id, class_id, x, y, z, w, h, d conf].

    Results are flattened to 2d for easier visualization.
    """
    idx = [i for i in range(6) if i % 3 != axis]
    targets: list[torch.Tensor] = []
    for i, o in enumerate(output):
        box, conf, cls = o[:max_det, :8].cpu().split((6, 1, 1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, ops.xyxy2xywh(box[:, idx]), conf), 1))
    targets_ = torch.cat(targets, 0).numpy()
    return targets_[:, 0], targets_[:, 1], targets_[:, 2:-1], targets_[:, -1]
