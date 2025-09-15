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
import torch
from PIL import Image, ImageDraw, ImageFont

from . import ops

if TYPE_CHECKING:
    from collections.abc import Callable

LOGGER = logging.getLogger(__name__)


class Colors:
    """Ultralytics color palette.

    https://docs.ultralytics.com/reference/utils/plotting/#ultralytics.utils.plotting.Colors.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.

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


colors = Colors()  # create instance for 'from utils.plots import colors'


class Annotator:
    """Ultralytics Annotator for train/val mosaics and JPGs and predictions annotations.

    Attributes:
        im (Image.Image or numpy array): The image to annotate.
        pil (bool): Whether to use PIL or cv2 for drawing annotations.
        font (ImageFont.truetype or ImageFont.load_default): Font used for text annotations.
        lw (float): Line width for drawing.
        limb_color (List[int]): Color palette for limbs.
        kpt_color (List[int]): Color palette for keypoints.
    """

    def __init__(
        self,
        im: Image.Image | npt.NDArray[np.uint8 | np.float32],
        line_width: int | None = None,
        font_size: float | None = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        # example: str = "abc",
    ) -> None:
        """Initialize the Annotator class with image and line width along with color palette for keypoints and limbs."""
        self.lw: int = line_width or max(
            round(sum(im.size if isinstance(im, Image.Image) else im.shape) / 2 * 0.003), 2
        )
        self.im: Image.Image | npt.NDArray[np.uint8 | np.float32]
        if isinstance(im, Image.Image) or pil:  # use PIL
            self.pil = True
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw: ImageDraw.ImageDraw = ImageDraw.Draw(self.im)
            try:
                # font = check_font("Arial.Unicode.ttf" if non_ascii else font)
                size = font_size or max(round(sum(self.im.size) / 2 * 0.035), 12)
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
            self.im = im if im.flags.writeable else im.copy()
            self.tf: int = max(self.lw - 1, 1)  # font thickness
            self.sf: float = self.lw / 3  # font scale
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

    def circle_label(
        self,
        box: tuple[float, float, float, float],
        label: str = "",
        color: tuple[int, int, int] = (128, 128, 128),
        txt_color: tuple[int, int, int] = (255, 255, 255),
        margin: int = 2,
    ) -> None:
        """Draws a label with a background circle centered within a given bounding box.

        Args:
            box (tuple): The bounding box coordinates (x1, y1, x2, y2).
            label (str): The text label to be displayed.
            color (tuple, optional): The background color of the rectangle (B, G, R).
            txt_color (tuple, optional): The color of the text (R, G, B).
            margin (int, optional): The margin between the text and the rectangle border.
        """
        # If label have more than 3 characters, skip other characters, due to circle size
        if len(label) > 3:
            print(
                f"Length of label is {len(label)}, initial 3 label characters will be considered for circle annotation!"
            )
            label = label[:3]

        # Calculate the center of the box
        x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        # Get the text size
        text_size = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, self.sf - 0.15, self.tf)[
            0
        ]
        # Calculate the required radius to fit the text with the margin
        required_radius = int(((text_size[0] ** 2 + text_size[1] ** 2) ** 0.5) / 2) + margin
        # Draw the circle with the required radius
        cv2.circle(self.im, (x_center, y_center), required_radius, color, -1)
        # Calculate the position for the text
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2
        # Draw the text
        cv2.putText(
            self.im,
            str(label),
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.sf - 0.15,
            self.get_txt_color(color, txt_color),
            self.tf,
            lineType=cv2.LINE_AA,
        )

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
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf - 0.1, self.tf)[0]
        # Calculate the top-left corner of the text (to center it)
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2
        # Calculate the coordinates of the background rectangle
        rect_x1 = text_x - margin
        rect_y1 = text_y - text_size[1] - margin
        rect_x2 = text_x + text_size[0] + margin
        rect_y2 = text_y + margin
        # Draw the background rectangle
        cv2.rectangle(self.im, (rect_x1, rect_y1), (rect_x2, rect_y2), color, -1)
        # Draw the text on top of the rectangle
        cv2.putText(
            self.im,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.sf - 0.1,
            self.get_txt_color(color, txt_color),
            self.tf,
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
        if self.pil:  # or not is_ascii(label):
            p1 = (box[0], box[1])
            self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.getsize(label)  # text width, height
                outside = p1[1] >= h  # label fits outside box
                if (
                    p1[0] > self.im.size[0] - w
                ):  # size is (w, h), check if label extend beyond right side of image
                    p1 = self.im.size[0] - w, p1[1]
                self.draw.rectangle(
                    (
                        p1[0],
                        p1[1] - h if outside else p1[1],
                        p1[0] + w + 1,
                        p1[1] + 1 if outside else p1[1] + h + 1,
                    ),
                    fill=color,
                )
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text(
                    (p1[0], p1[1] - h if outside else p1[1]), label, fill=txt_color, font=self.font
                )
        else:  # cv2
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[
                    0
                ]  # text width, height
                h += 3  # add pixels to pad text
                outside = p1[1] >= h  # label fits outside box
                if (
                    p1[0] > self.im.shape[1] - w
                ):  # shape is (h, w), check if label extend beyond right side of image
                    p1 = self.im.shape[1] - w, p1[1]
                p2 = p1[0] + w, p1[1] - h if outside else p1[1] + h
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    self.im,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h - 1),
                    0,
                    self.sf,
                    txt_color,
                    thickness=self.tf,
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
            w, h = self.getsize(text)  # text width, height
            xy[1] += 1 - h
        if self.pil:
            if box_style:
                w, h = self.getsize(text)
                self.draw.rectangle((xy[0], xy[1], xy[0] + w + 1, xy[1] + h + 1), fill=txt_color)
                # Using `txt_color` for background and draw fg with white color
                txt_color = (255, 255, 255)
            if "\n" in text:
                lines = text.split("\n")
                _, h = self.getsize(text)
                for line in lines:
                    self.draw.text(xy, line, fill=txt_color, font=self.font)
                    xy[1] += h
            else:
                self.draw.text(xy, text, fill=txt_color, font=self.font)
        else:
            if box_style:
                w, h = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[
                    0
                ]  # text width, height
                h += 3  # add pixels to pad text
                outside = xy[1] >= h  # label fits outside box
                p2 = xy[0] + w, xy[1] - h if outside else xy[1] + h
                cv2.rectangle(self.im, xy, p2, txt_color, -1, cv2.LINE_AA)  # filled
                # Using `txt_color` for background and draw fg with white color
                txt_color = (255, 255, 255)
            cv2.putText(
                self.im, text, xy, 0, self.sf, txt_color, thickness=self.tf, lineType=cv2.LINE_AA
            )

    def fromarray(self, im: npt.NDArray[np.uint8 | np.float32]) -> None:
        """Update self.im from a numpy array."""
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self) -> npt.NDArray:
        """Return annotated image as array."""
        return np.asarray(self.im)

    def show(self, title: str | None = None) -> None:
        """Show the annotated image."""
        im = Image.fromarray(
            np.asarray(self.im)[..., ::-1]
        )  # Convert numpy array to PIL Image with RGB to BGR
        im.show(title=title)

    def save(self, filename: str = "image.jpg") -> None:
        """Save the annotated image to 'filename'."""
        cv2.imwrite(filename, np.asarray(self.im))

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
        masks: Instance segmentation masks. Shape: (num_detections, height, width) or (1, height, width).
        paths: List of file paths for each image in the batch.
        fname: Output filename for the plotted image grid.
        names: Dictionary mapping class indices to class names.
        on_plot: Optional callback function to be called after saving the plot.
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
                    c = names.get(c, c) if names else c
                    annotator.text((x, y), f"{c}", txt_color=color, box_style=True)

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

                im = np.asarray(annotator.im).copy()
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
        return np.asarray(annotator.im)
    annotator.im.save(fname)
    return None


def plot_results(
    file: Path | str | None = "path/to/results.csv",
    dir: Path | str = "",
):
    """Plot training results from a results CSV file.

    Plots are saved as 'results.png' in the directory where the CSV is located.

    Args:
        file (str, optional): Path to the CSV file containing the training results. Defaults to
            'path/to/results.csv'.
        dir (str, optional): Directory where the CSV file is located if 'file' is not provided.
            Defaults to ''.

    Example:
        ```python
        from ultralytics.utils.plotting import plot_results

        plot_results("path/to/results.csv", segment=True)
        ```
    """
    import pandas as pd  # scope for faster import of parent modules

    save_dir = Path(file).parent if file else Path(dir)
    fig, ax = plt.subplots(3, 5, figsize=(16, 7), tight_layout=True, sharex='col')
    indices = [
        2,  3,  4,  5,  6,
        12, 13, 14, 15, 16,
        7,  8,  9,  10, 11
    ]

    ax = ax.ravel()
    files = list(save_dir.glob("results*.csv"))
    assert files, f"No results.csv files found in {save_dir.resolve()}, nothing to plot."
    for f in files:
        data = pd.read_csv(f)
        column_names = [name.strip() for name in data.columns]
        epochs = data.values[:, 0]
        for plot_index, csv_index in enumerate(indices):
            try:
                y = data.values[:, csv_index].astype("float")
                ax[plot_index].plot(epochs, y, label=f.stem, linewidth=2)
                ax[plot_index].set_ylabel(column_names[csv_index])
                if "metrics" in column_names[csv_index]:
                    ax[plot_index].set_xlabel("Epochs")
                if "loss" in column_names[csv_index]:
                    ax[plot_index].set_ylim(np.min(y) - 0.1, np.percentile(y, 95) + 0.1)
            except Exception as e:
                LOGGER.warning(f"WARNING: Plotting error for {f} ({column_names[csv_index]}): {e}")
        ax[0].legend()
        fname = save_dir / "results.png"
        fig.savefig(fname, dpi=200)
        plt.close()


def output_to_target(
    output: list[torch.Tensor], max_det: int = 300, axis: int = 2
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Convert model output to target format [batch_id, class_id, x, y, z, w, h, d conf] for plotting.

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
