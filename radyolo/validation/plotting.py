# Ultralytics YOLO 🚀, AGPL-3.0 license

from __future__ import annotations

import colorsys
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from PIL import Image
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sksurv.metrics import concordance_index_censored

from radyolo.utils import ops

if TYPE_CHECKING:
    from radyolo.data.image_labels import ImageLabelTaskSpec

plt.switch_backend("Agg")

LOGGER = logging.getLogger(__name__)

RESULTS_PLOT_COLUMNS = (
    "train/box_loss",
    "train/dfl_loss",
    "train/cls_obj_loss",
    "train/cls_sep_loss",
    "train/mask_loss",
    "val/box_loss",
    "val/dfl_loss",
    "val/cls_obj_loss",
    "val/cls_sep_loss",
    "val/mask_loss",
    "metrics/fitness",
    "metrics/mAP10",
    "metrics/mAP10-95Agn",
    "metrics/mAP10-95",
    "metrics/dice",
)

IMAGE_LABEL_RESULTS_PLOT_COLUMNS = (
    "train/img_cce",
    "train/img_cst",
    "train/img_cox",
    "val/img_cce",
    "val/img_cst",
    "val/img_cox",
    "metrics/image_label_balanced_accuracy",
    "metrics/image_label_multiclass_roc_auc",
    "metrics/image_label_c_index",
)

IMAGE_LABEL_LOSS_COLUMNS = tuple(
    column_name
    for column_name in IMAGE_LABEL_RESULTS_PLOT_COLUMNS
    if column_name.startswith(("train/", "val/"))
)

BASE_PLOT_HEX_COLORS = (
    "#005b7f",
    "#179c7d",
    "#a6bbc8",
    "#f58220",
    "#1c3f52",
    "#d3c7ae",
    "#008598",
    "#39c1cd",
    "#b2d235",
    "#fdb913",
    "#bb0056",
    "#7c154d",
)

MAIN_CURVE_COLOR = "#005b7f"
BACKGROUND_CURVE_COLOR = "#a6bbc8"
OPERATING_POINT_COLOR = "#f58220"
BEST_POINT_COLOR = "#179c7d"
CONFUSION_MATRIX_BASE_COLOR = "#005b7f"
IMAGE_LABEL_SUMMARY_FNAME = "validation_image_labels.png"


def _hex_to_rgb(color_hex: str) -> tuple[int, int, int]:
    """Convert a hex color string to an RGB tuple."""
    color_hex = color_hex.removeprefix("#")
    return tuple(int(color_hex[index : index + 2], 16) for index in (0, 2, 4))


def _rgb_to_mpl(color_rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert 0-255 RGB values to matplotlib's 0-1 float range."""
    return tuple(channel / 255.0 for channel in color_rgb)


def palette_rgb(index: int) -> tuple[int, int, int]:
    """Return a palette color with deterministic overflow variations."""
    base_index = index % len(BASE_PLOT_HEX_COLORS)
    cycle_index = index // len(BASE_PLOT_HEX_COLORS)
    base_rgb = _hex_to_rgb(BASE_PLOT_HEX_COLORS[base_index])
    if cycle_index == 0:
        return base_rgb

    hue, saturation, value = colorsys.rgb_to_hsv(*_rgb_to_mpl(base_rgb))
    hue_shift_direction = -1.0 if cycle_index % 2 else 1.0
    hue_shift = (0.055 + 0.015 * ((cycle_index - 1) % 3)) * hue_shift_direction
    saturation_scale = (0.88, 1.08, 0.78)[(cycle_index - 1) % 3]
    value_scale = (1.0, 0.92, 1.08)[(cycle_index - 1) % 3]
    adjusted_rgb = colorsys.hsv_to_rgb(
        (hue + hue_shift) % 1.0,
        min(1.0, max(0.35, saturation * saturation_scale)),
        min(1.0, max(0.3, value * value_scale)),
    )
    return tuple(round(channel * 255.0) for channel in adjusted_rgb)


def palette_mpl(index: int) -> tuple[float, float, float]:
    """Return a palette color as a matplotlib RGB triple."""
    return _rgb_to_mpl(palette_rgb(index))


def named_plot_color(name: Literal["mean", "background", "operating", "best"]) -> str:
    """Resolve a semantic plotting color name to its hex color."""
    return {
        "mean": MAIN_CURVE_COLOR,
        "background": BACKGROUND_CURVE_COLOR,
        "operating": OPERATING_POINT_COLOR,
        "best": BEST_POINT_COLOR,
    }[name]


def confusion_matrix_cmap() -> LinearSegmentedColormap:
    """Build the custom colormap used for confusion matrix plots."""
    base_hue, _, base_saturation = colorsys.rgb_to_hls(
        *_rgb_to_mpl(_hex_to_rgb(CONFUSION_MATRIX_BASE_COLOR))
    )
    lightness_values = np.linspace(0.97, 0.2, 7)
    saturation_values = np.linspace(0.18, max(base_saturation, 0.95), 7)
    color_stops = [
        colorsys.hls_to_rgb(base_hue, float(lightness), float(saturation))
        for lightness, saturation in zip(lightness_values, saturation_values, strict=True)
    ]
    return LinearSegmentedColormap.from_list(
        "validation_confusion_matrix",
        color_stops,
    )


class Colors:
    """Color palette.

    This class provides methods to work with the color palette, including converting
    hex color codes to RGB values.
    """

    def __init__(self) -> None:
        """Initialize the shared plotting palette used across validation artifacts."""
        self.palette: list[tuple[int, int, int]] = [
            _hex_to_rgb(color) for color in BASE_PLOT_HEX_COLORS
        ]
        self.n: int = len(self.palette)

    def __call__(self, i: int | str, bgr: bool = False) -> tuple[int, int, int]:
        """Return a palette color, generating distinguishable overflow variants as needed."""
        c: tuple[int, int, int] = palette_rgb(int(i))
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h: str) -> tuple[int, int, int]:
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


def _label_text_color(
    color: tuple[int, int, int],
    default_text_color: tuple[int, int, int] = (255, 255, 255),
) -> tuple[int, int, int]:
    """Select readable label text color from the background luminance."""
    relative_luminance = (0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]) / 255.0
    if relative_luminance >= 0.62:
        return 104, 31, 17
    if relative_luminance <= 0.3:
        return 255, 255, 255
    return default_text_color


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
    line_width = max(round((h + w) * ns * 0.001), 1)
    font_thickness = max(line_width - 1, 1)
    font_scale = max(line_width / 3, 0.4)
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        cv2.rectangle(mosaic, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)
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
                        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                        cv2.rectangle(
                            mosaic,
                            p1,
                            p2,
                            color,
                            thickness=line_width,
                            lineType=cv2.LINE_AA,
                        )
                        if label:
                            text_width, text_height = cv2.getTextSize(
                                label,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=font_scale,
                                thickness=font_thickness,
                            )[0]
                            text_height += 3
                            outside = p1[1] >= text_height
                            p1_text = (
                                min(p1[0], max(mosaic.shape[1] - text_width, 0)),
                                p1[1],
                            )
                            p2_text = (
                                p1_text[0] + text_width,
                                p1_text[1] - text_height if outside else p1_text[1] + text_height,
                            )
                            cv2.rectangle(mosaic, p1_text, p2_text, color, -1, cv2.LINE_AA)
                            cv2.putText(
                                mosaic,
                                label,
                                (
                                    p1_text[0],
                                    p1_text[1] - 2 if outside else p1_text[1] + text_height - 1,
                                ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale,
                                _label_text_color(color),
                                thickness=font_thickness,
                                lineType=cv2.LINE_AA,
                            )

            elif len(classes):
                for c in classes:
                    color = colors(c)
                    class_text = names.get(c, c) if names else c
                    class_text = f"{class_text}"
                    text_width, text_height = cv2.getTextSize(
                        class_text,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale,
                        thickness=font_thickness,
                    )[0]
                    text_height += 3
                    cv2.rectangle(
                        mosaic,
                        (x, y),
                        (x + text_width, y + text_height),
                        color,
                        -1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        mosaic,
                        class_text,
                        (x, y + text_height - 1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        thickness=font_thickness,
                        lineType=cv2.LINE_AA,
                    )

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
                        if not mask.any():
                            continue
                        tile = mosaic[y : y + h, x : x + w, :]
                        tile[mask] = tile[mask] * 0.4 + np.array(color, dtype=np.float32) * 0.6
    if not save:
        return mosaic
    Image.fromarray(mosaic).save(fname)
    return None


def _get_case_plot_volume(
    image: torch.Tensor | npt.NDArray[np.float32],
    image_mode: Literal["default", "channel_difference"] = "default",
    difference_channels: tuple[int, int] = (2, 0),
) -> torch.Tensor:
    """Select the image volume used for validation-case projections."""
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
    """Project a 3D case volume onto a 2D plane and normalize intensities."""
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
    """Project 3D xyzxyz boxes to 2D xyxy boxes for a selected plane."""
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
    """Draw projected boxes and labels on a matplotlib axis."""
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
        if index >= len(classes_np) or not np.isfinite(box).all():
            continue
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
    """Save a 2x3 figure comparing reference and predicted boxes per plane."""
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    plane_titles = ("xy", "xz", "yz")

    for row_index, (row_title, boxes, classes, confs) in enumerate(
        (
            ("Reference", reference_boxes, reference_classes, None),
            ("Prediction", predicted_boxes, predicted_classes, prediction_confs),
        )
    ):
        for col_index, plane in enumerate(plane_titles):
            axis = axes[row_index, col_index]
            projection = _project_case_volume(
                image,
                plane,
                projection_mode,
                image_mode=image_mode,
                difference_channels=difference_channels,
            )
            axis.imshow(
                projection,
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
    figure.savefig(fname, dpi=120)
    plt.close(figure)


def plot_results(
    file: Path | str | None = "path/to/results.csv",
    directory: Path | str = "",
) -> None:
    """Plot training results from a results CSV file.

    Plots are saved as 'loss_curves.png' in the directory where the CSV is located.

    Args:
        file (str, optional): Path to the CSV file containing the training results. Defaults to
            'path/to/results.csv'.
        directory (str, optional): Directory where CSV file is located if 'file' is not provided.
            Defaults to ''.
    """
    save_dir = Path(file).parent if file else Path(directory)
    files = list(save_dir.glob("results*.csv"))
    assert files, f"No results.csv files found in {save_dir.resolve()}, nothing to plot."
    _plot_results_grid(
        files=files,
        columns=RESULTS_PLOT_COLUMNS,
        save_path=save_dir / "loss_curves.png",
        shape=(3, 5),
        figsize=(16, 7),
    )
    _plot_results_grid(
        files=files,
        columns=IMAGE_LABEL_RESULTS_PLOT_COLUMNS,
        save_path=save_dir / "loss_curves_image_labels.png",
        shape=(3, 3),
        figsize=(12, 7),
    )


def _plot_results_grid(
    *,
    files: list[Path],
    columns: tuple[str, ...],
    save_path: Path,
    shape: tuple[int, int],
    figsize: tuple[float, float],
) -> None:
    """Plot selected metric columns from one or more results CSV files."""
    fig, ax = plt.subplots(*shape, figsize=figsize, tight_layout=True, sharex="col")
    axes = np.atleast_1d(ax).ravel()
    has_any_series = False
    visible_columns = [False] * len(columns)
    fitness_axis: plt.Axes | None = None
    best_epoch_legend_added = False

    for f in files:
        file_color = palette_mpl(files.index(f))
        data = pd.read_csv(f).rename(columns=lambda name: name.strip())
        epochs = data.iloc[:, 0].to_numpy(dtype=float)
        for plot_index, column_name in enumerate(columns):
            try:
                if column_name not in data.columns:
                    continue
                visible_columns[plot_index] = True
                has_any_series = True
                y = data[column_name].to_numpy(dtype=float)
                plotted_line = axes[plot_index].plot(
                    epochs,
                    y,
                    label=f.stem,
                    linewidth=2,
                    color=file_color,
                )[0]
                axes[plot_index].set_ylabel(column_name)
                axes[plot_index].grid(True, axis="both", alpha=0.2, linewidth=0.6)
                if "metrics" in column_name:
                    axes[plot_index].set_xlabel("Epochs")
                    axes[plot_index].set_ylim(0, 1)
                if column_name == "metrics/fitness":
                    fitness_axis = axes[plot_index]
                    best_epoch = float(epochs[int(np.argmax(y))])
                    axes[plot_index].axvline(
                        best_epoch,
                        color=plotted_line.get_color(),
                        alpha=0.3,
                        linewidth=1.0,
                        linestyle="--",
                        label="Best epoch" if not best_epoch_legend_added else "_nolegend_",
                    )
                    best_epoch_legend_added = True
                if "loss" in column_name or column_name in IMAGE_LABEL_LOSS_COLUMNS:
                    clean = y[np.isfinite(y)]
                    if clean.size > 0:
                        y_min = np.min(clean) * 0.95
                        y_max = np.percentile(clean, 95) * 1.1
                        if y_min < y_max:
                            axes[plot_index].set_ylim(y_min, y_max)
            except Exception as e:
                LOGGER.warning("WARNING: Plotting error for %s (%s): %s", f, column_name, e)

    for plot_index, axis in enumerate(axes):
        axis.set_visible(plot_index < len(columns) and visible_columns[plot_index])

    if not has_any_series:
        plt.close(fig)
        return

    if fitness_axis is not None and fitness_axis.get_visible():
        handles, labels = fitness_axis.get_legend_handles_labels()
        fitness_axis.legend(handles, labels, loc="best")
    try:
        fig.savefig(save_path, dpi=200)
    except OSError as e:
        LOGGER.warning("Could not save plot %s: %s", save_path, e)
    plt.close(fig)


def _concat_or_empty(
    arrays: list[npt.NDArray],
    *,
    dtype: object,
    shape: tuple[int, ...],
) -> npt.NDArray:
    """Concatenate arrays or return an empty array with the requested shape."""
    if not arrays:
        return np.zeros(shape, dtype=dtype)
    return np.concatenate(arrays, axis=0)


def _softmax(logits: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute row-wise softmax probabilities from logits."""
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted_logits)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _survival_bin_widths(thresholds: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute bin widths for survival thresholds."""
    if thresholds.size == 1:
        return np.ones_like(thresholds)
    trailing_width = thresholds[-1] - thresholds[-2]
    if trailing_width <= 0:
        trailing_width = 1.0
    return np.diff(thresholds, append=thresholds[-1] + trailing_width)


def _plot_confusion_matrix_axis(
    ax: plt.Axes,
    matrix: npt.NDArray[np.float64],
    *,
    class_names: list[str],
    normalize: bool,
    title: str,
) -> None:
    """Render one confusion-matrix heatmap axis."""
    display_matrix = matrix.astype(np.float64, copy=True)
    if normalize:
        column_sums = display_matrix.sum(axis=0, keepdims=True)
        display_matrix = np.divide(
            display_matrix,
            np.where(column_sums == 0.0, 1.0, column_sums),
            out=np.zeros_like(display_matrix),
            where=column_sums != 0.0,
        )
    display_matrix[display_matrix < 0.005] = np.nan
    sns.heatmap(
        display_matrix,
        ax=ax,
        annot=True,
        fmt=".2f" if normalize else ".0f",
        cmap=confusion_matrix_cmap(),
        square=True,
        vmin=0.0,
        xticklabels=class_names,
        yticklabels=class_names,
    ).set_facecolor((1, 1, 1))
    ax.set_xlabel("Reference")
    ax.set_ylabel("Predicted")
    ax.set_title(title)


def _plot_multiclass_roc_axis(
    ax: plt.Axes,
    probabilities: npt.NDArray[np.float64],
    targets: npt.NDArray[np.int64],
    *,
    class_names: list[str],
    title_prefix: str,
) -> None:
    """Plot per-class and macro-average ROC curves on a provided axis."""
    if probabilities.size == 0 or targets.size == 0:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{title_prefix} ROC Curve")
        ax.text(0.5, 0.5, "No ROC data", ha="center", va="center", transform=ax.transAxes)
        return

    num_classes = probabilities.shape[1]
    one_hot_targets = np.eye(num_classes, dtype=np.int64)[targets]
    auc_values: list[float] = []
    for class_index, class_name in enumerate(class_names):
        class_targets = one_hot_targets[:, class_index]
        if class_targets.min() == class_targets.max():
            continue
        fpr, tpr, _ = roc_curve(class_targets, probabilities[:, class_index])
        class_auc = float(roc_auc_score(class_targets, probabilities[:, class_index]))
        auc_values.append(class_auc)
        ax.plot(
            fpr,
            tpr,
            linewidth=1.5,
            label=f"{class_name} {class_auc:.3f}",
            color=palette_mpl(class_index),
        )

    if not auc_values:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{title_prefix} ROC Curve")
        ax.text(0.5, 0.5, "Undefined ROC", ha="center", va="center", transform=ax.transAxes)
        return

    mean_fpr = np.linspace(0.0, 1.0, 200)
    interpolated_tprs: list[npt.NDArray[np.float64]] = []
    for class_index in range(num_classes):
        class_targets = one_hot_targets[:, class_index]
        if class_targets.min() == class_targets.max():
            continue
        fpr, tpr, _ = roc_curve(class_targets, probabilities[:, class_index])
        interpolated_tprs.append(np.interp(mean_fpr, fpr, tpr))
    if interpolated_tprs:
        mean_tpr = np.mean(interpolated_tprs, axis=0)
        mean_auc = float(np.mean(auc_values))
        ax.plot(
            mean_fpr,
            mean_tpr,
            linewidth=3,
            color=named_plot_color("mean"),
            label=f"macro avg {mean_auc:.3f}",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color=named_plot_color("background"), linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, axis="both", alpha=0.2, linewidth=0.6)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_title(f"{title_prefix} ROC Curve")


def _plot_survival_axis(
    ax: plt.Axes,
    logits: npt.NDArray[np.float64],
    survival_times: npt.NDArray[np.float64],
    event_indicators: npt.NDArray[np.bool_],
    thresholds: npt.NDArray[np.float64],
    *,
    title_prefix: str,
    unit: str | None,
) -> None:
    """Plot survival probability summary curves and report c-index in the title."""
    if logits.size == 0 or survival_times.size == 0 or event_indicators.size == 0:
        ax.set_title(f"{title_prefix} Survival Metric")
        ax.text(
            0.5, 0.5, "No survival metric data", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_xlabel(unit or "Time")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        return

    probabilities = 1.0 / (1.0 + np.exp(-logits))
    all_mean = probabilities.mean(axis=0)
    event_mask = event_indicators.astype(bool)
    censored_mask = np.logical_not(event_mask)
    empirical = np.mean(survival_times[:, None] >= thresholds[None, :], axis=0)

    ax.plot(
        thresholds,
        all_mean,
        linewidth=3,
        color=named_plot_color("mean"),
        label="Predicted mean",
    )
    ax.plot(
        thresholds,
        empirical,
        linewidth=2,
        linestyle="--",
        color=named_plot_color("operating"),
        label="Observed fraction",
    )
    if event_mask.any():
        ax.plot(
            thresholds,
            probabilities[event_mask].mean(axis=0),
            linewidth=1.5,
            color=palette_mpl(0),
            alpha=0.8,
            label="Events mean",
        )
    if censored_mask.any():
        ax.plot(
            thresholds,
            probabilities[censored_mask].mean(axis=0),
            linewidth=1.5,
            color=palette_mpl(1),
            alpha=0.8,
            label="Censored mean",
        )

    risk_scores = -(probabilities * _survival_bin_widths(thresholds)).sum(axis=1)
    c_index = float(concordance_index_censored(event_indicators, survival_times, risk_scores)[0])

    x_label = "Time"
    if unit:
        x_label = f"Time ({unit})"
    ax.set_xlabel(x_label)
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="both", alpha=0.2, linewidth=0.6)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.set_title(f"{title_prefix} Survival Metric (c-index {c_index:.3f})")


def plot_image_label_validation_summary(
    *,
    save_dir: Path,
    task_specs: dict[str, "ImageLabelTaskSpec"],
    multiclass_logits: dict[str, list[npt.NDArray[np.float64]]],
    multiclass_targets: dict[str, list[npt.NDArray[np.int64]]],
    survival_logits: dict[str, list[npt.NDArray[np.float64]]],
    survival_times: dict[str, list[npt.NDArray[np.float64]]],
    survival_events: dict[str, list[npt.NDArray[np.bool_]]],
) -> None:
    """Save a combined validation figure for image-label metrics."""
    multiclass_tasks = [
        (task_name, task_spec)
        for task_name, task_spec in task_specs.items()
        if task_spec.type == "multiclass"
    ]
    survival_tasks = [
        (task_name, task_spec)
        for task_name, task_spec in task_specs.items()
        if task_spec.type == "survival"
    ]
    if not multiclass_tasks and not survival_tasks:
        return

    row_count = len(multiclass_tasks) + len(survival_tasks)
    figure = plt.figure(figsize=(18, max(4.8 * row_count, 4.8)), constrained_layout=True)
    outer_grid = figure.add_gridspec(row_count, 3, hspace=0.06, wspace=0.06)

    current_row = 0
    for task_name, task_spec in multiclass_tasks:
        logits_batches = multiclass_logits.get(task_name, [])
        target_batches = multiclass_targets.get(task_name, [])
        class_ids = sorted(task_spec.classes or {})
        class_names = (
            [task_spec.classes[class_id] for class_id in class_ids]
            if task_spec.classes
            else [str(class_id) for class_id in class_ids]
        )

        ax_norm = figure.add_subplot(outer_grid[current_row, 0])
        ax_raw = figure.add_subplot(outer_grid[current_row, 1])
        ax_roc = figure.add_subplot(outer_grid[current_row, 2])

        if logits_batches and target_batches:
            logits = _concat_or_empty(logits_batches, dtype=np.float64, shape=(0, len(class_ids)))
            targets = _concat_or_empty(target_batches, dtype=np.int64, shape=(0,))
            probabilities = _softmax(logits)
            predictions = probabilities.argmax(axis=1)
            # sklearn returns rows=true(reference), cols=predicted; transpose to keep
            # this project's confusion-matrix convention rows=predicted, cols=reference.
            raw_matrix = sk_confusion_matrix(targets, predictions, labels=class_ids).T
            _plot_confusion_matrix_axis(
                ax_norm,
                raw_matrix,
                class_names=class_names,
                normalize=True,
                title=f"{task_name} Confusion Matrix (Normalized)",
            )
            _plot_confusion_matrix_axis(
                ax_raw,
                raw_matrix,
                class_names=class_names,
                normalize=False,
                title=f"{task_name} Confusion Matrix",
            )
            _plot_multiclass_roc_axis(
                ax_roc,
                probabilities,
                targets,
                class_names=class_names,
                title_prefix=task_name,
            )
        else:
            for axis, title in (
                (ax_norm, f"{task_name} Confusion Matrix (Normalized)"),
                (ax_raw, f"{task_name} Confusion Matrix"),
                (ax_roc, f"{task_name} ROC Curve"),
            ):
                axis.set_title(title)
                axis.text(
                    0.5,
                    0.5,
                    "No image-label data",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                axis.set_xticks([])
                axis.set_yticks([])

        current_row += 1

    for task_name, task_spec in survival_tasks:
        logits_batches = survival_logits.get(task_name, [])
        time_batches = survival_times.get(task_name, [])
        event_batches = survival_events.get(task_name, [])
        ax_survival = figure.add_subplot(outer_grid[current_row, :])

        if logits_batches and time_batches and event_batches:
            thresholds = np.asarray(task_spec.survival_bin_thresholds or (), dtype=np.float64)
            logits = _concat_or_empty(logits_batches, dtype=np.float64, shape=(0, thresholds.size))
            times = _concat_or_empty(time_batches, dtype=np.float64, shape=(0,))
            events = _concat_or_empty(event_batches, dtype=np.bool_, shape=(0,))
            _plot_survival_axis(
                ax_survival,
                logits,
                times,
                events,
                thresholds,
                title_prefix=task_name,
                unit=task_spec.unit,
            )
        else:
            ax_survival.set_title(f"{task_name} Survival Metric")
            ax_survival.text(
                0.5,
                0.5,
                "No image-label data",
                ha="center",
                va="center",
                transform=ax_survival.transAxes,
            )
            ax_survival.set_xticks([])
            ax_survival.set_yticks([])
        current_row += 1

    output_path = save_dir / IMAGE_LABEL_SUMMARY_FNAME
    figure.savefig(output_path, dpi=250)
    plt.close(figure)
