from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import seaborn as sns
import torch
from matplotlib import patheffects
from matplotlib import pyplot as plt

from radyolo.utils.ops import box_iou
from radyolo.validation.plotting import (
    confusion_matrix_cmap,
    named_plot_color,
    palette_mpl,
    save_figure_with_warning,
)

plt.switch_backend("Agg")

LOGGER = logging.getLogger(__name__)
MC_CURVE_SMOOTHING_FRACTION = 0.05
FROC_TARGETS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


class ConfusionMatrix:
    """A class for calculating and updating a confusion matrix for object detection tasks.

    Attributes:
        matrix (np.ndarray): The confusion matrix.
        nc (int): The number of classes.
        conf (float): The confidence threshold for detections.
        iou_thres (float): The Intersection over Union threshold.
    """

    def __init__(
        self,
        nc: int,
        conf: float = 0.1,
        iou_thres: float = 0.2,
    ) -> None:
        """Initialize a confusion matrix that always includes an 'unclassified' slot.

        Layout (indices):
            0..nc-1 : base classes
            nc      : unclassified (for GT class == -1)
            nc+1    : background (FP/FN)
        """
        self.nc = nc  # base class count
        self.unclassified_index = nc
        self.background_index = nc + 1
        size = nc + 2
        self.matrix_: npt.NDArray[np.float32] = np.zeros((size, size), dtype=np.float32)
        self.conf = 0.1 if conf in {None, 0.001} else conf
        self.iou_thres = iou_thres

    def process_batch(
        self, detections: torch.Tensor | None, gt_bboxes: torch.Tensor, gt_cls: torch.Tensor
    ) -> None:
        """Update confusion matrix for object detection task.

        Args:
            detections (Array[N, 7] | Array[N, 8]): Detected bounding boxes and their associated
                information. Each row should contain (x1, y1, z1, x2, y2, z2, conf, class)
            gt_bboxes (Array[M, 6] | Array[N, 7]): Ground truth bounding boxes with
                xyzxyz/xyzxyzr format.
            gt_cls (Array[M]): The class labels.
        """
        # Keep -1 (unclassified) GT boxes if requested, else filter them out as before
        if gt_cls.numel():
            # Keep -1 (unclassified) labels; drop only values < -1
            valid = gt_cls >= -1
            gt_cls = gt_cls[valid]
            gt_bboxes = gt_bboxes[valid]

        if gt_cls.shape[0] == 0:  # no ground-truth boxes after filtering
            if detections is not None:
                detections = detections[detections[:, 6] > self.conf]
                detection_classes = detections[:, 7].int()
                for dc in detection_classes:
                    self.matrix_[dc, self.background_index] += 1  # false positives (background)
            return

        if detections is None:  # no detections: all GT become FN (background)
            gt_classes = gt_cls.int()
            gt_classes = gt_classes.clone()
            gt_classes[gt_classes == -1] = self.unclassified_index
            for gc in gt_classes:
                self.matrix_[self.background_index, gc] += 1  # background FN
            return

        detections = detections[detections[:, 6] > self.conf]
        gt_classes = gt_cls.int()
        # Map -1 to dedicated unclassified index
        mask_unclassified = gt_classes == -1
        if mask_unclassified.any():
            gt_classes = gt_classes.clone()
            gt_classes[mask_unclassified] = self.unclassified_index
        detection_classes = detections[:, 7].int()
        iou = box_iou(gt_bboxes, detections[:, :6])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix_[detection_classes[m1[j]], gc] += 1  # correct match
            else:
                self.matrix_[self.background_index, gc] += 1  # false negative (background)

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix_[dc, self.background_index] += 1  # false positive (background)

    def render(
        self,
        ax: plt.Axes,
        normalize: bool = True,
        names: tuple[str, ...] = (),
    ) -> None:
        """Render the confusion matrix into an existing matplotlib axis."""
        array = self.matrix_ / (
            (self.matrix_.sum(0).reshape(1, -1) + 1e-9) if normalize else 1
        )  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        # Number of *display* classes excludes background but includes unclassified
        display_nc = self.nc + 1  # include unclassified in display (exclude background)
        names_list = list(names)
        if len(names_list) == self.nc:
            names_list.append("unclassified")
        nn = len(names_list)
        labels = (0 < nn < 99) and (nn == display_nc)
        ticklabels = ([*names_list, "background"]) if labels else "auto"
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sns.heatmap(
                array,
                ax=ax,
                annot=display_nc < 30,
                annot_kws={"size": 8},
                cmap=confusion_matrix_cmap(),
                fmt=".2f" if normalize else ".0f",
                square=True,
                vmin=0.0,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            ).set_facecolor((1, 1, 1))
        title = "Confusion Matrix" + " Normalized" * normalize
        ax.set_xlabel("Reference")
        ax.set_ylabel("Predicted")
        ax.set_title(title)


def smooth(y: np.ndarray, fraction: float = 0.05) -> np.ndarray:
    """Box filter of fraction f."""
    num_filtered_elements = round(len(y) * fraction * 2) // 2 + 1  # must be odd
    padding_ones = np.ones(num_filtered_elements // 2)
    y_padded = np.concatenate((padding_ones * y[0], y, padding_ones * y[-1]), 0)
    smoothing_filter = np.ones(num_filtered_elements) / num_filtered_elements
    return np.convolve(y_padded, smoothing_filter, mode="valid")


def _smoothed_mean_confidence_curve(
    py: npt.NDArray,
    px: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Return the smoothed mean curve used for the bold confidence plot line."""
    if not isinstance(py, np.ndarray) or py.size == 0:
        return np.zeros_like(px)
    return smooth(py.mean(0), MC_CURVE_SMOOTHING_FRACTION).astype(np.float32, copy=False)


def _annotate_points(
    ax: plt.Axes,
    x_values: list[float],
    y_values: list[float],
    labels: list[str],
    color: str,
    marker: str = "o",
) -> None:
    """Mark and label operating points on an axis."""
    if not x_values:
        return
    ax.scatter(x_values, y_values, color=color, marker=marker, s=40, zorder=5)
    for x_value, y_value, label in zip(x_values, y_values, labels, strict=True):
        annotation = ax.annotate(
            label,
            xy=(x_value, y_value),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=12,
            color=color,
        )
        annotation.set_path_effects(
            [patheffects.withStroke(linewidth=3, foreground=(1, 1, 1, 0.9))]
        )


def _project_points_onto_pr_curve(
    recall_values: list[float],
    recall_axis: npt.NDArray[np.float32],
    precision_curve: npt.NDArray[np.float32],
) -> list[float]:
    """Project recall coordinates onto the plotted all-classes PR curve."""
    if recall_axis.size == 0 or precision_curve.size == 0:
        return [0.0 for _ in recall_values]
    return [
        float(np.interp(np.clip(recall_value, 0.0, 1.0), recall_axis, precision_curve))
        for recall_value in recall_values
    ]


def _plot_pr_curve_on_axis(
    ax: plt.Axes,
    px: npt.NDArray,
    py: npt.NDArray,
    ap: npt.NDArray,
    names: dict[int, str],
    show_legend: bool = True,
    legend_loc: str = "upper right",
) -> None:
    """Plot a precision-recall curve into an existing axis."""
    if not isinstance(py, np.ndarray) or py.size == 0 or ap.size == 0:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Precision-Recall Curve")
        ax.text(0.5, 0.5, "No class-wise PR data", ha="center", va="center", transform=ax.transAxes)
        return
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}", color=palette_mpl(i))
    else:
        ax.plot(px, py, linewidth=1, color=named_plot_color("background"), alpha=0.35)

    ax.plot(
        px,
        py.mean(1),
        linewidth=3,
        color=named_plot_color("mean"),
        label=f"all classes {ap[:, 0].mean():.3f} mAP@0.1",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="both", alpha=0.2, linewidth=0.6)
    if show_legend:
        ax.legend(loc=legend_loc, fontsize=8, framealpha=0.9)
    ax.set_title("Precision-Recall Curve")


def _plot_mc_curve_on_axis(
    ax: plt.Axes,
    px: npt.NDArray,
    py: npt.NDArray,
    names: dict[int, str],
    xlabel: str = "Confidence",
    ylabel: str = "Metric",
    show_legend: bool = True,
) -> None:
    """Plot a metric-confidence curve into an existing axis."""
    if not isinstance(py, np.ndarray) or py.size == 0:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{ylabel}-Confidence Curve")
        ax.text(
            0.5,
            0.5,
            f"No {ylabel.lower()} curve data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return
    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}", color=palette_mpl(i))
    else:
        ax.plot(px, py.T, linewidth=1, color=named_plot_color("background"), alpha=0.35)

    y = _smoothed_mean_confidence_curve(py, px)
    ax.plot(
        px,
        y,
        linewidth=3,
        color=named_plot_color("mean"),
        label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="both", alpha=0.2, linewidth=0.6)
    if show_legend:
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title(f"{ylabel}-Confidence Curve")


def _compute_froc_curve(
    tp: npt.NDArray[np.floating],
    fp: npt.NDArray[np.floating],
    conf: npt.NDArray[np.floating],
    target_cls: npt.NDArray[np.integer | np.floating],
    num_images: int,
    eps: float = 1e-16,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """Compute aggregate operating curves for FROC and confidence-based annotations."""
    if tp.size == 0 or fp.size == 0 or conf.size == 0:
        zeros = np.zeros((1,), dtype=np.float32)
        ones = np.ones((1,), dtype=np.float32)
        return zeros, zeros, ones

    indices_sorted: npt.NDArray[np.intp] = np.argsort(-conf)
    tp_sorted = tp[indices_sorted, 0].astype(np.float32)
    fp_sorted = fp[indices_sorted, 0].astype(np.float32)
    conf_sorted = conf[indices_sorted].astype(np.float32)

    tp_accumulated = tp_sorted.cumsum()
    fp_accumulated = fp_sorted.cumsum()
    num_targets = max(int(np.count_nonzero(target_cls >= 0)), 1)
    num_images = max(int(num_images), 1)

    recall = tp_accumulated / (num_targets + eps)
    avg_fp = fp_accumulated / num_images

    return (
        np.concatenate((np.array([0.0], dtype=np.float32), avg_fp)),
        np.concatenate((np.array([0.0], dtype=np.float32), recall)),
        np.concatenate((np.array([1.0], dtype=np.float32), conf_sorted)),
    )


def _select_froc_target_indices(
    avg_fp: npt.NDArray[np.float32],
    targets: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0),
) -> dict[float, int]:
    """Select operating points that satisfy each average-false-positives target."""
    indices: dict[float, int] = {}
    for target in targets:
        index = int(np.searchsorted(avg_fp, target, side="right") - 1)
        indices[target] = max(index, 0)
    return indices


def _compute_froc_score(
    avg_fp: npt.NDArray[np.floating],
    sensitivity: npt.NDArray[np.floating],
    targets: tuple[float, ...] = FROC_TARGETS,
) -> float:
    """Compute a single FROC score by averaging sensitivity at standard FP/image targets."""
    if avg_fp.size == 0 or sensitivity.size == 0:
        return 0.0

    # FROC x-axis is cumulative average false positives/image, so enforce monotonicity for robust
    # interpolation in degenerate cases with duplicate confidence scores.
    avg_fp_monotonic = np.maximum.accumulate(avg_fp.astype(np.float64, copy=False))
    sensitivity_values = sensitivity.astype(np.float64, copy=False)
    target_values = np.asarray(targets, dtype=np.float64)
    sampled_sensitivity = np.interp(
        target_values,
        avg_fp_monotonic,
        sensitivity_values,
        left=float(sensitivity_values[0]),
        right=float(sensitivity_values[-1]),
    )
    return float(sampled_sensitivity.mean())


def compute_class_agnostic_map(
    tp: npt.NDArray[np.floating],
    fp: npt.NDArray[np.floating],
    conf: npt.NDArray[np.floating],
    target_cls: npt.NDArray[np.integer | np.floating],
) -> tuple[float, float]:
    """Compute class-agnostic mAP@0.1 and mAP@0.1:0.95."""
    if tp.size == 0 or fp.size == 0 or target_cls.size == 0:
        return 0.0, 0.0

    agnostic_target_cls = np.zeros(target_cls.shape, dtype=np.int64)
    agnostic_pred_cls = np.zeros(conf.shape[0], dtype=np.int64)
    agnostic_results = ap_per_class(
        tp,
        fp,
        conf,
        agnostic_pred_cls,
        agnostic_target_cls,
    )
    agnostic_ap = agnostic_results[5]
    if not agnostic_ap.size:
        return 0.0, 0.0
    return float(agnostic_ap[:, 0].mean()), float(agnostic_ap.mean())


def compute_ap(
    recall: list[float], precision: list[float]
) -> tuple[float, npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Compute the average precision (AP) given the recall and rpecision curves.

    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        (float): Average precision.
        (np.ndarray): Precision envelope curve.
        (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """
    # Append sentinel values to beginning and end
    mrec: npt.NDArray[np.float32] = np.concatenate(([0.0], recall, [1.0]))
    mpre: npt.NDArray[np.float32] = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101, dtype=np.float32)  # 101-point interp (COCO)
        ap = np.trapezoid(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def ap_per_class(
    tp: npt.NDArray[np.floating],
    fp: npt.NDArray[np.floating],
    conf: npt.NDArray[np.floating],
    pred_cls: npt.NDArray[np.integer],
    target_cls: npt.NDArray[np.integer],
    eps: float = 1e-16,
):
    """Computes the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True)
            or not (False).
        fp (np.ndarray): Binary array indicating whether the detection is a false positive (True)
            or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the references.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.

    Returns:
        tp (np.ndarray): True positive counts at threshold given by max F1 metric for each class.
            Shape: (nc,).
        fp (np.ndarray): False positive counts at threshold given by max F1 metric for each class.
            Shape: (nc,).
        p (np.ndarray): Precision values at threshold given by max F1 metric for each class.
            Shape: (nc,).
        r (np.ndarray): Recall values at threshold given by max F1 metric for each class.
            Shape: (nc,).
        f1 (np.ndarray): F1-score values at threshold given by max F1 metric for each class.
            Shape: (nc,).
        ap (np.ndarray): Average precision for each class at different IoU thresholds.
            Shape: (nc, 18).
        unique_classes (np.ndarray): An array of unique classes that have data. Shape: (nc,).
        p_curve (np.ndarray): Precision curves for each class. Shape: (nc, 1000).
        r_curve (np.ndarray): Recall curves for each class. Shape: (nc, 1000).
        f1_curve (np.ndarray): F1-score curves for each class. Shape: (nc, 1000).
        x (np.ndarray): X-axis values for the curves. Shape: (1000,).
        prec_values (np.ndarray): Precision values at mAP@0.1 for each class. Shape: (nc, 1000).
    """
    # Sort by objectness
    indices_sorted: npt.NDArray[np.intp] = np.argsort(-conf)
    tp, fp, conf, pred_cls = (
        tp[indices_sorted],
        fp[indices_sorted],
        conf[indices_sorted],
        pred_cls[indices_sorted],
    )

    unique_classes, counts = np.unique(target_cls, return_counts=True)
    # If objects without class (class = -1) are present, don't compute mAP for class -1
    if unique_classes.size and unique_classes[0] == -1:
        unique_classes = unique_classes[1:]
        counts = counts[1:]
    nc: int = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    x = np.linspace(0, 1, 1000)

    if nc == 0:
        empty_curve = np.zeros((0, 1000), dtype=np.float32)
        empty_ap = np.zeros((0, tp.shape[1]), dtype=np.float32)
        empty_scalar = np.zeros((0,), dtype=np.float32)
        return (
            empty_scalar,
            empty_scalar,
            empty_scalar,
            empty_scalar,
            empty_scalar,
            empty_ap,
            unique_classes.astype(int),
            empty_curve,
            empty_curve,
            empty_curve,
            x,
            empty_curve,
        )

    # Average precision, precision and recall curves
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    prec_values = np.zeros((nc, 1000), dtype=np.float32)
    for class_index, c in enumerate(unique_classes):
        indices = pred_cls == c
        number_of_labels = counts[class_index]
        number_of_predictions = indices.sum()
        if number_of_predictions == 0 or number_of_labels == 0:
            continue

        fp_accumulated = fp[indices].cumsum(0)
        tp_accumulated = tp[indices].cumsum(0)

        recall = tp_accumulated / (number_of_labels + eps)
        r_curve[class_index] = np.interp(
            -x, -conf[indices], recall[:, 0], left=0
        )  # negative x, conf because conf decreases

        precision = tp_accumulated / (tp_accumulated + fp_accumulated + eps)
        p_curve[class_index] = np.interp(
            -x, -conf[indices], precision[:, 0], left=1
        )  # precision at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[class_index, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                # Keep one row per class so this array stays aligned with unique_classes.
                prec_values[class_index] = np.interp(x, mrec, mpre).astype(np.float32, copy=False)

    # Compute F1 (harmonic mean of precision and recall)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)

    max_i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = (
        p_curve[:, max_i],
        r_curve[:, max_i],
        f1_curve[:, max_i],
    )  # max-F1 precision, recall, F1 values
    tp = (r * counts).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return (
        tp,
        fp,
        p,
        r,
        f1,
        ap,
        unique_classes.astype(int),
        p_curve,
        r_curve,
        f1_curve,
        x,
        prec_values,
    )


class Metric:
    """Class for computing evaluation metrics for YOLOv8 model.

    Attributes:
        p (list): Precision for each class. Shape: (nc,).
        r (list): Recall for each class. Shape: (nc,).
        f1 (list): F1 score for each class. Shape: (nc,).
        all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 18).
        ap_class_index (list): Index of class for each AP score. Shape: (nc,).
        nc (int): Number of classes.
    """

    def __init__(self) -> None:
        """Initializes a Metric instance for computing evaluation metrics."""
        self.p: npt.NDArray | list = []  # (nc, )
        self.r: npt.NDArray | list = []  # (nc, )
        self.f1: npt.NDArray | list = []  # (nc, )
        self.all_ap: npt.NDArray | list = []  # (nc, 18)
        self.ap_class_index: npt.NDArray | list[int] = []  # (nc, )
        self.nc: int = 0
        self.p_curve: npt.NDArray[np.float32] = np.zeros((0, 1000), dtype=np.float32)
        self.r_curve: npt.NDArray[np.float32] = np.zeros((0, 1000), dtype=np.float32)
        self.f1_curve: npt.NDArray[np.float32] = np.zeros((0, 1000), dtype=np.float32)
        self.px: npt.NDArray[np.float32] = np.linspace(0, 1, 1000, dtype=np.float32)
        self.prec_values: npt.NDArray[np.float32] = np.zeros((0, 1000), dtype=np.float32)
        self.map10_agnostic_value: float = 0.0
        self.map_agnostic_value: float = 0.0
        self.best_index: int = 0
        self.best_confidence: float = 1.0
        self.froc_avg_fp: npt.NDArray[np.float32] = np.zeros((1,), dtype=np.float32)
        self.froc_sensitivity: npt.NDArray[np.float32] = np.zeros((1,), dtype=np.float32)
        self.froc_confidence: npt.NDArray[np.float32] = np.ones((1,), dtype=np.float32)

    @property
    def ap10(self) -> npt.NDArray | list:
        """Returns the Average Precision (AP) at an IoU threshold of 0.1 for all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP10 values per class, or an empty list
                if not available.
        """
        return self.all_ap[:, 0] if isinstance(self.all_ap, np.ndarray) else []

    @property
    def ap(self) -> npt.NDArray | list:
        """Returns the Average Precision (AP) at an IoU threshold of 0.1-0.95 for all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP10-95 values per class, or an empty
                list if not available.
        """
        return self.all_ap.mean(1) if isinstance(self.all_ap, np.ndarray) else []

    @property
    def mp(self) -> float:
        """Returns the Mean Precision of all classes.

        Returns:
            (float): The mean precision of all classes.
        """
        return self.p.mean() if isinstance(self.p, np.ndarray) and self.p.size else 0.0

    @property
    def mr(self) -> float:
        """Returns the Mean Recall of all classes.

        Returns:
            (float): The mean recall of all classes.
        """
        return self.r.mean() if isinstance(self.r, np.ndarray) and self.r.size else 0.0

    @property
    def map10(self) -> float:
        """Returns the mean Average Precision (mAP) at an IoU threshold of 0.1.

        Returns:
            (float): The mAP at an IoU threshold of 0.1.
        """
        return (
            self.all_ap[:, 0].mean()
            if isinstance(self.all_ap, np.ndarray) and self.all_ap.size
            else 0.0
        )

    @property
    def map(self) -> float:
        """Returns mean Average Precision (mAP) over IoU thresholds of 0.1 - 0.95 in steps of 0.05.

        Returns:
            (float): The mAP over IoU thresholds of 0.1 - 0.95 in steps of 0.05.
        """
        return (
            self.all_ap.mean() if isinstance(self.all_ap, np.ndarray) and self.all_ap.size else 0.0
        )

    @property
    def map_agnostic(self) -> float:
        """Returns class-agnostic mAP over IoU thresholds of 0.1 - 0.95."""
        return self.map_agnostic_value

    @property
    def map10_agnostic(self) -> float:
        """Returns class-agnostic mAP at an IoU threshold of 0.1."""
        return self.map10_agnostic_value

    def mean_results(self) -> list[float]:
        """Mean of results, return mp, mr, map10, map, and agnostic map."""
        return [self.mp, self.mr, self.map10, self.map10_agnostic, self.map, self.map_agnostic]

    def class_result(
        self, i: int
    ) -> tuple[npt.NDArray | Any, npt.NDArray | Any, npt.NDArray | Any, npt.NDArray | Any]:
        """Class-aware result, return p[i], r[i], ap10[i], ap[i]."""
        return self.p[i], self.r[i], self.ap10[i], self.ap[i]

    @property
    def maps(self) -> npt.NDArray:
        """MAP of each class."""
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def update(
        self,
        *,
        p: npt.NDArray,
        r: npt.NDArray,
        f1: npt.NDArray,
        all_ap: npt.NDArray,
        ap_class_index: npt.NDArray,
        p_curve: npt.NDArray,
        r_curve: npt.NDArray,
        f1_curve: npt.NDArray,
        px: npt.NDArray,
        prec_values: npt.NDArray,
        map10_agnostic_value: float,
        map_agnostic_value: float,
        best_index: int,
        best_confidence: float,
        froc_avg_fp: npt.NDArray[np.float32],
        froc_sensitivity: npt.NDArray[np.float32],
        froc_confidence: npt.NDArray[np.float32],
    ) -> None:
        """Updates the evaluation metrics of the model with a new set of results.

        Args:
            p: Precision per class at the selected operating point.
            r: Recall per class at the selected operating point.
            f1: F1 score per class at the selected operating point.
            all_ap: AP values per class across IoU thresholds.
            ap_class_index: Class ids corresponding to rows in the metric arrays.
            p_curve: Precision-confidence curves per class.
            r_curve: Recall-confidence curves per class.
            f1_curve: F1-confidence curves per class.
            px: Shared x-axis for confidence-based curves.
            prec_values: Precision-recall curve values at mAP@0.1.
            map10_agnostic_value: Class-agnostic mAP@0.1.
            map_agnostic_value: Class-agnostic mAP@0.1:0.95.
            best_index: Index of the max-F1 operating point on `px`.
            best_confidence: Confidence threshold at the max-F1 operating point.
            froc_avg_fp: FROC x-axis values in average false positives per image.
            froc_sensitivity: FROC sensitivity values.
            froc_confidence: Confidence thresholds corresponding to FROC points.

        Side Effects:
            Updates the stored metric arrays and operating-point metadata on this instance.
        """
        self.p = p
        self.r = r
        self.f1 = f1
        self.all_ap = all_ap
        self.ap_class_index = ap_class_index
        self.p_curve = p_curve
        self.r_curve = r_curve
        self.f1_curve = f1_curve
        self.px = px
        self.prec_values = prec_values
        self.map10_agnostic_value = map10_agnostic_value
        self.map_agnostic_value = map_agnostic_value
        self.best_index = best_index
        self.best_confidence = best_confidence
        self.froc_avg_fp = froc_avg_fp
        self.froc_sensitivity = froc_sensitivity
        self.froc_confidence = froc_confidence


class DetMetrics:
    """Utility class for computing detection metrics of an object detection model.

    Provides metrics `such as precision, recall, and mean average precision (mAP).

    Args:
        save_dir (Path): Path to the directory where the output plots will be saved.
            Defaults to current directory.
        plot (bool): Flag that indicates whether to plot precision-recall curves for each class.
            Defaults to False.
        names (dict of str): Dict of strings that represents the names of the classes.
            Defaults to an empty tuple.

    Attributes:
        save_dir (Path): A path to the directory where the output plots will be saved.
        plot (bool): Flag that indicates whether to plot the precision-recall curves for each class.
        names (dict of str): A dict of strings that represents the names of the classes.
        metrics (Metric): Instance of Metric class for storing the results of the detection metrics.
        speed (dict): Dictionary for storing execution time of different parts of detection process.
    """

    def __init__(
        self,
        save_dir: Path = Path(),
        names: dict[int, str] = {},
    ) -> None:
        """Initialize with save directory, plot flag, callback function, and class names."""
        self.save_dir: Path = save_dir
        self.names: dict[int, str] = names
        self.metrics = Metric()
        self.speed: dict[str, float] = {
            "preprocess": 0.0,
            "inference": 0.0,
            "loss": 0.0,
            "postprocess": 0.0,
        }

    def process(
        self,
        tp: npt.NDArray[np.floating],
        fp: npt.NDArray[np.floating],
        tp_agnostic: npt.NDArray[np.floating],
        fp_agnostic: npt.NDArray[np.floating],
        conf: npt.NDArray[np.floating],
        pred_cls: npt.NDArray[np.integer],
        target_cls: npt.NDArray[np.integer],
        num_images: int | None = None,
    ) -> None:
        """Process predicted results for object detection and update metrics."""
        class_sensitive_results = ap_per_class(
            tp,
            fp,
            conf,
            pred_cls,
            target_cls,
        )
        map10_agnostic, map_agnostic = compute_class_agnostic_map(
            tp_agnostic,
            fp_agnostic,
            conf,
            target_cls,
        )
        _, _, f1_curve, px, _ = class_sensitive_results[7:12]
        if isinstance(f1_curve, np.ndarray) and f1_curve.size:
            best_index = int(_smoothed_mean_confidence_curve(f1_curve, px).argmax())
            best_confidence = float(px[best_index])
        else:
            best_index = 0
            best_confidence = 1.0
        froc_avg_fp, froc_sensitivity, froc_confidence = _compute_froc_curve(
            tp=tp,
            fp=fp,
            conf=conf,
            target_cls=target_cls,
            num_images=0 if num_images is None else num_images,
        )
        self.metrics.nc = len(self.names)
        self.metrics.update(
            p=class_sensitive_results[2],
            r=class_sensitive_results[3],
            f1=class_sensitive_results[4],
            all_ap=class_sensitive_results[5],
            ap_class_index=class_sensitive_results[6],
            p_curve=class_sensitive_results[7],
            r_curve=class_sensitive_results[8],
            f1_curve=class_sensitive_results[9],
            px=class_sensitive_results[10],
            prec_values=class_sensitive_results[11],
            map10_agnostic_value=map10_agnostic,
            map_agnostic_value=map_agnostic,
            best_index=best_index,
            best_confidence=best_confidence,
            froc_avg_fp=froc_avg_fp,
            froc_sensitivity=froc_sensitivity,
            froc_confidence=froc_confidence,
        )

    def plot_summary(
        self,
        confusion_matrix: ConfusionMatrix,
    ) -> None:
        """Save one combined validation figure with confusion matrices and metric curves."""
        plot_names = {
            metric_index: self.names.get(int(class_id), str(class_id))
            for metric_index, class_id in enumerate(self.metrics.ap_class_index)
        }
        fig = plt.figure(figsize=(14, 16), constrained_layout=True)
        outer_grid = fig.add_gridspec(2, 2, height_ratios=(2, 1), hspace=0.05, wspace=0.02)
        ax_cm_norm = fig.add_subplot(outer_grid[1, 0])
        ax_cm_raw = fig.add_subplot(outer_grid[1, 1])

        curves_grid = outer_grid[0, :].subgridspec(1, 2, width_ratios=(1.15, 1.0), wspace=0.18)
        left_curves = curves_grid[0, 0].subgridspec(2, 1, hspace=0.04)
        right_curves = curves_grid[0, 1].subgridspec(3, 1, hspace=0.04)
        ax_pr = fig.add_subplot(left_curves[0, 0])
        ax_froc = fig.add_subplot(left_curves[1, 0])
        ax_f1 = fig.add_subplot(right_curves[0, 0])
        ax_p = fig.add_subplot(right_curves[1, 0])
        ax_r = fig.add_subplot(right_curves[2, 0])

        confusion_matrix.render(
            ax=ax_cm_norm,
            normalize=True,
            names=tuple(self.names.values()),
        )
        confusion_matrix.render(
            ax=ax_cm_raw,
            normalize=False,
            names=tuple(self.names.values()),
        )

        _plot_pr_curve_on_axis(
            ax=ax_pr,
            px=self.metrics.px,
            py=self.metrics.prec_values,
            ap=self.metrics.all_ap,
            names=plot_names,
            legend_loc="lower left",
        )
        _plot_mc_curve_on_axis(
            ax=ax_f1,
            px=self.metrics.px,
            py=self.metrics.f1_curve,
            names=plot_names,
            ylabel="F1",
            show_legend=False,
        )
        _plot_mc_curve_on_axis(
            ax=ax_p,
            px=self.metrics.px,
            py=self.metrics.p_curve,
            names=plot_names,
            ylabel="Precision",
            show_legend=False,
        )
        _plot_mc_curve_on_axis(
            ax=ax_r,
            px=self.metrics.px,
            py=self.metrics.r_curve,
            names=plot_names,
            ylabel="Recall",
            show_legend=False,
        )

        froc_ticks = np.array([1 / 16, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=np.float32)
        positive_froc_mask = self.metrics.froc_avg_fp > 0
        if positive_froc_mask.any():
            ax_froc.plot(
                self.metrics.froc_avg_fp[positive_froc_mask],
                self.metrics.froc_sensitivity[positive_froc_mask],
                linewidth=3,
                color=named_plot_color("mean"),
            )
        ax_froc.set_xlabel("Average False Positives per Image")
        ax_froc.set_ylabel("Sensitivity")
        ax_froc.set_xscale("log", base=2)
        ax_froc.set_xlim(float(froc_ticks[0]), float(froc_ticks[-1]))
        ax_froc.set_xticks(froc_ticks)
        ax_froc.set_xticklabels(["0.0625", "0.125", "0.25", "0.5", "1", "2", "4", "8"])
        ax_froc.set_ylim(0, 1)
        ax_froc.grid(True, which="both", axis="both", alpha=0.2, linewidth=0.6)
        ax_froc.set_title("FROC Curve")

        target_indices = _select_froc_target_indices(self.metrics.froc_avg_fp)
        target_confidences = [
            float(self.metrics.froc_confidence[index]) for index in target_indices.values()
        ]
        target_labels = [f"{confidence:.3f}" for confidence in target_confidences]

        mean_precision_curve = _smoothed_mean_confidence_curve(
            self.metrics.p_curve,
            self.metrics.px,
        )
        mean_recall_curve = _smoothed_mean_confidence_curve(
            self.metrics.r_curve,
            self.metrics.px,
        )
        mean_f1_curve = _smoothed_mean_confidence_curve(
            self.metrics.f1_curve,
            self.metrics.px,
        )
        mean_pr_curve = (
            self.metrics.prec_values.mean(0)
            if self.metrics.prec_values.size
            else np.zeros_like(self.metrics.px)
        )
        target_curve_indices = [
            int(np.abs(self.metrics.px - confidence).argmin()) for confidence in target_confidences
        ]
        target_pr_recalls = [float(mean_recall_curve[index]) for index in target_curve_indices]
        target_pr_precisions = _project_points_onto_pr_curve(
            recall_values=target_pr_recalls,
            recall_axis=self.metrics.px,
            precision_curve=mean_pr_curve,
        )

        _annotate_points(
            ax=ax_pr,
            x_values=target_pr_recalls,
            y_values=target_pr_precisions,
            labels=target_labels,
            color=named_plot_color("operating"),
        )
        _annotate_points(
            ax=ax_f1,
            x_values=target_confidences,
            y_values=[float(mean_f1_curve[index]) for index in target_curve_indices],
            labels=target_labels,
            color=named_plot_color("operating"),
        )
        _annotate_points(
            ax=ax_p,
            x_values=target_confidences,
            y_values=[float(mean_precision_curve[index]) for index in target_curve_indices],
            labels=target_labels,
            color=named_plot_color("operating"),
        )
        _annotate_points(
            ax=ax_r,
            x_values=target_confidences,
            y_values=[float(mean_recall_curve[index]) for index in target_curve_indices],
            labels=target_labels,
            color=named_plot_color("operating"),
        )
        _annotate_points(
            ax=ax_froc,
            x_values=[
                max(float(self.metrics.froc_avg_fp[index]), float(froc_ticks[0]))
                for index in target_indices.values()
            ],
            y_values=[
                float(self.metrics.froc_sensitivity[index]) for index in target_indices.values()
            ],
            labels=target_labels,
            color=named_plot_color("operating"),
        )

        best_confidence = float(self.metrics.best_confidence)
        best_curve_index = (
            int(np.abs(self.metrics.px - best_confidence).argmin()) if self.metrics.px.size else 0
        )
        best_froc_index = int(np.abs(self.metrics.froc_confidence - best_confidence).argmin())
        best_label = f"F1 max {best_confidence:.3f}"
        best_pr_recall = float(mean_recall_curve[best_curve_index])
        best_pr_precision = _project_points_onto_pr_curve(
            recall_values=[best_pr_recall],
            recall_axis=self.metrics.px,
            precision_curve=mean_pr_curve,
        )[0]
        _annotate_points(
            ax=ax_pr,
            x_values=[best_pr_recall],
            y_values=[best_pr_precision],
            labels=[best_label],
            color=named_plot_color("best"),
            marker="*",
        )
        _annotate_points(
            ax=ax_f1,
            x_values=[best_confidence],
            y_values=[float(mean_f1_curve[best_curve_index])],
            labels=[best_label],
            color=named_plot_color("best"),
            marker="*",
        )
        _annotate_points(
            ax=ax_p,
            x_values=[best_confidence],
            y_values=[float(mean_precision_curve[best_curve_index])],
            labels=[best_label],
            color=named_plot_color("best"),
            marker="*",
        )
        _annotate_points(
            ax=ax_r,
            x_values=[best_confidence],
            y_values=[float(mean_recall_curve[best_curve_index])],
            labels=[best_label],
            color=named_plot_color("best"),
            marker="*",
        )
        _annotate_points(
            ax=ax_froc,
            x_values=[max(float(self.metrics.froc_avg_fp[best_froc_index]), float(froc_ticks[0]))],
            y_values=[float(self.metrics.froc_sensitivity[best_froc_index])],
            labels=[best_label],
            color=named_plot_color("best"),
            marker="*",
        )

        plot_fname = self.save_dir / "validation_summary.png"
        try:
            save_figure_with_warning(fig, plot_fname, dpi=250)
        finally:
            plt.close(fig)

    @property
    def keys(self) -> list[str]:
        """Returns a list of keys for accessing specific metrics."""
        return [
            "metrics/precision",
            "metrics/recall",
            "metrics/mAP10",
            "metrics/mAP10Agn",
            "metrics/mAP10-95",
            "metrics/mAP10-95Agn",
        ]

    def mean_results(self) -> list[float]:
        """Calculate mean detection metrics including class-agnostic mAP10-95."""
        return self.metrics.mean_results()

    @property
    def summary_keys(self) -> list[str]:
        """Returns metric keys used for terminal summaries only."""
        return [
            "metrics/precision",
            "metrics/recall",
            "metrics/mAP10",
            "metrics/mAP10-95",
        ]

    def summary_mean_results(self) -> list[float]:
        """Calculate mean detection metrics shown in the terminal summary."""
        return [self.metrics.mp, self.metrics.mr, self.metrics.map10, self.metrics.map]

    def class_result(self, i: int) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """Return performance of an object detection model on a specific class."""
        return self.metrics.class_result(i)

    @property
    def maps(self) -> npt.NDArray:
        """Returns mean Average Precision (mAP) scores per class."""
        return self.metrics.maps

    @property
    def ap_class_index(self) -> npt.NDArray | list[int]:
        """Returns the average precision index per class."""
        return self.metrics.ap_class_index

    @property
    def results_dict(self) -> dict[str, float]:
        """Returns dictionary of computed performance metrics and statistics."""
        return dict(zip(self.keys, self.mean_results(), strict=True))

    @property
    def curves(self) -> list[str]:
        """Returns a list of curves for accessing specific metrics curves."""
        return [
            "Precision-Recall",
            "F1-Confidence",
            "Precision-Confidence",
            "Recall-Confidence",
        ]
