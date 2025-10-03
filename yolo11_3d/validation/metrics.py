# Ultralytics YOLO 🚀, AGPL-3.0 license

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
import torch

LOGGER = logging.getLogger(__name__)


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Calculate intersection-over-union (IoU) of boxes.

    Both sets of boxes are expected to be in (x1, y1, z1, x2, y2, z2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py.

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 6) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 6) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1
        and box2.
    """
    # NOTE: Need .float() to get accurate iou values
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = (
        box1.float().unsqueeze(1).chunk(2, 2),
        box2.float().unsqueeze(0).chunk(2, 2),
    )
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xyzwhd: bool = True,
    GIoU: bool = False,  # noqa: N803
    DIoU: bool = False,  # noqa: N803
    CIoU: bool = False,  # noqa: N803
    eps: float = 1e-7,
) -> torch.Tensor:
    """Calculate Intersection over Union (IoU) of box1(1, 6) to box2(n, 6).

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 6).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 6).
        xyzwhd (bool, optional): If True, input boxes are in (x, y, z, w, h, h) format. If False,
                               input boxes are in (x1, y1, z1, x2, y2, z2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    # Get the coordinates of bounding boxes
    if xyzwhd:  # transform from xyzwhd to xyzxyz
        (x1, y1, z1, w1, h1, d1), (x2, y2, z2, w2, h2, d2) = box1.chunk(6, -1), box2.chunk(6, -1)
        w1_, h1_, d1_, w2_, h2_, d2_ = w1 / 2, h1 / 2, d1 / 2, w2 / 2, h2 / 2, d2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2, b1_z1, b1_z2 = (
            x1 - w1_,
            x1 + w1_,
            y1 - h1_,
            y1 + h1_,
            z1 - d1_,
            z1 + d1_,
        )
        b2_x1, b2_x2, b2_y1, b2_y2, b2_z1, b2_z2 = (
            x2 - w2_,
            x2 + w2_,
            y2 - h2_,
            y2 + h2_,
            z2 - d2_,
            z2 + d2_,
        )
    else:  # x1, y1, z1, x2, y2, z2 = box1
        b1_x1, b1_y1, b1_z1, b1_x2, b1_y2, b1_z2 = box1.chunk(6, -1)
        b2_x1, b2_y1, b2_z1, b2_x2, b2_y2, b2_z2 = box2.chunk(6, -1)
        w1, h1, d1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps, b1_z2 - b1_z1 + eps
        w2, h2, d2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps, b2_z2 - b2_z1 + eps

    # Intersection area
    inter = (
        (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0)
        * (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
        * (b1_z2.minimum(b2_z2) - b1_z1.maximum(b2_z1)).clamp_(0)
    )

    # Union Area
    union = w1 * h1 * d1 + w2 * h2 * d2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        convex_width = b1_x2.maximum(b2_x2) - b1_x1.minimum(
            b2_x1
        )  # convex (smallest enclosing box) width
        convex_height = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        convex_depth = b1_z2.maximum(b2_z2) - b1_z1.minimum(b2_z1)
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # c2: diagonal length of convex box omitting sqrt operation for performance
            c2 = (
                convex_width.pow(2) + convex_height.pow(2) + convex_depth.pow(2) + eps
            )  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                # v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                v = (
                    (2 / math.pi)
                    * (
                        ((w2 / h2).atan() - (w1 / h1).atan())
                        + ((w2 / d2).atan() - (w1 / d2).atan())
                        + ((h2 / d2).atan() - (h1 / d1).atan())
                    )
                    / 3
                ).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = convex_width * convex_height + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


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

    def matrix(self) -> npt.NDArray[np.float32]:
        """Returns the confusion matrix."""
        return self.matrix_

    def plot(
        self,
        normalize: bool = True,
        save_dir: Path | str = "",
        names: tuple[str, ...] = (),
    ) -> None:
        """Plot the confusion matrix using seaborn and save it to a file.

        Args:
            normalize (bool): Whether to normalize the confusion matrix.
            save_dir (str): Directory where the plot will be saved.
            names (tuple): Names of classes, used as labels on the plot.
        """
        array = self.matrix_ / (
            (self.matrix_.sum(0).reshape(1, -1) + 1e-9) if normalize else 1
        )  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        # Number of *display* classes excludes background but includes unclassified
        display_nc = self.nc + 1  # include unclassified in display (exclude background)
        sns.set_theme(font_scale=1.0 if display_nc < 50 else 0.8)
        # Prepare tick labels
        names_list = list(names)
        # Append unclassified name if not supplied already (len check ensures backward compat)
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
                cmap="Blues",
                fmt=".2f" if normalize else ".0f",
                square=True,
                vmin=0.0,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            ).set_facecolor((1, 1, 1))
        title = "Confusion Matrix" + " Normalized" * normalize
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(title)
        plot_fname = Path(save_dir) / f"{title.lower().replace(' ', '_')}.png"
        fig.savefig(plot_fname, dpi=250)
        plt.close(fig)

    def print(self) -> None:
        """Print the confusion matrix to the console (including unclassified and background)."""
        for i in range(self.matrix_.shape[0]):
            LOGGER.info(" ".join(map(str, self.matrix_[i])))


def smooth(y: np.ndarray, fraction: float = 0.05) -> np.ndarray:
    """Box filter of fraction f."""
    num_filtered_elements = round(len(y) * fraction * 2) // 2 + 1  # must be odd
    padding_ones = np.ones(num_filtered_elements // 2)
    y_padded = np.concatenate((padding_ones * y[0], y, padding_ones * y[-1]), 0)
    smoothing_filter = np.ones(num_filtered_elements) / num_filtered_elements
    return np.convolve(y_padded, smoothing_filter, mode="valid")


def plot_pr_curve(
    px: npt.NDArray,
    py: npt.NDArray,
    ap: npt.NDArray,
    save_dir: Path = Path("pr_curve.png"),
    names: dict[int, str] = {},
) -> None:
    """Plots a precision-recall curve."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(
                px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}"
            )  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(
        px,
        py.mean(1),
        linewidth=3,
        color="blue",
        label=f"all classes {ap[:, 0].mean():.3f} mAP@0.1",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


def plot_mc_curve(
    px: npt.NDArray,
    py: npt.NDArray,
    save_dir: Path = Path("mc_curve.png"),
    names: dict[int, str] = {},
    xlabel: str = "Confidence",
    ylabel: str = "Metric",
) -> None:
    """Plots a metric-confidence curve."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(
        px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


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
    plot: bool = False,
    save_dir: Path = Path(),
    names: dict = {},
    eps: float = 1e-16,
    prefix: str = "",
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
        plot (bool, optional): Whether to plot PR curves or not. Defaults to False.
        save_dir (Path, optional): Directory to save the PR curves. Defaults to an empty path.
        names (dict, optional): Dict of class names to plot PR curves. Defaults to an empty tuple.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.
        prefix (str, optional): Prefix string for saving plot files. Defaults to an empty string.

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
    if unique_classes[0] == -1:
        unique_classes = unique_classes[1:]
        counts = counts[1:]
    nc: int = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    x, prec_values = np.linspace(0, 1, 1000), []

    # Average precision, precision and recall curves
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
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
                prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.1

    prec_values = np.array(prec_values)  # (nc, 1000)

    # Compute F1 (harmonic mean of precision and recall)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    names_ = [
        v for k, v in names.items() if k in unique_classes
    ]  # list: only classes that have data
    names = dict(enumerate(names_))
    if plot:
        plot_pr_curve(
            x,
            prec_values,
            ap,
            save_dir / f"{prefix}PR_curve.png",
            names,
        )
        plot_mc_curve(
            x,
            f1_curve,
            save_dir / f"{prefix}F1_curve.png",
            names,
            ylabel="F1",
        )
        plot_mc_curve(
            x,
            p_curve,
            save_dir / f"{prefix}P_curve.png",
            names,
            ylabel="Precision",
        )
        plot_mc_curve(
            x,
            r_curve,
            save_dir / f"{prefix}R_curve.png",
            names,
            ylabel="Recall",
        )

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
        """Initializes a Metric instance for computing evaluation metrics for the YOLOv8 model."""
        self.p: npt.NDArray | list = []  # (nc, )
        self.r: npt.NDArray | list = []  # (nc, )
        self.f1: npt.NDArray | list = []  # (nc, )
        self.all_ap: npt.NDArray | list = []  # (nc, 18)
        self.ap_class_index: npt.NDArray | list[int] = []  # (nc, )
        self.nc: int = 0

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
        return self.p.mean() if isinstance(self.p, np.ndarray) else 0.0

    @property
    def mr(self) -> float:
        """Returns the Mean Recall of all classes.

        Returns:
            (float): The mean recall of all classes.
        """
        return self.r.mean() if isinstance(self.r, np.ndarray) else 0.0

    @property
    def map10(self) -> float:
        """Returns the mean Average Precision (mAP) at an IoU threshold of 0.1.

        Returns:
            (float): The mAP at an IoU threshold of 0.1.
        """
        return self.all_ap[:, 0].mean() if isinstance(self.all_ap, np.ndarray) else 0.0

    @property
    def map(self) -> float:
        """Returns mean Average Precision (mAP) over IoU thresholds of 0.1 - 0.95 in steps of 0.05.

        Returns:
            (float): The mAP over IoU thresholds of 0.1 - 0.95 in steps of 0.05.
        """
        return self.all_ap.mean() if isinstance(self.all_ap, np.ndarray) else 0.0

    def mean_results(self) -> list[float]:
        """Mean of results, return mp, mr, map10, map."""
        return [self.mp, self.mr, self.map10, self.map]

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
        results: tuple[
            npt.NDArray,
            npt.NDArray,
            npt.NDArray,
            npt.NDArray,
            npt.NDArray,
            npt.NDArray,
            npt.NDArray,
            npt.NDArray,
            npt.NDArray,
            list,
        ],
    ) -> None:
        """Updates the evaluation metrics of the model with a new set of results.

        Args:
            results (tuple): A tuple containing the following evaluation metrics:
                - p (list): Precision for each class. Shape: (nc,).
                - r (list): Recall for each class. Shape: (nc,).
                - f1 (list): F1 score for each class. Shape: (nc,).
                - all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
                - ap_class_index (list): Index of class for each AP score. Shape: (nc,).

        Side Effects:
            Updates the class attributes `self.p`, `self.r`, `self.f1`, `self.all_ap`, and
            `self.ap_class_index` based on the values provided in the `results` tuple.
        """
        (
            self.p,
            self.r,
            self.f1,
            self.all_ap,
            self.ap_class_index,
            self.p_curve,
            self.r_curve,
            self.f1_curve,
            self.px,
            self.prec_values,
        ) = results

    @property
    def curves(self) -> list:
        """Returns a list of curves for accessing specific metrics curves."""
        return []

    @property
    def curves_results(self) -> list[list[Any | str]]:
        """Returns a list of curves for accessing specific metrics curves."""
        return [
            [self.px, self.prec_values, "Recall", "Precision"],
            [self.px, self.f1_curve, "Confidence", "F1"],
            [self.px, self.p_curve, "Confidence", "Precision"],
            [self.px, self.r_curve, "Confidence", "Recall"],
        ]


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
        box (Metric): Instance of the Metric class for storing the results of the detection metrics.
        speed (dict): Dictionary for storing execution time of different parts of detection process.
    """

    def __init__(
        self,
        save_dir: Path = Path(),
        plot: bool = False,
        names: dict[int, str] = {},
    ) -> None:
        """Initialize with save directory, plot flag, callback function, and class names."""
        self.save_dir: Path = save_dir
        self.plot: bool = plot
        self.names: dict[int, str] = names
        self.box = Metric()
        self.speed: dict[str, float] = {
            "preprocess": 0.0,
            "inference": 0.0,
            "loss": 0.0,
            "postprocess": 0.0,
        }
        self.confusion_matrix: ConfusionMatrix | None = None

    def process(
        self,
        tp: npt.NDArray[np.floating],
        fp: npt.NDArray[np.floating],
        conf: npt.NDArray[np.floating],
        pred_cls: npt.NDArray[np.integer],
        target_cls: npt.NDArray[np.integer],
    ) -> None:
        """Process predicted results for object detection and update metrics."""
        results = ap_per_class(
            tp,
            fp,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            save_dir=self.save_dir,
            names=self.names,
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results)

    @property
    def keys(self) -> list[str]:
        """Returns a list of keys for accessing specific metrics."""
        return [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP10(B)",
            "metrics/mAP10-95(B)",
        ]

    def mean_results(self) -> list[float]:
        """Calculate mean of detected objects & return precision, recall, mAP10, and mAP10-95."""
        return self.box.mean_results()

    def class_result(self, i: int) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """Return performance of an object detection model on a specific class."""
        return self.box.class_result(i)

    @property
    def maps(self) -> npt.NDArray:
        """Returns mean Average Precision (mAP) scores per class."""
        return self.box.maps

    @property
    def ap_class_index(self) -> npt.NDArray | list[int]:
        """Returns the average precision index per class."""
        return self.box.ap_class_index

    @property
    def results_dict(self) -> dict[str, float]:
        """Returns dictionary of computed performance metrics and statistics."""
        return dict(zip(self.keys, self.mean_results(), strict=True))

    @property
    def curves(self) -> list[str]:
        """Returns a list of curves for accessing specific metrics curves."""
        return [
            "Precision-Recall(B)",
            "F1-Confidence(B)",
            "Precision-Confidence(B)",
            "Recall-Confidence(B)",
        ]

    @property
    def curves_results(self) -> list[list[Any | str]]:
        """Returns dictionary of computed performance metrics and statistics."""
        return self.box.curves_results
