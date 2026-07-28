from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from radyolo.validation.metrics import (
    ConfusionMatrix,
    DetMetrics,
    _compute_froc_curve,
    _compute_froc_score,
    _plot_mc_curve_on_axis,
    _plot_pr_curve_on_axis,
    _project_points_onto_pr_curve,
    _select_froc_target_indices,
    ap_per_class,
    compute_class_agnostic_map,
)


def test_confusion_matrix_process_batch_handles_empty_gt_and_missing_detections() -> None:
    confusion_matrix = ConfusionMatrix(nc=2, conf=0.2, iou_thres=0.1)

    detections = torch.tensor(
        [
            [0, 0, 0, 2, 2, 2, 0.9, 0],
            [0, 0, 0, 2, 2, 2, 0.1, 1],
        ],
        dtype=torch.float32,
    )
    empty_boxes = torch.zeros((0, 6), dtype=torch.float32)
    empty_classes = torch.zeros((0,), dtype=torch.float32)
    confusion_matrix.process_batch(detections, empty_boxes, empty_classes)

    assert confusion_matrix.matrix_[0, confusion_matrix.background_index] == 1
    assert confusion_matrix.matrix_[1, confusion_matrix.background_index] == 0

    gt_boxes = torch.tensor([[0, 0, 0, 2, 2, 2]], dtype=torch.float32)
    gt_cls = torch.tensor([-1], dtype=torch.float32)
    confusion_matrix.process_batch(None, gt_boxes, gt_cls)

    assert (
        confusion_matrix.matrix_[
            confusion_matrix.background_index,
            confusion_matrix.unclassified_index,
        ]
        >= 1
    )


def test_confusion_matrix_process_batch_counts_tp_fp_and_fn() -> None:
    confusion_matrix = ConfusionMatrix(nc=1, conf=0.2, iou_thres=0.1)
    gt_boxes = torch.tensor([[0, 0, 0, 2, 2, 2]], dtype=torch.float32)
    gt_cls = torch.tensor([0], dtype=torch.float32)
    detections = torch.tensor(
        [
            [0, 0, 0, 2, 2, 2, 0.9, 0],
            [4, 4, 4, 6, 6, 6, 0.8, 0],
        ],
        dtype=torch.float32,
    )

    confusion_matrix.process_batch(detections, gt_boxes, gt_cls)

    assert confusion_matrix.matrix_[0, 0] == 1
    assert confusion_matrix.matrix_[0, confusion_matrix.background_index] == 1


def test_confusion_matrix_render_sets_axes_labels_and_title() -> None:
    confusion_matrix = ConfusionMatrix(nc=1)
    fig, ax = plt.subplots(1, 1)

    confusion_matrix.render(ax=ax, normalize=True, names=("lesion",))

    assert ax.get_xlabel() == "Reference"
    assert ax.get_ylabel() == "Predicted"
    assert "Confusion Matrix" in ax.get_title()
    plt.close(fig)


def test_plot_curves_and_projection_helpers() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    px = np.linspace(0, 1, 50, dtype=np.float32)
    py_pr = np.stack([np.linspace(1, 0.5, 50), np.linspace(0.9, 0.4, 50)], axis=0)
    ap = np.array([[0.5], [0.4]], dtype=np.float32)

    _plot_pr_curve_on_axis(axes[0], px, py_pr, ap, names={0: "a", 1: "b"})
    _plot_mc_curve_on_axis(
        axes[1],
        px,
        np.stack([np.linspace(0.1, 0.9, 50), np.linspace(0.2, 0.8, 50)], axis=0),
        names={0: "a", 1: "b"},
        ylabel="F1",
    )

    assert axes[0].get_title() == "Precision-Recall Curve"
    assert axes[1].get_title() == "F1-Confidence Curve"
    plt.close(fig)

    projected = _project_points_onto_pr_curve([0.1, 0.5, 1.5], px, np.linspace(1, 0, 50))
    assert len(projected) == 3
    assert projected[0] >= projected[1] >= projected[2]


def test_froc_curve_targets_and_score() -> None:
    tp = np.array([[1.0], [0.0], [1.0]], dtype=np.float32)
    fp = np.array([[0.0], [1.0], [1.0]], dtype=np.float32)
    conf = np.array([0.95, 0.6, 0.3], dtype=np.float32)
    target_cls = np.array([0, 0], dtype=np.int64)

    avg_fp, sensitivity, confidence = _compute_froc_curve(tp, fp, conf, target_cls, num_images=2)

    assert avg_fp[0] == 0.0
    assert sensitivity[0] == 0.0
    assert confidence[0] == 1.0

    indices = _select_froc_target_indices(avg_fp, targets=(0.125, 0.5, 2.0))
    assert set(indices) == {0.125, 0.5, 2.0}

    score = _compute_froc_score(avg_fp, sensitivity)
    assert 0.0 <= score <= 1.0


def test_compute_class_agnostic_map_handles_empty_and_nonempty_inputs() -> None:
    map10_empty, map_empty = compute_class_agnostic_map(
        np.zeros((0, 18), dtype=np.float32),
        np.zeros((0, 18), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
    )
    assert map10_empty == 0.0
    assert map_empty == 0.0

    tp = np.array([[1.0] * 18], dtype=np.float32)
    fp = np.array([[0.0] * 18], dtype=np.float32)
    conf = np.array([0.9], dtype=np.float32)
    target_cls = np.array([0], dtype=np.int64)
    map10, map_all = compute_class_agnostic_map(tp, fp, conf, target_cls)
    assert 0.0 <= map10 <= 1.0
    assert 0.0 <= map_all <= 1.0


def test_ap_per_class_returns_empty_arrays_when_only_unclassified_targets() -> None:
    tp = np.zeros((1, 18), dtype=np.float32)
    fp = np.ones((1, 18), dtype=np.float32)
    conf = np.array([0.8], dtype=np.float32)
    pred_cls = np.array([0], dtype=np.int64)
    target_cls = np.array([-1], dtype=np.int64)

    result = ap_per_class(tp, fp, conf, pred_cls, target_cls)

    assert result[0].shape == (0,)
    assert result[5].shape == (0, 18)
    assert result[7].shape == (0, 1000)


def test_detmetrics_process_and_plot_summary(tmp_path: Path) -> None:
    metrics = DetMetrics(save_dir=tmp_path, names={0: "lesion"})

    tp = np.array(
        [
            [1.0] * 18,
            [0.0] * 18,
            [0.0] * 18,
        ],
        dtype=np.float32,
    )
    fp = np.array(
        [
            [0.0] * 18,
            [1.0] * 18,
            [1.0] * 18,
        ],
        dtype=np.float32,
    )
    conf = np.array([0.9, 0.6, 0.2], dtype=np.float32)
    pred_cls = np.array([0, 0, 0], dtype=np.int64)
    target_cls = np.array([0, 0], dtype=np.int64)

    metrics.process(
        tp=tp,
        fp=fp,
        tp_agnostic=tp,
        fp_agnostic=fp,
        conf=conf,
        pred_cls=pred_cls,
        target_cls=target_cls,
        num_images=2,
    )

    confusion_matrix = ConfusionMatrix(nc=1)
    metrics.plot_summary(confusion_matrix=confusion_matrix)

    assert (tmp_path / "validation_summary.png").exists()
    assert metrics.metrics.best_confidence <= 1.0
    assert metrics.results_dict["metrics/mAP10"] >= 0.0


def test_detmetrics_process_handles_empty_class_curve() -> None:
    metrics = DetMetrics(save_dir=Path(), names={0: "lesion"})
    tp = np.zeros((1, 18), dtype=np.float32)
    fp = np.ones((1, 18), dtype=np.float32)
    conf = np.array([0.7], dtype=np.float32)
    pred_cls = np.array([0], dtype=np.int64)
    target_cls = np.array([-1], dtype=np.int64)

    metrics.process(
        tp=tp,
        fp=fp,
        tp_agnostic=tp,
        fp_agnostic=fp,
        conf=conf,
        pred_cls=pred_cls,
        target_cls=target_cls,
        num_images=1,
    )

    assert metrics.metrics.best_index == 0
    assert metrics.metrics.best_confidence == 1.0
