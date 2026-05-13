from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from radyolo.validation.metrics import (
    ConfusionMatrix,
    DetMetrics,
    Metric,
    _project_points_onto_pr_curve,
)


def test_detmetrics_reports_class_agnostic_map_for_class_confusion(tmp_path: Path) -> None:
    metrics = DetMetrics(save_dir=tmp_path, names={0: "a", 1: "b"})

    metrics.process(
        tp=np.zeros((1, 18), dtype=bool),
        fp=np.ones((1, 18), dtype=bool),
        tp_agnostic=np.ones((1, 18), dtype=bool),
        fp_agnostic=np.zeros((1, 18), dtype=bool),
        conf=np.array([0.95], dtype=np.float32),
        pred_cls=np.array([1], dtype=np.int64),
        target_cls=np.array([0], dtype=np.int64),
    )

    results = metrics.results_dict

    assert results["metrics/mAP10-95"] == pytest.approx(0.0)
    assert results["metrics/mAP10-95Agn"] == pytest.approx(0.995)
    assert metrics.metrics.map10_agnostic == pytest.approx(0.995)


def test_detmetrics_counts_minus_one_targets_in_class_agnostic_map(tmp_path: Path) -> None:
    metrics = DetMetrics(save_dir=tmp_path, names={0: "a", 1: "b"})

    metrics.process(
        tp=np.zeros((1, 18), dtype=bool),
        fp=np.zeros((1, 18), dtype=bool),
        tp_agnostic=np.ones((1, 18), dtype=bool),
        fp_agnostic=np.zeros((1, 18), dtype=bool),
        conf=np.array([0.95], dtype=np.float32),
        pred_cls=np.array([1], dtype=np.int64),
        target_cls=np.array([-1], dtype=np.int64),
    )

    results = metrics.results_dict

    assert results["metrics/mAP10-95"] == pytest.approx(0.0)
    assert results["metrics/mAP10-95Agn"] == pytest.approx(0.995)
    assert metrics.metrics.map10_agnostic == pytest.approx(0.995)


def test_detmetrics_summary_metrics_exclude_agnostic_map(tmp_path: Path) -> None:
    metrics = DetMetrics(save_dir=tmp_path, names={0: "a", 1: "b"})

    metrics.process(
        tp=np.zeros((1, 18), dtype=bool),
        fp=np.ones((1, 18), dtype=bool),
        tp_agnostic=np.ones((1, 18), dtype=bool),
        fp_agnostic=np.zeros((1, 18), dtype=bool),
        conf=np.array([0.95], dtype=np.float32),
        pred_cls=np.array([1], dtype=np.int64),
        target_cls=np.array([0], dtype=np.int64),
    )

    assert metrics.summary_keys == [
        "metrics/precision",
        "metrics/recall",
        "metrics/mAP10",
        "metrics/mAP10-95",
    ]
    assert metrics.summary_mean_results() == [0.0, 0.0, 0.0, 0.0]


def test_detmetrics_plot_summary_saves_combined_validation_figure(tmp_path: Path) -> None:
    metrics = DetMetrics(save_dir=tmp_path, names={0: "lesion"})

    tp = np.ones((1, 18), dtype=bool)
    fp = np.zeros((1, 18), dtype=bool)
    conf = np.array([0.9], dtype=np.float32)
    pred_cls = np.array([0], dtype=np.int64)
    target_cls = np.array([0], dtype=np.int64)

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

    confusion_matrix = ConfusionMatrix(nc=1, conf=metrics.metrics.best_confidence)
    detections = torch.tensor([[0, 0, 0, 4, 4, 4, 0.9, 0]], dtype=torch.float32)
    gt_boxes = torch.tensor([[0, 0, 0, 4, 4, 4]], dtype=torch.float32)
    gt_cls = torch.tensor([0], dtype=torch.float32)
    confusion_matrix.process_batch(detections=detections, gt_bboxes=gt_boxes, gt_cls=gt_cls)

    metrics.plot_summary(confusion_matrix=confusion_matrix)

    output_path = tmp_path / "validation_summary.png"
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_detmetrics_plot_summary_handles_unprocessed_metrics(tmp_path: Path) -> None:
    metrics = DetMetrics(save_dir=tmp_path, names={0: "lesion"})

    confusion_matrix = ConfusionMatrix(nc=1, conf=0.5)
    metrics.plot_summary(confusion_matrix=confusion_matrix)

    output_path = tmp_path / "validation_summary.png"
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_metric_update_accepts_named_arguments() -> None:
    metric = Metric()
    metric.update(
        p=np.array([0.5], dtype=np.float32),
        r=np.array([0.6], dtype=np.float32),
        f1=np.array([0.55], dtype=np.float32),
        all_ap=np.array([[0.7] * 18], dtype=np.float32),
        ap_class_index=np.array([0], dtype=np.int64),
        p_curve=np.ones((1, 1000), dtype=np.float32) * 0.5,
        r_curve=np.ones((1, 1000), dtype=np.float32) * 0.6,
        f1_curve=np.ones((1, 1000), dtype=np.float32) * 0.55,
        px=np.linspace(0, 1, 1000, dtype=np.float32),
        prec_values=np.ones((1, 1000), dtype=np.float32) * 0.5,
        map10_agnostic_value=0.8,
        map_agnostic_value=0.75,
        best_index=123,
        best_confidence=0.42,
        froc_avg_fp=np.array([0.125, 0.25], dtype=np.float32),
        froc_sensitivity=np.array([0.5, 0.75], dtype=np.float32),
        froc_confidence=np.array([0.9, 0.4], dtype=np.float32),
    )

    assert metric.p[0] == pytest.approx(0.5)
    assert metric.best_confidence == pytest.approx(0.42)
    assert metric.froc_avg_fp[1] == pytest.approx(0.25)


def test_project_points_onto_pr_curve_uses_pr_curve_values() -> None:
    recall_axis = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    precision_curve = np.array([1.0, 0.7, 0.4], dtype=np.float32)

    projected = _project_points_onto_pr_curve(
        recall_values=[0.25, 0.5, 0.75],
        recall_axis=recall_axis,
        precision_curve=precision_curve,
    )

    assert projected == pytest.approx([0.85, 0.7, 0.55])