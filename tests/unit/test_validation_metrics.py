from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from radyolo.data.image_labels import ImageLabelTaskSpec
from radyolo.validation.validate import Validator
from radyolo.validation.metrics import (
    ConfusionMatrix,
    DetMetrics,
    Metric,
    _project_points_onto_pr_curve,
    _smoothed_mean_confidence_curve,
    smooth,
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


def test_validator_reports_image_label_metrics(tmp_path: Path) -> None:
    config = type(
        "Config",
        (),
        {
            "confidence_threshold_validation": 0.001,
            "plots": False,
            "verbose": False,
        },
    )()
    validator = Validator(dataloader=[], save_dir=tmp_path, config=config, names={0: "lesion"})
    validator.image_label_task_specs = {
        "stage": ImageLabelTaskSpec(
            type="multiclass",
            classes={0: "I", 1: "II", 2: "III"},
        ),
        "rfs": ImageLabelTaskSpec(
            type="survival",
            masking="relapse",
            survival_bin_thresholds=(0.0, 1.0, 2.0),
        ),
    }
    validator.init_metrics()
    validator.image_label_multiclass_targets["stage"] = [np.array([0, 1, 2], dtype=np.int64)]
    validator.image_label_multiclass_logits["stage"] = [
        np.array(
            [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ],
            dtype=np.float64,
        )
    ]
    validator.image_label_survival_times["rfs"] = [np.array([1.0, 2.0, 3.0], dtype=np.float64)]
    validator.image_label_survival_events["rfs"] = [
        np.array([True, True, False], dtype=np.bool_)
    ]
    validator.image_label_survival_logits["rfs"] = [
        np.array(
            [
                [-6.0, -6.0, -6.0],
                [0.0, 0.0, 0.0],
                [6.0, 6.0, 6.0],
            ],
            dtype=np.float64,
        )
    ]

    results = validator._get_image_label_stats()

    assert results["metrics/image_label_stage_balanced_accuracy"] == pytest.approx(1.0)
    assert results["metrics/image_label_stage_multiclass_roc_auc"] == pytest.approx(1.0)
    assert results["metrics/image_label_rfs_c_index"] == pytest.approx(1.0)
    assert results["metrics/image_label_balanced_accuracy"] == pytest.approx(1.0)
    assert results["metrics/image_label_multiclass_roc_auc"] == pytest.approx(1.0)
    assert results["metrics/image_label_c_index"] == pytest.approx(1.0)


def test_validator_replaces_undefined_multiclass_roc_auc_with_random_baseline(
    tmp_path: Path,
) -> None:
    config = type(
        "Config",
        (),
        {
            "confidence_threshold_validation": 0.001,
            "plots": False,
            "verbose": False,
        },
    )()
    validator = Validator(dataloader=[], save_dir=tmp_path, config=config, names={0: "lesion"})
    validator.image_label_task_specs = {
        "stage": ImageLabelTaskSpec(
            type="multiclass",
            classes={0: "I", 1: "II", 2: "III"},
        )
    }
    validator.init_metrics()
    validator.image_label_multiclass_targets["stage"] = [np.array([0, 1, 0, 1], dtype=np.int64)]
    validator.image_label_multiclass_logits["stage"] = [
        np.array(
            [
                [8.0, 0.0, -8.0],
                [0.0, 8.0, -8.0],
                [7.0, 0.0, -8.0],
                [0.0, 7.0, -8.0],
            ],
            dtype=np.float64,
        )
    ]

    results = validator._get_image_label_stats()

    assert results["metrics/image_label_stage_multiclass_roc_auc"] == pytest.approx(0.5)
    assert np.isfinite(results["metrics/image_label_multiclass_roc_auc"])


def test_validator_print_results_includes_image_label_metrics_in_main_summary(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = type(
        "Config",
        (),
        {
            "confidence_threshold_validation": 0.001,
            "plots": False,
            "verbose": False,
        },
    )()
    validator = Validator(dataloader=[], save_dir=tmp_path, config=config, names={0: "lesion"})
    validator.image_label_task_specs = {
        "stage": ImageLabelTaskSpec(
            type="multiclass",
            classes={0: "I", 1: "II", 2: "III"},
        ),
        "rfs": ImageLabelTaskSpec(
            type="survival",
            masking="relapse",
            survival_bin_thresholds=(0.0, 1.0, 2.0),
        ),
    }
    validator.init_metrics()
    validator.seen = 133
    validator.nt_per_class = np.array([387], dtype=np.int64)
    validator.nt_per_image = np.array([133], dtype=np.int64)
    validator.image_label_multiclass_targets["stage"] = [np.array([0, 1, 2], dtype=np.int64)]
    validator.image_label_multiclass_logits["stage"] = [
        np.array(
            [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ],
            dtype=np.float64,
        )
    ]
    validator.image_label_survival_times["rfs"] = [np.array([1.0, 2.0, 3.0], dtype=np.float64)]
    validator.image_label_survival_events["rfs"] = [
        np.array([True, True, False], dtype=np.bool_)
    ]
    validator.image_label_survival_logits["rfs"] = [
        np.array(
            [
                [-6.0, -6.0, -6.0],
                [0.0, 0.0, 0.0],
                [6.0, 6.0, 6.0],
            ],
            dtype=np.float64,
        )
    ]

    caplog.set_level("INFO")
    validator.print_results()

    assert "BalAcc" in caplog.text
    assert "mcAUC" in caplog.text
    assert "C-Index" in caplog.text
    assert "Image labels " not in caplog.text


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


def test_smoothed_mean_confidence_curve_matches_plot_line_definition() -> None:
    px = np.linspace(0, 1, 5, dtype=np.float32)
    py = np.array(
        [
            [0.1, 0.3, 0.7, 0.2, 0.1],
            [0.2, 0.4, 0.6, 0.3, 0.0],
        ],
        dtype=np.float32,
    )

    smoothed_curve = _smoothed_mean_confidence_curve(py, px)

    assert smoothed_curve == pytest.approx(smooth(py.mean(0), 0.05))
    assert _smoothed_mean_confidence_curve(np.zeros((0, 5), dtype=np.float32), px) == pytest.approx(
        np.zeros_like(px)
    )