from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from radyolo.validation.metrics import DetMetrics


def test_detmetrics_reports_class_agnostic_map_for_class_confusion(tmp_path: Path) -> None:
    metrics = DetMetrics(save_dir=tmp_path, plot=False, names={0: "a", 1: "b"})

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
    assert metrics.box.map10_agnostic == pytest.approx(0.995)


def test_detmetrics_counts_minus_one_targets_in_class_agnostic_map(tmp_path: Path) -> None:
    metrics = DetMetrics(save_dir=tmp_path, plot=False, names={0: "a", 1: "b"})

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
    assert metrics.box.map10_agnostic == pytest.approx(0.995)


def test_detmetrics_summary_metrics_exclude_agnostic_map(tmp_path: Path) -> None:
    metrics = DetMetrics(save_dir=tmp_path, plot=False, names={0: "a", 1: "b"})

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