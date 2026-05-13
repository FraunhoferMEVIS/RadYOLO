from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import yaml

from radyolo.utils.ops import box_iou
from radyolo.validation.metrics import DetMetrics

if TYPE_CHECKING:
    from pathlib import Path

IOU_THRESHOLDS = torch.linspace(0.1, 0.95, 18)


def _parse_label_line(line: str) -> tuple[int, float, float, float, float, float, float]:
    parts = line.strip().split()
    if len(parts) != 7:
        msg = (
            "Label line must have 7 entries: class x y z w h d (normalized), "
            f"but has {len(parts)} entries."
        )
        raise ValueError(msg)
    cls = int(float(parts[0]))
    x, y, z, w, h, d = map(float, parts[1:])
    return cls, x, y, z, w, h, d


def _normalized_xyzwhd_to_xyzxyz(
    x: float,
    y: float,
    z: float,
    w: float,
    h: float,
    d: float,
) -> list[float]:
    half_w = w / 2.0
    half_h = h / 2.0
    half_d = d / 2.0
    return [x - half_w, y - half_h, z - half_d, x + half_w, y + half_h, z + half_d]


def _load_reference_case(labels_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    classes: list[int] = []
    boxes: list[list[float]] = []
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, x, y, z, w, h, d = _parse_label_line(line)
        classes.append(cls)
        boxes.append(_normalized_xyzwhd_to_xyzxyz(x, y, z, w, h, d))

    if not classes:
        return (
            torch.zeros((0,), dtype=torch.int64),
            torch.zeros((0, 6), dtype=torch.float32),
        )
    return (
        torch.tensor(classes, dtype=torch.int64),
        torch.tensor(boxes, dtype=torch.float32),
    )


def _load_prediction_case(prediction_path: Path) -> torch.Tensor:
    parsed = json.loads(prediction_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        msg = f"Prediction file {prediction_path} must contain a list of detections."
        raise TypeError(msg)

    detections: list[list[float]] = []
    for index, entry in enumerate(parsed):
        if not isinstance(entry, dict):
            msg = f"Prediction {index} in {prediction_path} must be a JSON object."
            raise TypeError(msg)
        rel_box = entry.get("box_relative_xyzxyz")
        if not isinstance(rel_box, list) or len(rel_box) != 6:
            msg = (
                f"Prediction {index} in {prediction_path} is missing box_relative_xyzxyz with 6 values."
            )
            raise ValueError(msg)
        class_id = int(entry["class"])
        confidence = float(entry["confidence"])
        detections.append([*map(float, rel_box), confidence, float(class_id)])

    if not detections:
        return torch.zeros((0, 8), dtype=torch.float32)
    return torch.tensor(detections, dtype=torch.float32)


def _match_predictions(
    pred_classes: torch.Tensor,
    true_classes: torch.Tensor,
    iou: torch.Tensor,
    *,
    class_agnostic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    true_positives_np = np.zeros((pred_classes.shape[0], IOU_THRESHOLDS.shape[0]), dtype=bool)
    true_positives_missing_np = np.zeros(
        (pred_classes.shape[0], IOU_THRESHOLDS.shape[0]),
        dtype=bool,
    )

    if class_agnostic:
        correct_class = torch.ones_like(iou, dtype=torch.bool)
        missing_class = torch.zeros_like(iou, dtype=torch.bool)
    else:
        correct_class = true_classes[:, None] == pred_classes
        missing_class = torch.logical_and(
            true_classes[:, None] != pred_classes,
            true_classes[:, None] == -1,
        )

    iou_tp = iou * correct_class
    iou_missing_class = iou * missing_class
    iou_tp_np = iou_tp.cpu().numpy()
    iou_missing_np = iou_missing_class.cpu().numpy()

    for threshold_index, iou_threshold in enumerate(IOU_THRESHOLDS.cpu().tolist()):
        matches_tp = np.array(np.nonzero(iou_tp_np >= iou_threshold)).T
        matches_missing = np.array(np.nonzero(iou_missing_np >= iou_threshold)).T

        if matches_tp.shape[0] > 0:
            sorted_matches_tp = matches_tp[
                iou_tp_np[matches_tp[:, 0], matches_tp[:, 1]].argsort()[::-1]
            ]
            unique_matches_tp = sorted_matches_tp[
                np.unique(sorted_matches_tp[:, 1], return_index=True)[1]
            ]
            unique_matches_tp = unique_matches_tp[
                np.unique(unique_matches_tp[:, 0], return_index=True)[1]
            ]
            true_positives_np[unique_matches_tp[:, 1].astype(int), threshold_index] = True

        if matches_missing.shape[0] > 0:
            sorted_matches_missing = matches_missing[
                iou_missing_np[matches_missing[:, 0], matches_missing[:, 1]].argsort()[::-1]
            ]
            unique_matches_missing = sorted_matches_missing[
                np.unique(sorted_matches_missing[:, 1], return_index=True)[1]
            ]
            unique_matches_missing = unique_matches_missing[
                np.unique(unique_matches_missing[:, 0], return_index=True)[1]
            ]
            true_positives_missing_np[
                unique_matches_missing[:, 1].astype(int),
                threshold_index,
            ] = True

    true_positives = torch.tensor(true_positives_np, dtype=torch.bool)
    true_positives_missing = torch.tensor(true_positives_missing_np, dtype=torch.bool)
    false_positives = torch.logical_and(
        torch.logical_not(true_positives),
        torch.logical_not(true_positives_missing),
    )
    return true_positives, false_positives


def _concat_or_default(
    tensors: list[torch.Tensor],
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> np.ndarray:
    if not tensors:
        return np.zeros(shape, dtype=dtype)
    return torch.cat(tensors, 0).cpu().numpy()


def _load_names(labels_dir: Path, observed_class_ids: set[int]) -> dict[int, str]:
    dataset_yaml = labels_dir.parent / "dataset.yaml"
    if dataset_yaml.exists():
        parsed = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
        raw_names = parsed.get("names", {}) if isinstance(parsed, dict) else {}
        if isinstance(raw_names, list):
            return {index: str(name) for index, name in enumerate(raw_names)}
        if isinstance(raw_names, dict):
            return {int(key): str(value) for key, value in sorted(raw_names.items())}

    max_class_id = max(observed_class_ids, default=-1)
    return {class_id: f"class_{class_id}" for class_id in range(max_class_id + 1)}


def evaluate_prediction_folder(predictions_dir: Path, labels_dir: Path) -> dict[str, Any]:
    prediction_files = {path.stem: path for path in sorted(predictions_dir.glob("*.json"))}
    label_files = {path.stem: path for path in sorted(labels_dir.glob("*.txt"))}

    common_cases = sorted(set(prediction_files) & set(label_files))
    missing_in_predictions = sorted(set(label_files) - set(prediction_files))
    missing_in_labels = sorted(set(prediction_files) - set(label_files))

    if not common_cases:
        msg = "No common case names found between prediction JSON files and reference label files."
        raise ValueError(msg)

    tp_stats: list[torch.Tensor] = []
    fp_stats: list[torch.Tensor] = []
    tp_agnostic_stats: list[torch.Tensor] = []
    fp_agnostic_stats: list[torch.Tensor] = []
    conf_stats: list[torch.Tensor] = []
    pred_class_stats: list[torch.Tensor] = []
    target_class_stats: list[torch.Tensor] = []
    observed_class_ids: set[int] = set()
    total_predictions = 0
    total_references = 0

    for case_name in common_cases:
        detections = _load_prediction_case(prediction_files[case_name])
        gt_cls, gt_boxes = _load_reference_case(label_files[case_name])

        total_predictions += int(detections.shape[0])
        total_references += int(gt_cls.shape[0])

        observed_class_ids.update(int(cls) for cls in gt_cls.tolist() if int(cls) >= 0)
        if detections.shape[0]:
            observed_class_ids.update(int(cls) for cls in detections[:, 7].tolist())

        target_class_stats.append(gt_cls)

        if detections.shape[0] == 0:
            continue

        pred_cls = detections[:, 7].to(dtype=torch.int64)
        conf = detections[:, 6]
        if gt_boxes.shape[0] == 0:
            shape = (detections.shape[0], IOU_THRESHOLDS.shape[0])
            tp = torch.zeros(shape, dtype=torch.bool)
            fp = torch.ones(shape, dtype=torch.bool)
            tp_agnostic = torch.zeros(shape, dtype=torch.bool)
            fp_agnostic = torch.ones(shape, dtype=torch.bool)
        else:
            iou = box_iou(gt_boxes, detections[:, :6])
            tp, fp = _match_predictions(
                pred_cls,
                gt_cls,
                iou,
                class_agnostic=False,
            )
            tp_agnostic, fp_agnostic = _match_predictions(
                pred_cls,
                gt_cls,
                iou,
                class_agnostic=True,
            )

        tp_stats.append(tp)
        fp_stats.append(fp)
        tp_agnostic_stats.append(tp_agnostic)
        fp_agnostic_stats.append(fp_agnostic)
        conf_stats.append(conf)
        pred_class_stats.append(pred_cls)

    names = _load_names(labels_dir, observed_class_ids)
    metrics = DetMetrics(names=names)
    metrics.process(
        tp=_concat_or_default(tp_stats, shape=(0, IOU_THRESHOLDS.shape[0]), dtype=np.bool_),
        fp=_concat_or_default(fp_stats, shape=(0, IOU_THRESHOLDS.shape[0]), dtype=np.bool_),
        tp_agnostic=_concat_or_default(
            tp_agnostic_stats,
            shape=(0, IOU_THRESHOLDS.shape[0]),
            dtype=np.bool_,
        ),
        fp_agnostic=_concat_or_default(
            fp_agnostic_stats,
            shape=(0, IOU_THRESHOLDS.shape[0]),
            dtype=np.bool_,
        ),
        conf=_concat_or_default(conf_stats, shape=(0,), dtype=np.float32),
        pred_cls=_concat_or_default(pred_class_stats, shape=(0,), dtype=np.int64),
        target_cls=_concat_or_default(target_class_stats, shape=(0,), dtype=np.int64),
    )

    per_class: list[dict[str, Any]] = []
    for metric_index, class_id in enumerate(metrics.ap_class_index):
        precision, recall, map10, map10_95 = metrics.class_result(metric_index)
        per_class.append(
            {
                "class": int(class_id),
                "name": names.get(int(class_id), f"class_{int(class_id)}"),
                "precision": float(precision),
                "recall": float(recall),
                "mAP10": float(map10),
                "mAP10_95": float(map10_95),
            }
        )

    return {
        "prediction_dir": str(predictions_dir),
        "labels_dir": str(labels_dir),
        "case_counts": {
            "predictions": len(prediction_files),
            "references": len(label_files),
            "common": len(common_cases),
            "missing_in_predictions": len(missing_in_predictions),
            "missing_in_labels": len(missing_in_labels),
        },
        "missing_cases": {
            "missing_in_predictions": missing_in_predictions,
            "missing_in_labels": missing_in_labels,
        },
        "object_counts": {
            "predictions": total_predictions,
            "references": total_references,
        },
        "metrics": {
            key: float(value)
            for key, value in metrics.results_dict.items()
        },
        "per_class": per_class,
    }
