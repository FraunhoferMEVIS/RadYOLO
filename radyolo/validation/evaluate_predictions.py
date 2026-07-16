from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import yaml

from radyolo.data.object_labels import load_case_object_labels
from radyolo.utils.ops import box_iou, xyzwhd2xyzxyz
from radyolo.validation.metrics import FROC_TARGETS, DetMetrics, _compute_froc_score

if TYPE_CHECKING:
    from pathlib import Path

IOU_THRESHOLDS = torch.linspace(0.1, 0.95, 18)


def _mean_ap_at_iou(metrics: DetMetrics, iou_threshold: float) -> float:
    """Return the mean AP at a specific IoU threshold from the stored AP table."""
    if not isinstance(metrics.metrics.all_ap, np.ndarray) or metrics.metrics.all_ap.size == 0:
        return 0.0

    threshold_matches = torch.isclose(IOU_THRESHOLDS, torch.tensor(iou_threshold)).nonzero()
    if threshold_matches.numel() == 0:
        msg = f"Unsupported IoU threshold {iou_threshold}."
        raise ValueError(msg)

    threshold_index = int(threshold_matches[0].item())
    return float(metrics.metrics.all_ap[:, threshold_index].mean())


def _load_reference_case(labels_dir: Path, case_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one reference case into class and xyzxyz box tensors."""
    entries = load_case_object_labels(labels_dir, case_name)
    if not entries:
        return (
            torch.zeros((0,), dtype=torch.int64),
            torch.zeros((0, 6), dtype=torch.float32),
        )

    classes = [entry["class_"] for entry in entries]
    boxes_xyzwhd = torch.tensor(
        [entry["box_relative_xyzwhd"] for entry in entries],
        dtype=torch.float32,
    )
    return (
        torch.tensor(classes, dtype=torch.int64),
        xyzwhd2xyzxyz(boxes_xyzwhd),
    )


def _collect_label_files(labels_dir: Path) -> dict[str, Path]:
    """Collect reference label files by case, preferring JSON over TXT for each stem."""
    label_files = {path.stem: path for path in sorted(labels_dir.glob("*.txt"))}
    for path in sorted(labels_dir.glob("*.json")):
        label_files[path.stem] = path
    return label_files


def _load_prediction_case(prediction_path: Path) -> torch.Tensor:
    """Load one prediction case JSON into an (N, 8) detection tensor."""
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
                f"Prediction {index} in {prediction_path} is missing "
                "box_relative_xyzxyz with 6 values."
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
    """Match detections to references per IoU threshold and return TP/FP masks."""
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
    """Concatenate tensors or return an empty NumPy array with the requested shape."""
    if not tensors:
        return np.zeros(shape, dtype=dtype)
    return torch.cat(tensors, 0).cpu().numpy()


def _to_curve_metrics(
    tp: torch.Tensor,
    fp: torch.Tensor,
    num_references: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision, recall, F1, and FN arrays over IoU thresholds."""
    if tp.ndim != 2 or fp.ndim != 2 or tp.shape != fp.shape:
        msg = (
            "tp/fp tensors must have shape (num_predictions, num_thresholds), "
            f"got {tp.shape}/{fp.shape}"
        )
        raise ValueError(msg)

    tp_counts = tp.to(dtype=torch.int64).sum(dim=0).cpu().numpy().astype(np.float64, copy=False)
    fp_counts = fp.to(dtype=torch.int64).sum(dim=0).cpu().numpy().astype(np.float64, copy=False)

    if num_references > 0:
        fn_counts = np.maximum(float(num_references) - tp_counts, 0.0)
        recall = tp_counts / float(num_references)
    else:
        fn_counts = np.zeros_like(tp_counts)
        recall = np.where(fp_counts > 0.0, 0.0, 1.0)

    precision_den = tp_counts + fp_counts
    precision = np.divide(
        tp_counts,
        precision_den,
        out=np.zeros_like(tp_counts),
        where=precision_den > 0.0,
    )
    f1_den = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        f1_den,
        out=np.zeros_like(precision),
        where=f1_den > 0.0,
    )
    return precision, recall, f1, fn_counts


def _to_scalar_case_metrics(
    *,
    case_name: str,
    num_predictions: int,
    num_references: int,
    tp: torch.Tensor,
    fp: torch.Tensor,
    tp_agnostic: torch.Tensor,
    fp_agnostic: torch.Tensor,
) -> dict[str, Any]:
    """Build scalar per-case metrics and counts from TP/FP curves."""
    precision_curve, recall_curve, f1_curve, fn_curve = _to_curve_metrics(tp, fp, num_references)
    precision_curve_agn, recall_curve_agn, f1_curve_agn, fn_curve_agn = _to_curve_metrics(
        tp_agnostic,
        fp_agnostic,
        num_references,
    )
    iou10_index = 0

    return {
        "case": case_name,
        "counts": {
            "predictions": int(num_predictions),
            "references": int(num_references),
            "tp_iou10": int(tp[:, iou10_index].sum().item()),
            "fp_iou10": int(fp[:, iou10_index].sum().item()),
            "fn_iou10": int(fn_curve[iou10_index]),
            "tp_iou10_agnostic": int(tp_agnostic[:, iou10_index].sum().item()),
            "fp_iou10_agnostic": int(fp_agnostic[:, iou10_index].sum().item()),
            "fn_iou10_agnostic": int(fn_curve_agn[iou10_index]),
        },
        "metrics": {
            "precision_iou10": float(precision_curve[iou10_index]),
            "recall_iou10": float(recall_curve[iou10_index]),
            "f1_iou10": float(f1_curve[iou10_index]),
            "precision_iou10_95_mean": float(precision_curve.mean()),
            "recall_iou10_95_mean": float(recall_curve.mean()),
            "f1_iou10_95_mean": float(f1_curve.mean()),
            "precision_iou10_agnostic": float(precision_curve_agn[iou10_index]),
            "recall_iou10_agnostic": float(recall_curve_agn[iou10_index]),
            "f1_iou10_agnostic": float(f1_curve_agn[iou10_index]),
            "precision_iou10_95_mean_agnostic": float(precision_curve_agn.mean()),
            "recall_iou10_95_mean_agnostic": float(recall_curve_agn.mean()),
            "f1_iou10_95_mean_agnostic": float(f1_curve_agn.mean()),
        },
        # Lower values are worse and appear first after sorting.
        "ranking": {
            "worst_first_score": float(f1_curve.mean()),
        },
    }


def _sort_per_case_results(per_case_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort per-case metrics from worst to best using aggregate ranking keys."""
    return sorted(
        per_case_results,
        key=lambda case: (
            float(case["ranking"]["worst_first_score"]),
            float(case["metrics"]["recall_iou10_95_mean"]),
            float(case["metrics"]["precision_iou10_95_mean"]),
            case["case"],
        ),
    )


def _evaluate_single_case(
    *,
    case_name: str,
    detections: torch.Tensor,
    gt_cls: torch.Tensor,
    gt_boxes: torch.Tensor,
) -> tuple[
    dict[str, Any],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | None,
]:
    """Evaluate one case and return per-case metrics plus aggregate tensors."""
    pred_cls = detections[:, 7].to(dtype=torch.int64)
    conf = detections[:, 6]

    if gt_boxes.shape[0] == 0:
        shape = (detections.shape[0], IOU_THRESHOLDS.shape[0])
        tp = torch.zeros(shape, dtype=torch.bool)
        fp = torch.ones(shape, dtype=torch.bool)
        tp_agnostic = torch.zeros(shape, dtype=torch.bool)
        fp_agnostic = torch.ones(shape, dtype=torch.bool)
    elif detections.shape[0] == 0:
        shape = (0, IOU_THRESHOLDS.shape[0])
        tp = torch.zeros(shape, dtype=torch.bool)
        fp = torch.zeros(shape, dtype=torch.bool)
        tp_agnostic = torch.zeros(shape, dtype=torch.bool)
        fp_agnostic = torch.zeros(shape, dtype=torch.bool)
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

    per_case_metrics = _to_scalar_case_metrics(
        case_name=case_name,
        num_predictions=int(detections.shape[0]),
        num_references=int(gt_cls.shape[0]),
        tp=tp,
        fp=fp,
        tp_agnostic=tp_agnostic,
        fp_agnostic=fp_agnostic,
    )

    if detections.shape[0] == 0:
        return per_case_metrics, None

    return per_case_metrics, (tp, fp, tp_agnostic, fp_agnostic, conf, pred_cls)


def _load_names(labels_dir: Path, observed_class_ids: set[int]) -> dict[int, str]:
    """Resolve class names from dataset.yaml or fall back to generated names."""
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


def _normalize_case_name(case_name: str) -> str:
    """Normalize case identifiers so dots and underscores compare equally."""
    # Treat dots and underscores as equivalent delimiters when matching case IDs.
    return case_name.replace(".", "_")


def _index_cases_by_normalized_name(
    files: dict[str, Path],
    *,
    kind: str,
) -> dict[str, tuple[str, Path]]:
    """Index case files by normalized name and raise on normalization collisions."""
    indexed: dict[str, tuple[str, Path]] = {}
    for stem, path in files.items():
        normalized = _normalize_case_name(stem)
        existing = indexed.get(normalized)
        if existing is not None and existing[0] != stem:
            msg = (
                f"Ambiguous {kind} case names after normalization: "
                f"'{existing[0]}' and '{stem}' both normalize to '{normalized}'."
            )
            raise ValueError(msg)
        indexed[normalized] = (stem, path)
    return indexed


def evaluate_prediction_folder(predictions_dir: Path, labels_dir: Path) -> dict[str, Any]:
    """Evaluate a prediction folder against references and return metrics and curves."""
    prediction_files = {path.stem: path for path in sorted(predictions_dir.glob("*.json"))}
    label_files = _collect_label_files(labels_dir)

    prediction_index = _index_cases_by_normalized_name(prediction_files, kind="prediction")
    label_index = _index_cases_by_normalized_name(label_files, kind="reference")

    common_normalized_cases = sorted(set(prediction_index) & set(label_index))
    common_cases = [
        (
            prediction_index[normalized][0],
            label_index[normalized][0],
            prediction_index[normalized][1],
            label_index[normalized][1],
        )
        for normalized in common_normalized_cases
    ]

    missing_in_predictions = sorted(
        label_index[normalized][0] for normalized in set(label_index) - set(prediction_index)
    )
    missing_in_labels = sorted(
        prediction_index[normalized][0] for normalized in set(prediction_index) - set(label_index)
    )

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
    per_case_results: list[dict[str, Any]] = []

    for prediction_case_name, label_case_name, prediction_path, _label_path in common_cases:
        detections = _load_prediction_case(prediction_path)
        gt_cls, gt_boxes = _load_reference_case(labels_dir, label_case_name)

        total_predictions += int(detections.shape[0])
        total_references += int(gt_cls.shape[0])

        observed_class_ids.update(int(cls) for cls in gt_cls.tolist() if int(cls) >= 0)
        if detections.shape[0]:
            observed_class_ids.update(int(cls) for cls in detections[:, 7].tolist())

        target_class_stats.append(gt_cls)

        case_name = prediction_case_name
        if prediction_case_name != label_case_name:
            case_name = f"{prediction_case_name} (label:{label_case_name})"

        per_case_metrics, aggregate_payload = _evaluate_single_case(
            case_name=case_name,
            detections=detections,
            gt_cls=gt_cls,
            gt_boxes=gt_boxes,
        )
        per_case_results.append(per_case_metrics)

        if aggregate_payload is None:
            continue

        tp, fp, tp_agnostic, fp_agnostic, conf, pred_cls = aggregate_payload

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
        num_images=len(common_cases),
    )

    metrics_summary = {
        "metrics/precision": float(metrics.metrics.mp),
        "metrics/recall": float(metrics.metrics.mr),
        "metrics/mAP10": float(metrics.metrics.map10),
        "metrics/mAP30": _mean_ap_at_iou(metrics, 0.3),
        "metrics/mAP50": _mean_ap_at_iou(metrics, 0.5),
        "metrics/mAP10_agnostic": float(metrics.metrics.map10_agnostic),
        "metrics/mAP10_95": float(metrics.metrics.map),
        "metrics/mAP10_95_agnostic": float(metrics.metrics.map_agnostic),
    }
    metrics_summary["metrics/FROC"] = _compute_froc_score(
        metrics.metrics.froc_avg_fp,
        metrics.metrics.froc_sensitivity,
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

    pr_recall_axis = metrics.metrics.px.astype(np.float64, copy=False)
    pr_precision_curves = metrics.metrics.prec_values.astype(np.float64, copy=False)
    class_ids = [int(class_id) for class_id in metrics.ap_class_index]
    class_names_by_id = {
        class_id: names.get(class_id, f"class_{class_id}") for class_id in class_ids
    }
    num_pr_rows = int(pr_precision_curves.shape[0])
    per_class_precision = [
        {
            "class": class_id,
            "name": class_names_by_id.get(class_id, f"class_{class_id}"),
            "precision": pr_precision_curves[row_index].tolist(),
        }
        for row_index, class_id in enumerate(class_ids)
        if row_index < num_pr_rows
    ]

    return {
        "metrics": {
            "summary": metrics_summary,
            "per_class": per_class,
            "per_case": _sort_per_case_results(per_case_results),
        },
        "metadata": {
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
            "objects_counts": {
                "predictions": total_predictions,
                "references": total_references,
            },
        },
        "curves": {
            "froc": {
                "avg_fp_per_image": metrics.metrics.froc_avg_fp.astype(
                    np.float64,
                    copy=False,
                ).tolist(),
                "sensitivity": metrics.metrics.froc_sensitivity.astype(
                    np.float64,
                    copy=False,
                ).tolist(),
                "confidence": metrics.metrics.froc_confidence.astype(
                    np.float64,
                    copy=False,
                ).tolist(),
                "score_targets_fp_per_image": [float(target) for target in FROC_TARGETS],
                "score": float(metrics_summary["metrics/FROC"]),
            },
            "precision_recall": {
                "recall_axis": pr_recall_axis.tolist(),
                "mean_precision": (
                    pr_precision_curves.mean(axis=0)
                    if pr_precision_curves.size
                    else np.zeros_like(pr_recall_axis)
                ).tolist(),
                "per_class_precision": per_class_precision,
            },
        },
    }
