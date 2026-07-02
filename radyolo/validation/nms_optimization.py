from __future__ import annotations

import csv
import logging
from typing import TYPE_CHECKING

from radyolo.models.model import load_checkpoint_model
from radyolo.validation.metrics import _select_froc_target_indices
from radyolo.validation.validate import Validator

if TYPE_CHECKING:
    from pathlib import Path

    from radyolo.config.config import DefaultConfig


LOGGER = logging.getLogger(__name__)

OPERATING_POINT_FPPI_TARGETS: tuple[float, ...] = (0.1, 0.5, 1.0)


def _fppi_target_suffix(target: float) -> str:
    """Return a stable decimal suffix for FP/image target column names."""
    return f"{target:.1f}"


def _collect_operating_point_metrics(validator: Validator) -> dict[str, float]:
    """Collect confidence/sensitivity values for standard operating points."""
    metrics_state = validator.metrics.metrics
    target_indices = _select_froc_target_indices(
        metrics_state.froc_avg_fp,
        targets=OPERATING_POINT_FPPI_TARGETS,
    )
    operating_point_metrics: dict[str, float] = {}
    for target, index in target_indices.items():
        suffix = _fppi_target_suffix(target)
        operating_point_metrics[f"metrics/sensitivity_at_{suffix}_fppi"] = round(
            float(metrics_state.froc_sensitivity[index]),
            5,
        )
        operating_point_metrics[f"metrics/confidence_at_{suffix}_fppi"] = round(
            float(metrics_state.froc_confidence[index]),
            5,
        )

    operating_point_metrics["metrics/confidence_at_max_f1"] = round(
        float(metrics_state.best_confidence),
        5,
    )
    if hasattr(metrics_state.f1, "size") and metrics_state.f1.size:
        operating_point_metrics["metrics/f1"] = round(float(metrics_state.f1.mean()), 5)
    else:
        operating_point_metrics["metrics/f1"] = 0.0

    return operating_point_metrics


def compute_validation_fitness(
    metrics: dict[str, float],
    fitness_metric_weights: tuple[float, ...],
) -> float:
    """Compute weighted scalar fitness from validation metric values."""
    map_1095 = float(metrics.get("metrics/mAP10-95", 0.0))
    map_1095_agn = float(metrics.get("metrics/mAP10-95Agn", 0.0))
    dice = float(metrics.get("metrics/dice", 0.0))
    image_label_balanced_accuracy = float(metrics.get("metrics/image_label_balanced_accuracy", 0.0))
    image_label_multiclass_roc_auc = float(
        metrics.get("metrics/image_label_multiclass_roc_auc", 0.0)
    )
    image_label_c_index = float(metrics.get("metrics/image_label_c_index", 0.0))
    if len(fitness_metric_weights) == 3:
        fitness_metric_weights = (*fitness_metric_weights, 0.0, 0.0, 0.0)
    total_weight = sum(fitness_metric_weights)
    return (
        fitness_metric_weights[0] * map_1095
        + fitness_metric_weights[1] * map_1095_agn
        + fitness_metric_weights[2] * dice
        + fitness_metric_weights[3] * image_label_balanced_accuracy
        + fitness_metric_weights[4] * image_label_multiclass_roc_auc
        + fitness_metric_weights[5] * image_label_c_index
    ) / total_weight


def run_nms_threshold_optimization(
    *,
    validator: Validator,
    config: DefaultConfig,
    save_dir: Path,
    last_model_path: Path,
    best_model_path: Path,
    device: str | object,
) -> None:
    """Evaluate candidate NMS IoU thresholds for last/best checkpoints and save CSV."""
    candidate_thresholds = config.nms_iou_threshold_candidates
    if not candidate_thresholds:
        return

    csv_path = save_dir / "nms_optimization.csv"
    result_rows: list[dict[str, str | float | bool]] = []
    checkpoints = (
        ("last", last_model_path),
        ("best", best_model_path),
    )

    for checkpoint_name, checkpoint_path in checkpoints:
        if not checkpoint_path.exists():
            LOGGER.warning(
                "Skipping NMS IoU optimization for missing checkpoint %s.", checkpoint_path
            )
            continue
        LOGGER.info(
            f"Running NMS IoU optimization for checkpoint {checkpoint_path} "
            f"trying thresholds {candidate_thresholds}"
        )

        checkpoint_rows = evaluate_nms_thresholds_for_checkpoint(
            validator=validator,
            checkpoint_name=checkpoint_name,
            checkpoint_path=checkpoint_path,
            candidate_thresholds=candidate_thresholds,
            config=config,
            save_dir=save_dir,
            device=device,
        )
        if not checkpoint_rows:
            continue

        best_row = max(checkpoint_rows, key=lambda row: float(row["fitness"]))
        for row in checkpoint_rows:
            row["suggested"] = row is best_row
        result_rows.extend(checkpoint_rows)

        LOGGER.info(
            (
                "Suggested NMS IoU threshold for %s checkpoint: %.3f "
                "(fitness=%.5f, mAP10-95=%.5f, dice=%.5f)"
            ),
            checkpoint_name,
            float(best_row["nms_iou_threshold"]),
            float(best_row["fitness"]),
            float(best_row["metrics/mAP10-95"]),
            float(best_row["metrics/dice"]),
        )

    if not result_rows:
        return

    fieldnames = [
        "checkpoint",
        "checkpoint_path",
        "nms_iou_threshold",
        "suggested",
        "fitness",
        "metrics/precision",
        "metrics/recall",
        "metrics/mAP10",
        "metrics/mAP10Agn",
        "metrics/mAP10-95",
        "metrics/mAP10-95Agn",
        "metrics/dice",
        "metrics/f1",
        "metrics/sensitivity_at_0.1_fppi",
        "metrics/confidence_at_0.1_fppi",
        "metrics/sensitivity_at_0.5_fppi",
        "metrics/confidence_at_0.5_fppi",
        "metrics/sensitivity_at_1.0_fppi",
        "metrics/confidence_at_1.0_fppi",
        "metrics/confidence_at_max_f1",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    LOGGER.info("Saved NMS IoU optimization results to %s", csv_path)


def evaluate_nms_thresholds_for_checkpoint(
    *,
    validator: Validator,
    checkpoint_name: str,
    checkpoint_path: Path,
    candidate_thresholds: list[float],
    config: DefaultConfig,
    save_dir: Path,
    device: str | object,
) -> list[dict[str, str | float | bool]]:
    """Run validation for each candidate NMS threshold on one checkpoint."""
    loaded_checkpoint = load_checkpoint_model(checkpoint_path, device_override=device)
    evaluation_config = config.model_copy(update={"plots": False, "verbose": False})
    result_rows: list[dict[str, str | float | bool]] = []
    for threshold in candidate_thresholds:
        evaluation_config.nms_iou_threshold = threshold
        nms_validator = Validator(
            validator.dataloader,
            save_dir=save_dir,
            config=evaluation_config,
            names=validator.names,
        )
        metrics = nms_validator(
            trainer=None,
            model=loaded_checkpoint.model,
        )
        operating_point_metrics = _collect_operating_point_metrics(nms_validator)
        fitness = compute_validation_fitness(metrics, config.fitness_metric_weights)
        result_rows.append(
            {
                "checkpoint": checkpoint_name,
                "checkpoint_path": checkpoint_path.name,
                "nms_iou_threshold": float(threshold),
                "suggested": False,
                "fitness": round(float(fitness), 5),
                "metrics/precision": round(float(metrics.get("metrics/precision", 0.0)), 5),
                "metrics/recall": round(float(metrics.get("metrics/recall", 0.0)), 5),
                "metrics/mAP10": round(float(metrics.get("metrics/mAP10", 0.0)), 5),
                "metrics/mAP10Agn": round(float(metrics.get("metrics/mAP10Agn", 0.0)), 5),
                "metrics/mAP10-95": round(float(metrics.get("metrics/mAP10-95", 0.0)), 5),
                "metrics/mAP10-95Agn": round(float(metrics.get("metrics/mAP10-95Agn", 0.0)), 5),
                "metrics/dice": round(float(metrics.get("metrics/dice", 0.0)), 5),
                **operating_point_metrics,
            }
        )
    return result_rows
