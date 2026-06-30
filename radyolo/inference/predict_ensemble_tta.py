from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from radyolo.inference.predict_utils import (
    LoadedImageTask,
    WriteTask,
    build_prediction_entries,
    reconstruct_mask_probability_to_original_shape,
    to_numpy_float32_contiguous,
    to_torch_float32_contiguous,
)
from radyolo.utils import ops

if TYPE_CHECKING:
    from pathlib import Path

    from radyolo.config.config import DefaultConfig
    from radyolo.data.image_labels import ImageLabelTaskSpec
    from radyolo.inference.predict import PredictionConfig
    from radyolo.models.model import HardcodedModel


@dataclass(slots=True)
class _LoadedEnsembleModel:
    model: HardcodedModel
    train_config: DefaultConfig
    names: dict[int, str]
    image_label_task_specs: dict[str, ImageLabelTaskSpec]
    tabular_feature_names: tuple[str, ...]
    prediction_config: PredictionConfig


@dataclass(slots=True)
class _PassPrediction:
    detections: torch.Tensor
    masks: list[torch.Tensor | None]
    image_label_results: dict[str, object] | None
    original_shape: tuple[int, int, int]
    pass_index: int


def _iter_flip_axis_combinations(enabled: bool) -> list[tuple[int, ...]]:
    if not enabled:
        return [()]
    return [
        tuple(axis for axis, should_flip in enumerate(flags) if should_flip)
        for flags in itertools.product((False, True), repeat=3)
    ]


def _flip_loaded_image(loaded_image: LoadedImageTask, axes: tuple[int, ...]) -> LoadedImageTask:
    if not axes:
        return loaded_image
    return LoadedImageTask(
        case_name=loaded_image.case_name,
        affine=loaded_image.affine,
        image=np.ascontiguousarray(np.flip(loaded_image.image, axis=axes)),
        voxel_size_xyz=loaded_image.voxel_size_xyz,
    )


def _unflip_detections(
    detections: torch.Tensor,
    axes: tuple[int, ...],
    original_shape: tuple[int, int, int],
) -> torch.Tensor:
    if not axes or not detections.numel():
        return detections
    unflipped = detections.clone()
    d, h, w = original_shape
    axis_sizes = (w, h, d)
    for axis in axes:
        low_index = axis
        high_index = axis + 3
        old_low = unflipped[:, low_index].clone()
        old_high = unflipped[:, high_index].clone()
        unflipped[:, low_index] = axis_sizes[axis] - old_high
        unflipped[:, high_index] = axis_sizes[axis] - old_low
    return unflipped


def _unflip_mask(
    mask: torch.Tensor | None,
    axes: tuple[int, ...],
) -> torch.Tensor | None:
    if mask is None or not axes:
        return mask
    spatial_dims_zyx = {0: 2, 1: 1, 2: 0}
    return torch.flip(mask, dims=tuple(spatial_dims_zyx[axis] for axis in axes))


def _detections_from_write_task(write_task: WriteTask) -> torch.Tensor:
    rows: list[list[float]] = []
    for entry in write_task.image_results:
        box = entry.get("box_image_xyzxyz")
        if not isinstance(box, list) or len(box) != 6:
            continue
        rows.append(
            [
                *[float(value) for value in box],
                float(entry["confidence"]),
                float(entry["class"]),
            ]
        )
    if not rows:
        return torch.zeros((0, 8), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def _resize_mask_to_original_shape(
    mask_probability: torch.Tensor | np.ndarray,
    original_shape: tuple[int, int, int],
) -> torch.Tensor:
    mask_tensor = to_torch_float32_contiguous(mask_probability)
    if tuple(mask_tensor.shape) == original_shape:
        return mask_tensor
    return F.interpolate(
        mask_tensor.unsqueeze(0).unsqueeze(0),
        size=original_shape,
        mode="trilinear",
        align_corners=False,
    )[0, 0].to(dtype=torch.float32)


def _masks_from_write_task(
    write_task: WriteTask, detections: torch.Tensor
) -> list[torch.Tensor | None]:
    if not detections.numel():
        return []
    if write_task.prototype_masks is not None:
        masks = [
            reconstruct_mask_probability_to_original_shape(
                det,
                write_task.prototype_masks,
                write_task.original_shape,
            )
            for det in write_task.mask_detections
        ]
        return [*masks, *([None] * max(0, detections.shape[0] - len(masks)))]
    if write_task.merged_mask_probabilities:
        masks = [
            _resize_mask_to_original_shape(mask_probability, write_task.original_shape)
            for mask_probability in write_task.merged_mask_probabilities
        ]
        return [*masks, *([None] * max(0, detections.shape[0] - len(masks)))]
    return [None] * detections.shape[0]


def _write_task_to_pass_prediction(
    write_task: WriteTask,
    *,
    axes: tuple[int, ...],
    pass_index: int,
) -> _PassPrediction:
    detections = _unflip_detections(
        _detections_from_write_task(write_task),
        axes,
        write_task.original_shape,
    )
    masks = [
        _unflip_mask(mask, axes) for mask in _masks_from_write_task(write_task, detections)
    ]
    return _PassPrediction(
        detections=detections,
        masks=masks,
        image_label_results=write_task.image_label_results,
        original_shape=write_task.original_shape,
        pass_index=pass_index,
    )


def _average_numeric_lists(values: list[list[object]]) -> list[object]:
    width = min(len(value) for value in values)
    averaged: list[object] = []
    for index in range(width):
        column = [value[index] for value in values]
        if all(isinstance(value, (int, float)) for value in column):
            averaged.append(round(float(sum(column)) / len(column), 6))
        else:
            averaged.append(column[0])
    return averaged


def _average_image_label_values(values: list[object]) -> object:
    if all(isinstance(value, (int, float)) for value in values):
        return round(float(sum(values)) / len(values), 6)
    if all(isinstance(value, list) for value in values):
        return _average_numeric_lists(values)  # type: ignore[arg-type]
    if all(isinstance(value, dict) for value in values):
        dict_values = [value for value in values if isinstance(value, dict)]
        keys = set.intersection(*(set(value) for value in dict_values))
        averaged = {
            key: _average_image_label_values([value[key] for value in dict_values])
            for key in keys
        }
        probabilities = averaged.get("probabilities")
        if isinstance(probabilities, list) and probabilities:
            predicted_class = int(np.argmax(np.asarray(probabilities, dtype=np.float32)))
            averaged["predicted_class"] = predicted_class
            if "predicted_class_name" in averaged:
                averaged["predicted_class_name"] = str(predicted_class)
        return averaged
    return values[0]


def _average_image_label_results(
    pass_predictions: list[_PassPrediction],
) -> dict[str, object] | None:
    results = [prediction.image_label_results for prediction in pass_predictions]
    dict_results = [result for result in results if isinstance(result, dict)]
    if not dict_results:
        return None
    keys = set.intersection(*(set(result) for result in dict_results))
    return {
        key: _average_image_label_values([result[key] for result in dict_results])
        for key in keys
    }


def _flatten_pass_detections(
    pass_predictions: list[_PassPrediction],
) -> tuple[torch.Tensor, list[torch.Tensor | None], list[int]]:
    detections: list[torch.Tensor] = []
    masks: list[torch.Tensor | None] = []
    pass_indices: list[int] = []
    for prediction in pass_predictions:
        for detection_index, detection in enumerate(prediction.detections):
            detections.append(detection)
            masks.append(prediction.masks[detection_index])
            pass_indices.append(prediction.pass_index)
    if not detections:
        return torch.zeros((0, 8), dtype=torch.float32), [], []
    return torch.stack(detections, dim=0), masks, pass_indices


def _select_cluster_class(cluster: list[int], detections: torch.Tensor) -> float:
    cluster_detections = detections[cluster]
    class_confidences: dict[float, float] = {}
    for detection in cluster_detections:
        class_id = float(detection[7].item())
        class_confidences[class_id] = class_confidences.get(class_id, 0.0) + float(
            detection[6].item()
        )
    return max(class_confidences.items(), key=lambda item: item[1])[0]


def _aggregate_detection_clusters(
    pass_predictions: list[_PassPrediction],
    *,
    iou_threshold: float,
    confidence_threshold: float,
    class_agnostic: bool,
    total_passes: int,
    max_det: int,
) -> tuple[torch.Tensor, list[np.ndarray]]:
    detections, masks, pass_indices = _flatten_pass_detections(pass_predictions)
    if not detections.numel():
        return torch.zeros((0, 8), dtype=torch.float32), []

    order = detections[:, 6].argsort(descending=True).tolist()
    used: set[int] = set()
    clusters: list[list[int]] = []
    for anchor_index in order:
        if anchor_index in used:
            continue
        cluster = [anchor_index]
        used.add(anchor_index)
        cluster_passes = {pass_indices[anchor_index]}
        remaining = [index for index in order if index not in used]
        if remaining:
            ious = ops.box_iou(
                detections[anchor_index : anchor_index + 1, :6],
                detections[remaining, :6],
            )[0]
            ranked_matches = sorted(
                zip(remaining, ious.tolist(), strict=True),
                key=lambda item: item[1],
                reverse=True,
            )
            for candidate_index, iou in ranked_matches:
                if iou < iou_threshold or pass_indices[candidate_index] in cluster_passes:
                    continue
                if (
                    not class_agnostic
                    and detections[candidate_index, 7] != detections[anchor_index, 7]
                ):
                    continue
                cluster.append(candidate_index)
                used.add(candidate_index)
                cluster_passes.add(pass_indices[candidate_index])
        clusters.append(cluster)

    aggregated_rows: list[torch.Tensor] = []
    aggregated_masks: list[np.ndarray | None] = []
    for cluster in clusters:
        cluster_detections = detections[cluster]
        averaged = torch.zeros(8, dtype=torch.float32)
        averaged[:6] = cluster_detections[:, :6].mean(dim=0)
        averaged[6] = cluster_detections[:, 6].mean() * (len(cluster) / total_passes)
        averaged[7] = _select_cluster_class(cluster, detections)
        aggregated_rows.append(averaged)
        cluster_masks = [masks[index] for index in cluster if masks[index] is not None]
        if cluster_masks:
            averaged_mask = torch.stack(cluster_masks, dim=0).mean(dim=0)
            aggregated_masks.append(to_numpy_float32_contiguous(averaged_mask))
        else:
            aggregated_masks.append(None)

    aggregated = torch.stack(aggregated_rows, dim=0)
    confidence_keep = aggregated[:, 6] >= confidence_threshold
    aggregated = aggregated[confidence_keep]
    aggregated_masks = [
        mask for mask, keep in zip(aggregated_masks, confidence_keep.tolist(), strict=True) if keep
    ]
    if not aggregated.numel():
        return torch.zeros((0, 8), dtype=torch.float32), []

    order_tensor = aggregated[:, 6].argsort(descending=True)[:max_det]
    kept = aggregated[order_tensor]
    kept_masks = [
        aggregated_masks[index]
        for index in order_tensor.tolist()
        if index < len(aggregated_masks) and aggregated_masks[index] is not None
    ]
    return kept, kept_masks


def _model_paths(prediction_cfg: PredictionConfig) -> tuple[Path, ...]:
    return (prediction_cfg.model_path, *prediction_cfg.ensemble_model_paths)


def _load_ensemble_models(prediction_cfg: PredictionConfig) -> list[_LoadedEnsembleModel]:
    from radyolo.inference.predict import (  # noqa: PLC0415
        _load_model,
        _validate_loaded_model_prediction_config,
    )

    loaded_models: list[_LoadedEnsembleModel] = []
    for model_path in _model_paths(prediction_cfg):
        model_prediction_cfg = replace(prediction_cfg, model_path=model_path)
        model, train_config, names, image_label_task_specs, tabular_feature_names = _load_model(
            model_prediction_cfg
        )
        _validate_loaded_model_prediction_config(train_config, model_prediction_cfg)
        loaded_models.append(
            _LoadedEnsembleModel(
                model=model,
                train_config=train_config,
                names=names,
                image_label_task_specs=image_label_task_specs,
                tabular_feature_names=tabular_feature_names,
                prediction_config=model_prediction_cfg,
            )
        )
    return loaded_models


def predict_loaded_image_ensemble_tta(
    loaded_image: LoadedImageTask,
    *,
    loaded_models: list[_LoadedEnsembleModel],
    prediction_config: PredictionConfig,
    tabular_values: torch.Tensor | None,
    tabular_mask: torch.Tensor | None,
) -> WriteTask:
    from radyolo.inference.predict import _predict_loaded_image  # noqa: PLC0415

    pass_predictions: list[_PassPrediction] = []
    flip_axes = _iter_flip_axis_combinations(prediction_config.tta_flipping)
    pass_index = 0
    for loaded_model in loaded_models:
        for axes in flip_axes:
            write_task = _predict_loaded_image(
                _flip_loaded_image(loaded_image, axes),
                model=loaded_model.model,
                train_config=loaded_model.train_config,
                names=loaded_model.names,
                image_label_task_specs=loaded_model.image_label_task_specs,
                prediction_config=loaded_model.prediction_config,
                tabular_values=tabular_values,
                tabular_mask=tabular_mask,
            )
            pass_predictions.append(
                _write_task_to_pass_prediction(
                    write_task,
                    axes=axes,
                    pass_index=pass_index,
                )
            )
            pass_index += 1

    reference_model = loaded_models[0]
    output_original_shape = pass_predictions[0].original_shape
    aggregated_detections, aggregated_masks = _aggregate_detection_clusters(
        pass_predictions,
        iou_threshold=prediction_config.ensembling_iou,
        confidence_threshold=prediction_config.ensembling_confidence_threshold,
        class_agnostic=prediction_config.ensembling_class_agnostic,
        total_passes=max(pass_index, 1),
        max_det=reference_model.train_config.max_det,
    )
    image_results, _ = build_prediction_entries(
        aggregated_detections,
        case_name=loaded_image.case_name,
        affine=loaded_image.affine,
        original_shape=output_original_shape,
        names=reference_model.names,
        include_masks=prediction_config.save_masks and bool(aggregated_masks),
    )
    return WriteTask(
        case_name=loaded_image.case_name,
        image_results=image_results,
        image_label_results=_average_image_label_results(pass_predictions),
        affine=loaded_image.affine,
        original_shape=output_original_shape,
        prototype_masks=None,
        mask_detections=[],
        merged_mask_probabilities=aggregated_masks if prediction_config.save_masks else [],
    )
