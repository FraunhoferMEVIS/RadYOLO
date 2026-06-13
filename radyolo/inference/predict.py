from __future__ import annotations

import contextlib
import itertools
import json
import logging
import multiprocessing as mp
import queue
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from tqdm_loggable.auto import tqdm

from radyolo.config.config import DefaultConfig, DetectionPostprocessConfig
from radyolo.config.spatial import VoxelSize
from radyolo.data.preprocessing import preprocess_image
from radyolo.models.model import load_checkpoint_model
from radyolo.utils import ops

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from radyolo.config.spatial import PatchSize
    from radyolo.models.layers import ModelOutput
    from radyolo.models.model import HardcodedModel

LOGGER = logging.getLogger(__name__)

_SENTINEL = None


@dataclass(slots=True)
class _LoadedImageTask:
    case_name: str
    affine: np.ndarray
    image: np.ndarray
    voxel_size_xyz: tuple[float, float, float] | None = None


@dataclass(slots=True)
class _WriteTask:
    case_name: str
    image_results: list[dict[str, object]]
    affine: np.ndarray
    original_shape: tuple[int, int, int]
    prototype_masks: np.ndarray | None
    mask_detections: list[np.ndarray]
    merged_mask_probabilities: list[np.ndarray] = field(default_factory=list)


@dataclass(slots=True)
class _WorkerFailure:
    stage: str
    case_name: str | None
    error: str
    traceback_text: str


@dataclass
class PredictionConfig:
    model_path: Path
    input_dir: Path
    output_dir: Path
    confidence_threshold: float
    nms_iou_threshold: float
    device: str
    clip_boxes_to_image: bool
    overwrite_patch_size: PatchSize | None = None
    overwrite_normalization: dict | list[dict] | None = None
    overwrite_voxel_size: VoxelSize | None = None
    patch_overlap: float = 0.5
    patch_border_weight: float = 0.9
    preprocessing_on_gpu: bool = True
    num_load_workers: int = 2
    num_write_workers: int = 2
    split_file: Path | None = None
    split_fold: int = 0
    split_group: str = "test"


def _load_model(cfg: PredictionConfig) -> tuple[torch.nn.Module, DefaultConfig, dict[int, str]]:
    loaded_model = load_checkpoint_model(cfg.model_path, device_override=cfg.device)
    train_config = loaded_model.train_config
    num_classes = train_config.num_classes
    names = {i: f"class_{i}" for i in range(num_classes)}

    return loaded_model.model, train_config, names


def _iter_nifti(input_dir: Path) -> Iterable[Path]:
    return sorted(p for p in input_dir.glob("*.nii.gz") if p.is_file())


def _case_name_from_path(img_path: Path) -> str:
    return img_path.stem.replace(".nii", "")


def _load_case_names_from_split_file(
    split_file: Path,
    *,
    fold: int,
    group: str,
) -> set[str]:
    with split_file.open(encoding="utf-8") as file:
        split_data = json.load(file)

    if isinstance(split_data, dict):
        folds = [split_data]
    elif isinstance(split_data, list):
        folds = split_data
    else:
        msg = "Split file must contain either a dict or a list of dicts."
        raise TypeError(msg)

    if fold < 0 or fold >= len(folds):
        msg = f"Split fold index {fold} is out of range for file {split_file}."
        raise ValueError(msg)

    fold_entry = folds[fold]
    if not isinstance(fold_entry, dict):
        msg = f"Split fold entry at index {fold} must be a dict."
        raise TypeError(msg)

    selected_cases = fold_entry.get(group)
    if not isinstance(selected_cases, list) or not all(
        isinstance(case_name, str) for case_name in selected_cases
    ):
        msg = (
            f"Split group '{group}' must exist in fold {fold} and map to a list "
            "of case names."
        )
        raise ValueError(msg)

    return set(selected_cases)


def _resolve_normalization(
    train_config: DefaultConfig,
    prediction_config: PredictionConfig,
) -> dict | list[dict]:
    return (
        prediction_config.overwrite_normalization
        if prediction_config.overwrite_normalization is not None
        else train_config.normalization
    )


def _resolve_patch_size(
    train_config: DefaultConfig,
    prediction_config: PredictionConfig,
) -> PatchSize:
    return (
        prediction_config.overwrite_patch_size
        if prediction_config.overwrite_patch_size is not None
        else train_config.patch_size
    )


def _resolve_input_voxel_size(loaded_image: _LoadedImageTask) -> VoxelSize | None:
    if loaded_image.voxel_size_xyz is None:
        return None
    return VoxelSize.from_xyz(loaded_image.voxel_size_xyz)


def _resolve_target_voxel_size(
    train_config: DefaultConfig,
    prediction_config: PredictionConfig,
) -> VoxelSize | None:
    if prediction_config.overwrite_voxel_size is not None:
        return prediction_config.overwrite_voxel_size
    return train_config.voxel_size


def _tensor_shape_xyz(image: torch.Tensor) -> tuple[int, int, int]:
    return tuple(int(dim) for dim in reversed(image.shape[-3:]))


def _get_patch_starts(size: int, patch_size: int, overlap: float) -> list[int]:
    if patch_size >= size:
        return [0]

    last_start = size - patch_size
    max_stride = max(1.0, patch_size * (1.0 - overlap))
    num_patches = int(np.ceil(last_start / max_stride)) + 1
    return np.rint(np.linspace(0, last_start, num_patches)).astype(int).tolist()


def _iter_patch_origins(
    image_shape_xyz: tuple[int, int, int],
    patch_shape_xyz: tuple[int, int, int],
    overlap: float,
) -> Iterable[tuple[int, int, int]]:
    starts_per_axis = [
        _get_patch_starts(size, patch_size, overlap)
        for size, patch_size in zip(image_shape_xyz, patch_shape_xyz, strict=True)
    ]
    return itertools.product(*starts_per_axis)


def _extract_tensor_patch(
    image: torch.Tensor,
    patch_origin_xyz: tuple[int, int, int],
    patch_shape_xyz: tuple[int, int, int],
    fill_value: float,
) -> torch.Tensor:
    patch_shape_zyx = tuple(reversed(patch_shape_xyz))
    image_shape_zyx = tuple(int(dim) for dim in image.shape[-3:])
    patch_origin_zyx = tuple(reversed(patch_origin_xyz))
    patch = torch.full(
        (image.shape[0], *patch_shape_zyx),
        fill_value,
        dtype=image.dtype,
        device=image.device,
    )

    source_slices: list[slice] = []
    destination_slices: list[slice] = []
    for origin, patch_dim, image_dim in zip(
        patch_origin_zyx,
        patch_shape_zyx,
        image_shape_zyx,
        strict=True,
    ):
        source_start = max(origin, 0)
        source_end = min(origin + patch_dim, image_dim)
        destination_start = max(-origin, 0)
        destination_end = destination_start + max(source_end - source_start, 0)
        source_slices.append(slice(source_start, source_end))
        destination_slices.append(slice(destination_start, destination_end))

    patch[
        :,
        destination_slices[0],
        destination_slices[1],
        destination_slices[2],
    ] = image[
        :,
        source_slices[0],
        source_slices[1],
        source_slices[2],
    ]
    return patch


def _apply_patch_border_weight(
    detection: torch.Tensor,
    patch_shape_xyz: tuple[int, int, int],
    border_weight: float,
) -> torch.Tensor:
    if border_weight >= 1.0:
        return detection

    patch_extent = detection.new_tensor(
        [
            patch_shape_xyz[0],
            patch_shape_xyz[1],
            patch_shape_xyz[2],
            patch_shape_xyz[0],
            patch_shape_xyz[1],
            patch_shape_xyz[2],
        ]
    )
    border_distances = torch.minimum(detection[:6], patch_extent - detection[:6]).clamp(min=0)
    axis_norm = detection.new_tensor(
        [
            max(patch_shape_xyz[0] / 2.0, 1.0),
            max(patch_shape_xyz[1] / 2.0, 1.0),
            max(patch_shape_xyz[2] / 2.0, 1.0),
            max(patch_shape_xyz[0] / 2.0, 1.0),
            max(patch_shape_xyz[1] / 2.0, 1.0),
            max(patch_shape_xyz[2] / 2.0, 1.0),
        ]
    )
    center_score = torch.clamp(border_distances / axis_norm, min=0.0, max=1.0).amin()
    weighted_detection = detection.clone()
    weighted_detection[6] = weighted_detection[6] * (
        border_weight + (1.0 - border_weight) * center_score
    )
    return weighted_detection


def _shift_detection_to_image(
    detection: torch.Tensor,
    patch_origin_xyz: tuple[int, int, int],
) -> torch.Tensor:
    shifted_detection = detection.clone()
    shifted_detection[:6] += shifted_detection.new_tensor(
        [
            patch_origin_xyz[0],
            patch_origin_xyz[1],
            patch_origin_xyz[2],
            patch_origin_xyz[0],
            patch_origin_xyz[1],
            patch_origin_xyz[2],
        ]
    )
    return shifted_detection


def _merge_patch_detections(
    detections: list[torch.Tensor],
    *,
    iou_threshold: float,
    agnostic_nms: bool,
    max_det: int,
) -> tuple[torch.Tensor, list[int]]:
    if not detections:
        return torch.zeros((0, 8), dtype=torch.float32), []

    merged = torch.stack(detections, dim=0)
    order = merged[:, 6].argsort(descending=True)
    merged = merged[order]
    if merged.shape[0] > max_det:
        merged = merged[:max_det]
        order = order[:max_det]

    class_offsets = merged[:, 7:8] * (0 if agnostic_nms else 7680)
    keep = ops.nms(merged[:, :6] + class_offsets, merged[:, 6], iou_threshold, max_det)
    keep = keep[:max_det]
    keep_indices = order[keep].tolist()
    return merged[keep], keep_indices


def _reconstruct_patch_mask_to_image(
    detection: torch.Tensor,
    prototype_masks: torch.Tensor,
    patch_shape_zyx: tuple[int, int, int],
    image_shape_zyx: tuple[int, int, int],
    patch_origin_xyz: tuple[int, int, int],
) -> torch.Tensor | None:
    nm = detection.shape[0] - 8
    if nm <= 0:
        return None

    coefficients = detection[8 : 8 + nm].to(device=prototype_masks.device)
    patch_mask_prob = ops.reconstruct_mask_probabilities(coefficients, prototype_masks)
    patch_mask_prob = F.interpolate(
        patch_mask_prob.unsqueeze(0).unsqueeze(0),
        size=patch_shape_zyx,
        mode="trilinear",
        align_corners=False,
    )[0, 0].to(dtype=torch.float32)

    full_mask = torch.zeros(image_shape_zyx, dtype=torch.float32)
    patch_origin_zyx = tuple(reversed(patch_origin_xyz))
    source_slices: list[slice] = []
    destination_slices: list[slice] = []
    for origin, patch_dim, image_dim in zip(
        patch_origin_zyx,
        patch_shape_zyx,
        image_shape_zyx,
        strict=True,
    ):
        source_start = max(-origin, 0)
        source_end = source_start + max(min(origin + patch_dim, image_dim) - max(origin, 0), 0)
        destination_start = max(origin, 0)
        destination_end = destination_start + max(source_end - source_start, 0)
        source_slices.append(slice(source_start, source_end))
        destination_slices.append(slice(destination_start, destination_end))

    full_mask[
        destination_slices[0],
        destination_slices[1],
        destination_slices[2],
    ] = patch_mask_prob[
        source_slices[0],
        source_slices[1],
        source_slices[2],
    ]
    return full_mask


def _to_numpy_float32_contiguous(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(tensor, np.ndarray):
        return np.ascontiguousarray(tensor, dtype=np.float32)
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()


def _to_torch_float32_contiguous(array: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return torch.from_numpy(np.ascontiguousarray(array)).to(dtype=torch.float32)


def _save_merged_mask_probability(
    mask_probability: torch.Tensor | np.ndarray,
    original_shape: tuple[int, int, int],
    mask_affine: np.ndarray,
    mask_dir: Path,
    det_index: int,
    case_name: str,
) -> str:
    mask_probability_tensor = _to_torch_float32_contiguous(mask_probability)
    mask_up = F.interpolate(
        mask_probability_tensor.unsqueeze(0).unsqueeze(0),
        size=original_shape,
        mode="trilinear",
        align_corners=False,
    )[0, 0].permute(2, 1, 0)
    mask_bin = (mask_up >= 0.5).cpu().numpy().astype("uint8")
    mask_img = nib.Nifti1Image(mask_bin, affine=mask_affine)
    mask_dir.mkdir(exist_ok=True)
    mask_filename = f"{case_name}_det{det_index}.nii.gz"
    mask_path = mask_dir / mask_filename
    nib.save(mask_img, str(mask_path))
    return mask_filename


def _postprocess_single(
    model_output: ModelOutput,
    train_config: DefaultConfig,
    prediction_config: PredictionConfig,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return NMS-filtered detections with single (argmax) class per box.

    Output tensor columns: x1,y1,z1,x2,y2,z2,conf,class,(optional mask coeffs...)
    """
    detections = ops.postprocess_model_output(
        model_output,
        DetectionPostprocessConfig(
            confidence_threshold=prediction_config.confidence_threshold,
            nms_iou_threshold=prediction_config.nms_iou_threshold,
            agnostic_nms=train_config.agnostic_nms,
            max_det=train_config.max_det,
            nc=num_classes,
            multi_label=False,
        ),
    )[0].cpu()
    return detections, model_output.prototype_masks


def _relative_box(
    box_image_xyzxyz: torch.Tensor, original_shape: tuple[int, int, int]
) -> list[float]:
    rel_box = box_image_xyzxyz.clone()
    d, h, w = original_shape  # preprocess_image returns spatial shape in D, H, W (Z, Y, X)
    rel_box[0] /= w
    rel_box[1] /= h
    rel_box[2] /= d
    rel_box[3] /= w
    rel_box[4] /= h
    rel_box[5] /= d
    return [round(float(x), 6) for x in rel_box.tolist()]


def _reconstruct_and_save_mask(
    det: torch.Tensor | np.ndarray,
    proto: torch.Tensor | np.ndarray,
    original_shape: tuple[int, int, int],
    mask_affine: np.ndarray,
    mask_dir: Path,
    det_index: int,
    case_name: str,
) -> str | None:
    det_tensor = _to_torch_float32_contiguous(det)
    proto_tensor = _to_torch_float32_contiguous(proto)
    nm = det_tensor.shape[0] - 8
    if nm <= 0:
        return None
    coeff = det_tensor[8 : 8 + nm]
    mask_prob = ops.reconstruct_mask_probabilities(coeff, proto_tensor)
    mask_prob_r = mask_prob.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    mask_up = F.interpolate(
        mask_prob_r, size=original_shape, mode="trilinear", align_corners=False
    )[0, 0].permute(2, 1, 0)  # (W, H, D)
    mask_bin = (mask_up >= 0.5).cpu().numpy().astype("uint8")
    mask_img = nib.Nifti1Image(mask_bin, affine=mask_affine)
    mask_dir.mkdir(exist_ok=True)
    mask_filename = f"{case_name}_det{det_index}.nii.gz"
    mask_path = mask_dir / mask_filename
    nib.save(mask_img, str(mask_path))
    return mask_filename


def _build_prediction_entries(
    detections: torch.Tensor,
    *,
    case_name: str,
    affine: np.ndarray,
    original_shape: tuple[int, int, int],
    names: dict[int, str],
    include_masks: bool,
) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    image_results: list[dict[str, object]] = []
    mask_detections: list[np.ndarray] = []
    for det_i, det in enumerate(detections):
        box_image_xyzxyz = det[:6]
        try:
            box_world_xyzxyz = ops.box_image_to_world(
                box_image_xyzxyz, np.asarray(affine, dtype=np.float32)
            )
        except (RuntimeError, ValueError, TypeError):  # pragma: no cover
            box_world_xyzxyz = box_image_xyzxyz
        entry: dict[str, object] = {
            "class": int(det[7].item()),
            "class_name": names.get(int(det[7].item()), str(int(det[7].item()))),
            "confidence": round(float(det[6].item()), 6),
            "box_image_xyzxyz": [round(float(x), 2) for x in box_image_xyzxyz.tolist()],
            "box_world_xyzxyz": [round(float(x), 2) for x in box_world_xyzxyz.tolist()],
            "box_relative_xyzxyz": _relative_box(box_image_xyzxyz, original_shape),
        }
        if include_masks:
            entry["mask_path"] = f"{case_name}_det{det_i}.nii.gz"
            mask_detections.append(_to_numpy_float32_contiguous(det))
        image_results.append(entry)
    return image_results, mask_detections


def _predict_whole_image(
    loaded_image: _LoadedImageTask,
    *,
    model: HardcodedModel,
    train_config: DefaultConfig,
    names: dict[int, str],
    prediction_config: PredictionConfig,
) -> _WriteTask:
    img_tensor = torch.from_numpy(np.ascontiguousarray(loaded_image.image)).permute(
        *reversed(range(loaded_image.image.ndim))
    )
    normalization = _resolve_normalization(train_config, prediction_config)
    patch_size = _resolve_patch_size(train_config, prediction_config)
    input_voxel_size = _resolve_input_voxel_size(loaded_image)
    target_voxel_size = _resolve_target_voxel_size(train_config, prediction_config)
    target_shape = patch_size if target_voxel_size is None else None

    device = model.device
    img_tensor = img_tensor.to(device)
    img_tensor, original_shape, inference_shape = preprocess_image(
        img_tensor,
        normalization,
        target_shape=target_shape,
        input_voxel_size=input_voxel_size,
        target_voxel_size=target_voxel_size,
        apply_resampling=True,
        apply_normalization=True,
        output_dtype=torch.float32,
    )

    autocast_ctx = (
        torch.autocast(device.type) if device.type != "cpu" else contextlib.nullcontext()
    )
    with torch.no_grad(), autocast_ctx:
        model_output: ModelOutput = model(img_tensor.unsqueeze(0))
    detections, proto_masks = _postprocess_single(
        model_output,
        train_config,
        prediction_config,
        len(names),
    )
    prototype_masks: np.ndarray | None = None
    mask_detections: list[np.ndarray] = []
    image_results: list[dict[str, object]] = []
    if detections.numel():
        ops.scale_boxes(
            inference_shape,
            detections[:, :6],
            original_shape,
            ratio_pad=None,
            clip=prediction_config.clip_boxes_to_image,
        )
        image_results, mask_detections = _build_prediction_entries(
            detections,
            case_name=loaded_image.case_name,
            affine=loaded_image.affine,
            original_shape=original_shape,
            names=names,
            include_masks=model_output.mask_coefficients is not None and proto_masks is not None,
        )
        if mask_detections and proto_masks is not None:
            prototype_masks = _to_numpy_float32_contiguous(proto_masks[0])
    return _WriteTask(
        case_name=loaded_image.case_name,
        image_results=image_results,
        affine=loaded_image.affine,
        original_shape=original_shape,
        prototype_masks=prototype_masks,
        mask_detections=mask_detections,
    )


def _predict_patchwise_image(
    loaded_image: _LoadedImageTask,
    *,
    model: HardcodedModel,
    train_config: DefaultConfig,
    names: dict[int, str],
    prediction_config: PredictionConfig,
) -> _WriteTask:
    img_tensor = torch.from_numpy(np.ascontiguousarray(loaded_image.image)).permute(
        *reversed(range(loaded_image.image.ndim))
    )
    normalization = _resolve_normalization(train_config, prediction_config)
    patch_size = _resolve_patch_size(train_config, prediction_config)
    patch_shape_xyz = patch_size.get_xyz()
    patch_shape_zyx = patch_size.get_zyx()
    input_voxel_size = _resolve_input_voxel_size(loaded_image)
    target_voxel_size = _resolve_target_voxel_size(train_config, prediction_config)

    device = model.device
    if prediction_config.preprocessing_on_gpu:
        img_tensor = img_tensor.to(device)
    img_tensor, original_shape, inference_shape = preprocess_image(
        img_tensor,
        normalization,
        input_voxel_size=input_voxel_size,
        target_voxel_size=target_voxel_size,
        apply_resampling=True,
        apply_normalization=True,
        output_dtype=torch.float32,
    )
    inference_shape_xyz = _tensor_shape_xyz(img_tensor)
    patch_fill_value = float(img_tensor.min().item())

    autocast_ctx = (
        torch.autocast(device.type) if device.type != "cpu" else contextlib.nullcontext()
    )
    merged_detections: list[torch.Tensor] = []
    merged_masks: list[np.ndarray] = []
    for patch_origin_xyz in _iter_patch_origins(
        inference_shape_xyz,
        patch_shape_xyz,
        prediction_config.patch_overlap,
    ):
        patch = _extract_tensor_patch(
            img_tensor,
            patch_origin_xyz,
            patch_shape_xyz,
            patch_fill_value,
        )
        if not prediction_config.preprocessing_on_gpu:
            patch = patch.to(device)
        with torch.no_grad(), autocast_ctx:
            model_output: ModelOutput = model(patch.unsqueeze(0))
        detections, proto_masks = _postprocess_single(
            model_output,
            train_config,
            prediction_config,
            len(names),
        )
        if not detections.numel():
            continue

        prototype_cpu = (
            proto_masks[0].detach().to(device="cpu", dtype=torch.float32).contiguous()
            if proto_masks is not None
            else None
        )
        for detection in detections:
            weighted_detection = _apply_patch_border_weight(
                detection,
                patch_shape_xyz,
                prediction_config.patch_border_weight,
            )
            merged_detections.append(
                _shift_detection_to_image(weighted_detection, patch_origin_xyz)
            )
            if model_output.mask_coefficients is not None and prototype_cpu is not None:
                full_mask = _reconstruct_patch_mask_to_image(
                    detection.detach().to(dtype=torch.float32).clone(),
                    prototype_cpu,
                    patch_shape_zyx,
                    inference_shape,
                    patch_origin_xyz,
                )
                if full_mask is not None:
                    merged_masks.append(_to_numpy_float32_contiguous(full_mask))

    merged_prediction_tensor, keep_indices = _merge_patch_detections(
        merged_detections,
        iou_threshold=prediction_config.nms_iou_threshold,
        agnostic_nms=train_config.agnostic_nms,
        max_det=train_config.max_det,
    )
    if merged_prediction_tensor.numel() == 0:
        return _WriteTask(
            case_name=loaded_image.case_name,
            image_results=[],
            affine=loaded_image.affine,
            original_shape=original_shape,
            prototype_masks=None,
            mask_detections=[],
        )

    if prediction_config.clip_boxes_to_image:
        ops.clip_boxes(merged_prediction_tensor[:, :6], inference_shape)
    selected_masks = [merged_masks[index] for index in keep_indices] if merged_masks else []
    ops.scale_boxes(
        inference_shape,
        merged_prediction_tensor[:, :6],
        original_shape,
        ratio_pad=None,
        clip=prediction_config.clip_boxes_to_image,
    )
    image_results, _mask_detections = _build_prediction_entries(
        merged_prediction_tensor,
        case_name=loaded_image.case_name,
        affine=loaded_image.affine,
        original_shape=original_shape,
        names=names,
        include_masks=bool(selected_masks),
    )
    return _WriteTask(
        case_name=loaded_image.case_name,
        image_results=image_results,
        affine=loaded_image.affine,
        original_shape=original_shape,
        prototype_masks=None,
        mask_detections=[],
        merged_mask_probabilities=selected_masks,
    )


def _predict_loaded_image(
    loaded_image: _LoadedImageTask,
    *,
    model: HardcodedModel,
    train_config: DefaultConfig,
    names: dict[int, str],
    prediction_config: PredictionConfig,
) -> _WriteTask:
    if train_config.training_mode == "patchwise":
        return _predict_patchwise_image(
            loaded_image,
            model=model,
            train_config=train_config,
            names=names,
            prediction_config=prediction_config,
        )
    return _predict_whole_image(
        loaded_image,
        model=model,
        train_config=train_config,
        names=names,
        prediction_config=prediction_config,
    )


def _write_prediction_result(output_dir: Path, write_task: _WriteTask) -> None:
    if write_task.prototype_masks is not None or write_task.merged_mask_probabilities:
        mask_affine = np.asarray(write_task.affine, dtype=np.float32)
    if write_task.prototype_masks is not None:
        for det_i, det_tensor in enumerate(write_task.mask_detections):
            _reconstruct_and_save_mask(
                det_tensor,
                write_task.prototype_masks,
                write_task.original_shape,
                mask_affine,
                output_dir / "masks",
                det_i,
                write_task.case_name,
            )
    if write_task.merged_mask_probabilities:
        for det_i, mask_probability in enumerate(write_task.merged_mask_probabilities):
            _save_merged_mask_probability(
                mask_probability,
                write_task.original_shape,
                mask_affine,
                output_dir / "masks",
                det_i,
                write_task.case_name,
            )

    out_json = output_dir / f"{write_task.case_name}.json"
    with out_json.open("w", encoding="utf-8") as jf:
        json.dump(write_task.image_results, jf, indent=2)


def _fail_worker(
    error_queue: mp.queues.Queue[_WorkerFailure],
    *,
    stage: str,
    case_name: str | None,
) -> None:
    error_queue.put(
        _WorkerFailure(
            stage=stage,
            case_name=case_name,
            error="Worker crashed",
            traceback_text=traceback.format_exc(),
        )
    )


def _loader_worker(
    load_queue: mp.queues.Queue[Path | None],
    inference_queue: mp.queues.Queue[_LoadedImageTask | None],
    error_queue: mp.queues.Queue[_WorkerFailure],
) -> None:
    current_case: str | None = None
    try:
        while True:
            img_path = load_queue.get()
            if img_path is _SENTINEL:
                inference_queue.put(_SENTINEL)
                return
            current_case = _case_name_from_path(img_path)
            nifti_img = nib.load(str(img_path))
            img_np = np.asarray(nifti_img.dataobj, dtype=np.float32)
            inference_queue.put(
                _LoadedImageTask(
                    case_name=current_case,
                    affine=np.asarray(nifti_img.affine, dtype=np.float32),
                    image=np.ascontiguousarray(img_np),
                    voxel_size_xyz=tuple(float(zoom) for zoom in nifti_img.header.get_zooms()[:3]),
                )
            )
    except Exception:  # pragma: no cover - exercised via process supervision
        _fail_worker(error_queue, stage="loader", case_name=current_case)
        raise


def _inference_worker(
    prediction_cfg: PredictionConfig,
    inference_queue: mp.queues.Queue[_LoadedImageTask | None],
    write_queue: mp.queues.Queue[_WriteTask | None],
    error_queue: mp.queues.Queue[_WorkerFailure],
) -> None:
    current_case: str | None = None
    try:
        model, train_config, names = _load_model(prediction_cfg)
        _validate_loaded_model_prediction_config(train_config, prediction_cfg)
        finished_loaders = 0
        while finished_loaders < prediction_cfg.num_load_workers:
            loaded_image = inference_queue.get()
            if loaded_image is _SENTINEL:
                finished_loaders += 1
                continue
            current_case = loaded_image.case_name
            write_queue.put(
                _predict_loaded_image(
                loaded_image,
                model=model,
                train_config=train_config,
                names=names,
                prediction_config=prediction_cfg,
                )
            )
            current_case = None
        for _ in range(prediction_cfg.num_write_workers):
            write_queue.put(_SENTINEL)
    except Exception:  # pragma: no cover - exercised via process supervision
        _fail_worker(error_queue, stage="inference", case_name=current_case)
        raise


def _writer_worker(
    output_dir: Path,
    write_queue: mp.queues.Queue[_WriteTask | None],
    error_queue: mp.queues.Queue[_WorkerFailure],
) -> None:
    current_case: str | None = None
    try:
        while True:
            write_task = write_queue.get()
            if write_task is _SENTINEL:
                return
            current_case = write_task.case_name
            _write_prediction_result(output_dir, write_task)
            current_case = None
    except Exception:  # pragma: no cover - exercised via process supervision
        _fail_worker(error_queue, stage="writer", case_name=current_case)
        raise
def _terminate_processes(processes: list[mp.Process]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)


def _log_worker_failure(failure: _WorkerFailure) -> None:
    LOGGER.error(
        "Prediction %s worker failed%s: %s\n%s",
        failure.stage,
        f" for {failure.case_name}" if failure.case_name is not None else "",
        failure.error,
        failure.traceback_text,
    )


def _get_worker_failure(
    error_queue: mp.queues.Queue[_WorkerFailure],
) -> _WorkerFailure | None:
    try:
        return error_queue.get_nowait()
    except queue.Empty:
        return None


def _get_failed_processes(processes: list[mp.Process]) -> list[mp.Process]:
    return [
        process
        for process in processes
        if process.exitcode is not None and process.exitcode != 0
    ]


def _all_processes_succeeded(processes: list[mp.Process]) -> bool:
    return all(process.exitcode == 0 for process in processes)


def _log_failed_processes(processes: list[mp.Process]) -> None:
    for process in processes:
        LOGGER.error(
            "Prediction worker %s exited with code %s",
            process.name,
            process.exitcode,
        )


def _validate_prediction_config(prediction_cfg: PredictionConfig) -> None:
    if not 0.0 <= prediction_cfg.patch_overlap < 1.0:
        msg = "patch_overlap must be in the range [0, 1)."
        raise ValueError(msg)
    if not 0.0 < prediction_cfg.patch_border_weight <= 1.0:
        msg = "patch_border_weight must be in the range (0, 1]."
        raise ValueError(msg)
    if prediction_cfg.split_group not in {"train", "val", "test"}:
        msg = "split_group must be one of 'train', 'val' or 'test'."
        raise ValueError(msg)


def _validate_loaded_model_prediction_config(
    train_config: DefaultConfig,
    prediction_cfg: PredictionConfig,
) -> None:
    if (
        train_config.training_mode != "patchwise"
        and prediction_cfg.overwrite_voxel_size is not None
        and prediction_cfg.overwrite_patch_size is not None
    ):
        msg = (
            "Whole-image inference accepts either --overwrite-patch-size or "
            "--overwrite-voxel-size, "
            "but not both at the same time."
        )
        raise ValueError(msg)


def _update_progress_from_load_queue(
    load_queue: mp.queues.Queue[Path | None],
    progress_bar: tqdm,
    total_files: int,
) -> None:
    try:
        remaining = load_queue.qsize()
    except (AttributeError, NotImplementedError):
        return
    completed = max(0, min(total_files - remaining, total_files))
    if completed > progress_bar.n:
        progress_bar.update(completed - progress_bar.n)


def _monitor_prediction_processes(
    *,
    prediction_cfg: PredictionConfig,
    load_queue: mp.queues.Queue[Path | None],
    error_queue: mp.queues.Queue[_WorkerFailure],
    processes: list[mp.Process],
    progress_bar: tqdm,
    total_files: int,
) -> int:
    sent_sentinels = False
    while True:
        _update_progress_from_load_queue(load_queue, progress_bar, total_files)
        if not sent_sentinels and progress_bar.n >= total_files:
            for _ in range(prediction_cfg.num_load_workers):
                load_queue.put(_SENTINEL)
            sent_sentinels = True

        failure = _get_worker_failure(error_queue)
        if failure is not None:
            _log_worker_failure(failure)
            _terminate_processes(processes)
            return 1

        failed_processes = _get_failed_processes(processes)
        if failed_processes:
            _log_failed_processes(failed_processes)
            _terminate_processes(processes)
            return 1

        if _all_processes_succeeded(processes):
            _update_progress_from_load_queue(load_queue, progress_bar, total_files)
            return 0

        for process in processes:
            process.join(timeout=0.1)


def _run_prediction_pipeline(prediction_cfg: PredictionConfig, files: list[Path]) -> int:
    if prediction_cfg.num_load_workers < 1:
        msg = "num_load_workers must be at least 1"
        raise ValueError(msg)
    if prediction_cfg.num_write_workers < 1:
        msg = "num_write_workers must be at least 1"
        raise ValueError(msg)

    ctx = mp.get_context("spawn")
    load_queue: mp.queues.Queue[Path | None] = ctx.Queue()
    inference_queue: mp.queues.Queue[_LoadedImageTask | None] = ctx.Queue(
        maxsize=max(prediction_cfg.num_load_workers, 2)
    )
    write_queue: mp.queues.Queue[_WriteTask | None] = ctx.Queue(
        maxsize=max(prediction_cfg.num_write_workers, 2)
    )
    error_queue: mp.queues.Queue[_WorkerFailure] = ctx.Queue()
    loaders = [
        ctx.Process(
            target=_loader_worker,
            args=(load_queue, inference_queue, error_queue),
            name=f"predict-loader-{index}",
        )
        for index in range(prediction_cfg.num_load_workers)
    ]
    inference_process = ctx.Process(
        target=_inference_worker,
        args=(prediction_cfg, inference_queue, write_queue, error_queue),
        name="predict-inference",
    )
    writers = [
        ctx.Process(
            target=_writer_worker,
            args=(prediction_cfg.output_dir, write_queue, error_queue),
            name=f"predict-writer-{index}",
        )
        for index in range(prediction_cfg.num_write_workers)
    ]
    processes = [*loaders, inference_process, *writers]

    try:
        with tqdm(total=len(files), desc="Predicting", unit="case", smoothing=0.05) as pbar:
            for process in processes:
                process.start()
            for file_path in files:
                load_queue.put(file_path)
            return _monitor_prediction_processes(
                prediction_cfg=prediction_cfg,
                load_queue=load_queue,
                error_queue=error_queue,
                processes=processes,
                progress_bar=pbar,
                total_files=len(files),
            )
    finally:
        for queue_ in (
            load_queue,
            inference_queue,
            write_queue,
            error_queue,
        ):
            queue_.close()
            queue_.join_thread()


def run_prediction(prediction_cfg: PredictionConfig) -> int:
    try:
        _validate_prediction_config(prediction_cfg)
        if not prediction_cfg.output_dir.exists():
            prediction_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        files = list(_iter_nifti(prediction_cfg.input_dir))
        if prediction_cfg.split_file is not None:
            selected_case_names = _load_case_names_from_split_file(
                prediction_cfg.split_file,
                fold=prediction_cfg.split_fold,
                group=prediction_cfg.split_group,
            )
            files = [
                file_path
                for file_path in files
                if _case_name_from_path(file_path) in selected_case_names
            ]
        if not files:
            if prediction_cfg.split_file is None:
                LOGGER.error("No .nii.gz files found in %s", prediction_cfg.input_dir)
            else:
                LOGGER.error(
                    "No .nii.gz files in %s matched split selection %s (fold=%d, group=%s)",
                    prediction_cfg.input_dir,
                    prediction_cfg.split_file,
                    prediction_cfg.split_fold,
                    prediction_cfg.split_group,
                )
            return 1
        result = _run_prediction_pipeline(prediction_cfg, files)
        if result != 0:
            return result
        LOGGER.info("Saved predictions for %d volumes to %s", len(files), prediction_cfg.output_dir)
    except Exception as e:  # pragma: no cover - defensive
        LOGGER.exception("Prediction failed", exc_info=e)
        return 1
    else:
        return 0
