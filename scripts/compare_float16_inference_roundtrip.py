from __future__ import annotations

import argparse
import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import yaml

from radyolo.config.spatial import PatchSize, VoxelSize
from radyolo.data.preprocessing import preprocess_image
from radyolo.inference.predict import (
    PredictionConfig,
    _apply_patch_border_weight,
    _extract_tensor_patch,
    _iter_patch_origins,
    _merge_patch_detections,
    _postprocess_single,
    _resolve_normalization,
    _resolve_patch_size,
    _resolve_target_voxel_size,
    _shift_detection_to_image,
    _tensor_shape_xyz,
)
from radyolo.models.model import HardcodedModel, LoadedCheckpointModel, load_checkpoint_model
from radyolo.utils import ops


@dataclass(frozen=True)
class LoadedImage:
    image: np.ndarray
    voxel_size_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class InferenceResult:
    model_input: torch.Tensor
    detections: torch.Tensor
    original_shape: tuple[int, int, int]
    inference_shape: tuple[int, int, int]


def _parse_normalization_option(value: str | None) -> dict | list[dict] | None:
    if value is None:
        return None
    parsed = yaml.safe_load(value)
    if not isinstance(parsed, (dict, list)):
        msg = "Normalization override must parse to a dict or list of dicts."
        raise TypeError(msg)
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one inference run against the same run after float16->float32 "
            "round-tripping the model input tensor after preprocessing."
        )
    )
    parser.add_argument("image_path", type=Path, help="Input .nii.gz image path.")
    parser.add_argument("model_path", type=Path, help="Trained model checkpoint path.")
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device override, e.g. cpu or 0. Defaults to checkpoint config.",
    )
    parser.add_argument(
        "--overwrite-patch-size",
        nargs=3,
        type=int,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Override checkpoint patch size for inference as x y z.",
    )
    parser.add_argument(
        "--overwrite-voxel-size",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Override checkpoint voxel size for inference as x y z.",
    )
    parser.add_argument(
        "--overwrite-normalization",
        default=None,
        help="YAML or JSON normalization override.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold. Defaults to the checkpoint validation threshold.",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=None,
        help="NMS IoU threshold. Defaults to the checkpoint validation threshold.",
    )
    parser.add_argument(
        "--patch-overlap",
        type=float,
        default=0.5,
        help="Sliding-window overlap fraction for patchwise inference.",
    )
    parser.add_argument(
        "--patch-border-weight",
        type=float,
        default=0.9,
        help="Patch border confidence multiplier for patchwise inference.",
    )
    parser.add_argument(
        "--preprocessing-on-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match predict behavior for patchwise preprocessing and patch extraction.",
    )
    parser.add_argument(
        "--clip-boxes-to-image",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether predicted boxes are clipped to image bounds.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full detection tensors for both runs.",
    )
    return parser.parse_args()


def _load_image(image_path: Path) -> LoadedImage:
    nifti_image = nib.load(str(image_path))
    image = np.asarray(nifti_image.dataobj, dtype=np.float32)
    voxel_size_xyz = tuple(float(zoom) for zoom in nifti_image.header.get_zooms()[:3])
    return LoadedImage(image=np.ascontiguousarray(image), voxel_size_xyz=voxel_size_xyz)


def _build_prediction_config(
    args: argparse.Namespace,
    loaded_model: LoadedCheckpointModel,
) -> PredictionConfig:
    train_config = loaded_model.train_config
    return PredictionConfig(
        model_path=args.model_path,
        input_dir=args.image_path.parent,
        output_dir=Path.cwd(),
        confidence_threshold=(
            args.conf
            if args.conf is not None
            else train_config.confidence_threshold_validation
        ),
        nms_iou_threshold=(
            args.nms_iou if args.nms_iou is not None else train_config.nms_iou_threshold
        ),
        device=args.device or str(loaded_model.device),
        clip_boxes_to_image=(
            args.clip_boxes_to_image
            if args.clip_boxes_to_image is not None
            else train_config.clip_boxes_to_image
        ),
        overwrite_patch_size=(
            PatchSize.from_xyz(args.overwrite_patch_size)
            if args.overwrite_patch_size is not None
            else None
        ),
        overwrite_normalization=_parse_normalization_option(args.overwrite_normalization),
        overwrite_voxel_size=(
            VoxelSize.from_xyz(args.overwrite_voxel_size)
            if args.overwrite_voxel_size is not None
            else None
        ),
        patch_overlap=args.patch_overlap,
        patch_border_weight=args.patch_border_weight,
        preprocessing_on_gpu=args.preprocessing_on_gpu,
        num_load_workers=1,
        num_write_workers=1,
    )


def _preprocess_model_input(
    loaded_image: LoadedImage,
    loaded_model: LoadedCheckpointModel,
    prediction_config: PredictionConfig,
) -> tuple[torch.Tensor, tuple[int, int, int], tuple[int, int, int]]:
    train_config = loaded_model.train_config
    image_tensor = torch.from_numpy(np.ascontiguousarray(loaded_image.image)).permute(
        *reversed(range(loaded_image.image.ndim))
    )
    normalization = _resolve_normalization(train_config, prediction_config)
    target_voxel_size = _resolve_target_voxel_size(train_config, prediction_config)
    if train_config.training_mode == "patchwise":
        target_shape = None
    else:
        patch_size = _resolve_patch_size(train_config, prediction_config)
        target_shape = patch_size if target_voxel_size is None else None

    device = loaded_model.device
    if train_config.training_mode != "patchwise" or prediction_config.preprocessing_on_gpu:
        image_tensor = image_tensor.to(device)

    image_tensor, original_shape, inference_shape = preprocess_image(
        image_tensor,
        normalization,
        target_shape=target_shape,
        input_voxel_size=VoxelSize.from_xyz(loaded_image.voxel_size_xyz),
        target_voxel_size=target_voxel_size,
        apply_resampling=True,
        apply_normalization=True,
        output_dtype=torch.float32,
    )
    return image_tensor, original_shape, inference_shape


def _run_whole_image_inference(
    model_input: torch.Tensor,
    loaded_model: LoadedCheckpointModel,
    prediction_config: PredictionConfig,
    original_shape: tuple[int, int, int],
    inference_shape: tuple[int, int, int],
) -> torch.Tensor:
    model = loaded_model.model
    autocast_ctx = (
        torch.autocast(model.device.type)
        if model.device.type != "cpu"
        else contextlib.nullcontext()
    )
    with torch.no_grad(), autocast_ctx:
        model_output = model(model_input.unsqueeze(0))
    detections, _prototype_masks = _postprocess_single(
        model_output,
        loaded_model.train_config,
        prediction_config,
        loaded_model.train_config.num_classes,
    )
    if detections.numel():
        ops.scale_boxes(
            inference_shape,
            detections[:, :6],
            original_shape,
            ratio_pad=None,
            clip=prediction_config.clip_boxes_to_image,
        )
    return detections.cpu()


def _run_patchwise_inference(
    model_input: torch.Tensor,
    loaded_model: LoadedCheckpointModel,
    prediction_config: PredictionConfig,
    original_shape: tuple[int, int, int],
    inference_shape: tuple[int, int, int],
) -> torch.Tensor:
    model = loaded_model.model
    patch_size = _resolve_patch_size(loaded_model.train_config, prediction_config)
    patch_shape_xyz = patch_size.get_xyz()
    inference_shape_xyz = _tensor_shape_xyz(model_input)
    patch_fill_value = float(model_input.min().item())

    autocast_ctx = (
        torch.autocast(model.device.type)
        if model.device.type != "cpu"
        else contextlib.nullcontext()
    )
    merged_detections: list[torch.Tensor] = []
    for patch_origin_xyz in _iter_patch_origins(
        inference_shape_xyz,
        patch_shape_xyz,
        prediction_config.patch_overlap,
    ):
        patch = _extract_tensor_patch(
            model_input,
            patch_origin_xyz,
            patch_shape_xyz,
            patch_fill_value,
        )
        if not prediction_config.preprocessing_on_gpu:
            patch = patch.to(model.device)
        with torch.no_grad(), autocast_ctx:
            model_output = model(patch.unsqueeze(0))
        detections, _prototype_masks = _postprocess_single(
            model_output,
            loaded_model.train_config,
            prediction_config,
            loaded_model.train_config.num_classes,
        )
        if not detections.numel():
            continue
        for detection in detections:
            weighted_detection = _apply_patch_border_weight(
                detection,
                patch_shape_xyz,
                prediction_config.patch_border_weight,
            )
            merged_detections.append(
                _shift_detection_to_image(weighted_detection, patch_origin_xyz)
            )

    merged_prediction_tensor, _keep_indices = _merge_patch_detections(
        merged_detections,
        iou_threshold=prediction_config.nms_iou_threshold,
        agnostic_nms=loaded_model.train_config.agnostic_nms,
        max_det=loaded_model.train_config.max_det,
    )
    if merged_prediction_tensor.numel() == 0:
        return merged_prediction_tensor.cpu()

    if prediction_config.clip_boxes_to_image:
        ops.clip_boxes(merged_prediction_tensor[:, :6], inference_shape)
    ops.scale_boxes(
        inference_shape,
        merged_prediction_tensor[:, :6],
        original_shape,
        ratio_pad=None,
        clip=prediction_config.clip_boxes_to_image,
    )
    return merged_prediction_tensor.cpu()


def _run_single_inference(
    loaded_image: LoadedImage,
    loaded_model: LoadedCheckpointModel,
    prediction_config: PredictionConfig,
    *,
    apply_float16_roundtrip: bool,
) -> InferenceResult:
    model_input, original_shape, inference_shape = _preprocess_model_input(
        loaded_image,
        loaded_model,
        prediction_config,
    )
    if apply_float16_roundtrip:
        model_input = model_input.to(dtype=torch.float16).to(dtype=torch.float32)

    if loaded_model.train_config.training_mode == "patchwise":
        detections = _run_patchwise_inference(
            model_input,
            loaded_model,
            prediction_config,
            original_shape,
            inference_shape,
        )
    else:
        detections = _run_whole_image_inference(
            model_input,
            loaded_model,
            prediction_config,
            original_shape,
            inference_shape,
        )
    return InferenceResult(
        model_input=model_input.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        detections=detections.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        original_shape=original_shape,
        inference_shape=inference_shape,
    )


def _compute_difference_stats(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
) -> dict[str, float]:
    if lhs.shape != rhs.shape:
        msg = f"Tensor shapes must match, got {tuple(lhs.shape)} vs {tuple(rhs.shape)}"
        raise ValueError(msg)
    diff = rhs - lhs
    abs_diff = diff.abs()
    return {
        "mean_difference": float(diff.mean().item()),
        "mean_absolute_difference": float(abs_diff.mean().item()),
        "total_difference": float(diff.sum().item()),
        "total_absolute_difference": float(abs_diff.sum().item()),
        "max_absolute_difference": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
    }


def _detection_summary(detections: torch.Tensor) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for detection in detections:
        summary.append(
            {
                "class": int(detection[7].item()),
                "confidence": float(detection[6].item()),
                "box_xyzxyz": [float(value) for value in detection[:6].tolist()],
            }
        )
    return summary


def _format_stats(name: str, stats: dict[str, float]) -> str:
    lines = [name]
    for key, value in stats.items():
        lines.append(f"  {key}: {value:.10g}")
    return "\n".join(lines)


def _sort_detections_for_comparison(detections: torch.Tensor) -> torch.Tensor:
    if detections.numel() == 0:
        return detections
    order = torch.argsort(detections[:, 6], descending=True)
    return detections[order]


def main() -> int:
    args = _parse_args()
    loaded_model = load_checkpoint_model(args.model_path, device_override=args.device)
    prediction_config = _build_prediction_config(args, loaded_model)
    loaded_image = _load_image(args.image_path)

    normal_result = _run_single_inference(
        loaded_image,
        loaded_model,
        prediction_config,
        apply_float16_roundtrip=False,
    )
    roundtrip_result = _run_single_inference(
        loaded_image,
        loaded_model,
        prediction_config,
        apply_float16_roundtrip=True,
    )

    input_stats = _compute_difference_stats(
        normal_result.model_input,
        roundtrip_result.model_input,
    )

    print(f"model_training_mode: {loaded_model.train_config.training_mode}")
    print(f"original_shape_zyx: {normal_result.original_shape}")
    print(f"inference_shape_zyx: {normal_result.inference_shape}")
    print(f"normal_detection_count: {int(normal_result.detections.shape[0])}")
    print(f"float16_roundtrip_detection_count: {int(roundtrip_result.detections.shape[0])}")
    print(_format_stats("model_input_difference_stats", input_stats))

    normal_sorted = _sort_detections_for_comparison(normal_result.detections)
    roundtrip_sorted = _sort_detections_for_comparison(roundtrip_result.detections)
    common_count = min(normal_sorted.shape[0], roundtrip_sorted.shape[0])
    if common_count > 0:
        common_normal = normal_sorted[:common_count, :8]
        common_roundtrip = roundtrip_sorted[:common_count, :8]
        prediction_stats = _compute_difference_stats(common_normal, common_roundtrip)
        print(_format_stats("prediction_difference_stats_first_8_columns", prediction_stats))
        class_mismatch_count = int(
            (common_normal[:, 7].to(dtype=torch.int64) != common_roundtrip[:, 7].to(dtype=torch.int64))
            .sum()
            .item()
        )
        print(f"class_mismatch_count_in_common_detections: {class_mismatch_count}")
    else:
        print("prediction_difference_stats_first_8_columns")
        print("  no common detections to compare")

    if normal_sorted.shape == roundtrip_sorted.shape:
        full_prediction_stats = _compute_difference_stats(normal_sorted, roundtrip_sorted)
        print(_format_stats("prediction_difference_stats_all_columns", full_prediction_stats))
    else:
        print("prediction_difference_stats_all_columns")
        print(
            "  skipped because detection tensor shapes differ: "
            f"{tuple(normal_sorted.shape)} vs {tuple(roundtrip_sorted.shape)}"
        )

    if args.verbose:
        print("normal_detections_json=")
        print(json.dumps(_detection_summary(normal_sorted), indent=2))
        print("float16_roundtrip_detections_json=")
        print(json.dumps(_detection_summary(roundtrip_sorted), indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())