from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml

from radyolo.config.config import DatasetConfig, DefaultConfig
from radyolo.config.spatial import PatchSize
from radyolo.data.analyze_dataset import analyze_dataset as analyze_dataset_fn
from radyolo.models.model import build_model_from_config

LOGGER = logging.getLogger(__name__)

ModelScale = Literal["n", "s", "m", "l", "x"]
NormalizationTag = Literal["CT", "MRGlobal", "MRPerChannel"]

_DEFAULT_REFERENCE_CONFIG = (
    Path(__file__).resolve().parents[2] / "config" / "config.yaml"
)
_PATCH_STRIDE = 32
_MIN_PATCH_SIZE = np.array([32, 32, 32], dtype=np.int64)
_CANONICAL_NORMALIZATION_TAGS: dict[str, NormalizationTag] = {
    "ct": "CT",
    "mrglobal": "MRGlobal",
    "mr_global": "MRGlobal",
    "mrperchannel": "MRPerChannel",
    "mr_per_channel": "MRPerChannel",
    "mr": "MRPerChannel",
}


@dataclass(frozen=True)
class ProposedModelConfigResult:
    output_path: Path
    fingerprint_path: Path
    analysis_ran: bool
    estimated_memory_gb: float


@dataclass(frozen=True)
class PatchMemoryBreakdown:
    feature_map_bytes: int
    feature_gradient_bytes: int
    parameter_bytes: int
    parameter_gradient_bytes: int
    optimizer_state_bytes: int
    ema_state_bytes: int
    input_patch_bytes: int
    augmentation_overhead_bytes: int
    loss_computation_bytes: int
    buffer_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.feature_map_bytes
            + self.feature_gradient_bytes
            + self.parameter_bytes
            + self.parameter_gradient_bytes
            + self.optimizer_state_bytes
            + self.ema_state_bytes
            + self.input_patch_bytes
            + self.augmentation_overhead_bytes
            + self.loss_computation_bytes
            + self.buffer_bytes
        )

    @property
    def total_gb(self) -> float:
        return self.total_bytes / float(1024**3)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        msg = f"Expected YAML mapping in {path}, but found {type(loaded).__name__}."
        raise TypeError(msg)
    return loaded


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        msg = f"Expected JSON object in {path}, but found {type(loaded).__name__}."
        raise TypeError(msg)
    return loaded


def _align_down(values: np.ndarray, stride: int = _PATCH_STRIDE) -> np.ndarray:
    aligned = np.floor_divide(values.astype(np.int64), stride) * stride
    return np.maximum(aligned, _MIN_PATCH_SIZE)


def _align_up(values: np.ndarray, stride: int = _PATCH_STRIDE) -> np.ndarray:
    aligned = np.ceil(values.astype(np.float64) / stride).astype(np.int64) * stride
    return np.maximum(aligned, _MIN_PATCH_SIZE)


def _axis_ratio_error(candidate_xyz: np.ndarray, reference_xyz: np.ndarray) -> float:
    candidate = np.maximum(candidate_xyz.astype(np.float64), 1.0)
    reference = np.maximum(reference_xyz.astype(np.float64), 1.0)
    candidate_ratio = candidate / np.cbrt(np.prod(candidate))
    reference_ratio = reference / np.cbrt(np.prod(reference))
    return float(np.linalg.norm(np.log(candidate_ratio / reference_ratio)))


def _candidate_rank(
    candidate_xyz: np.ndarray,
    *,
    target_ratio_xyz: np.ndarray,
    representative_object_xyz: np.ndarray,
    preferred_lower_bound_xyz: np.ndarray,
    estimated_memory_gb: float,
) -> tuple[int, float, float, float, int, float]:
    candidate = candidate_xyz.astype(np.float64)
    axis_object_fit = candidate / representative_object_xyz
    whole_object_fit = int(np.prod(np.floor(axis_object_fit + 1e-6)))
    object_fit_margin = float(np.min(axis_object_fit))
    shortfall = float(np.maximum(preferred_lower_bound_xyz - candidate, 0.0).sum())
    ratio_error = _axis_ratio_error(candidate, target_ratio_xyz)
    volume = int(np.prod(candidate_xyz.astype(np.int64)))
    return (
        whole_object_fit,
        object_fit_margin,
        -shortfall,
        -ratio_error,
        volume,
        -estimated_memory_gb,
    )


def _flatten_tensors(value: object) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        tensors: list[torch.Tensor] = []
        for nested in value.values():
            tensors.extend(_flatten_tensors(nested))
        return tensors
    if isinstance(value, (list, tuple, set)):
        tensors = []
        for nested in value:
            tensors.extend(_flatten_tensors(nested))
        return tensors
    if hasattr(value, "__dict__"):
        return _flatten_tensors(vars(value))
    return []


def _build_estimation_config(  # noqa: PLR0913
    *,
    scale: ModelScale,
    batch_size: int,
    input_channels: int,
    num_classes: int,
    segmentation_output: bool,
    include_stride4_output: bool,
    prototype_masks: int,
    data_path: Path | None = None,
    enable_image_label_head: bool | None = None,
    reference_config: dict[str, Any] | DefaultConfig | None = None,
) -> DefaultConfig:
    if isinstance(reference_config, DefaultConfig):
        config_data = reference_config.model_dump(mode="python")
    elif isinstance(reference_config, dict):
        config_data = dict(reference_config)
    else:
        config_data = {}

    image_label_head_enabled = (
        bool(enable_image_label_head)
        if enable_image_label_head is not None
        else bool(config_data.get("enable_image_label_head", False))
    )
    # Tabular fusion is only valid when the image-label head is enabled.
    tabular_fusion_enabled = image_label_head_enabled and bool(
        config_data.get("enable_tabular_image_label_fusion", False)
    )

    config_data.update(
        {
            "input_channels": input_channels,
            "num_classes": num_classes,
            "model_scale": scale,
            "data": (
                Path(data_path).as_posix()
                if data_path is not None
                else config_data.get("data", ".")
            ),
            "fold": int(config_data.get("fold", 0)),
            "batch_size": batch_size,
            "patch_size": {"x": 64, "y": 64, "z": 64},
            "normalization": config_data.get("normalization", {"type": "none"}),
            "segmentation_output": segmentation_output,
            "include_stride4_output": include_stride4_output,
            "prototype_masks": prototype_masks,
            "enable_image_label_head": image_label_head_enabled,
            "enable_tabular_image_label_fusion": tabular_fusion_enabled,
            "experiment_name": config_data.get("experiment_name", "memory_estimate"),
        }
    )
    return DefaultConfig.model_validate(config_data)


class ModelMemoryEstimator:
    def __init__(
        self, train_config: DefaultConfig, estimated_boxes_per_image: int = 1
    ) -> None:
        self.train_config = train_config
        self.estimated_boxes_per_image = max(1, estimated_boxes_per_image)
        model = build_model_from_config(train_config).eval().to("meta")
        model.l23.train()
        self.model = model
        self._feature_map_elements_cache: dict[tuple[int, int, int, int], int] = {}
        self.parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.model.parameters()
        )
        self.buffer_bytes = sum(
            buffer.numel() * buffer.element_size() for buffer in self.model.buffers()
        )
        self.optimizer_state_bytes = self._estimate_optimizer_state_bytes()
        self.ema_state_bytes = self.parameter_bytes + self.buffer_bytes

    @property
    def activation_element_bytes(self) -> int:
        return 2 if self.train_config.amp else 4

    @property
    def integer_element_bytes(self) -> int:
        return torch.tensor(0, dtype=torch.int64).element_size()

    @property
    def boolean_element_bytes(self) -> int:
        return torch.tensor(False, dtype=torch.bool).element_size()

    def _estimate_optimizer_state_bytes(self) -> int:
        optimizer_name = self.train_config.optimizer.lower()
        state_multiplier = {
            "sgd": 1,
            "rmsprop": 2,
            "adam": 2,
            "adamax": 2,
            "adamw": 2,
            "nadam": 2,
            "radam": 2,
            "auto": 2,
        }.get(optimizer_name, 2)
        return self.parameter_bytes * state_multiplier

    def _collect_feature_map_elements(
        self, patch_size_xyz: tuple[int, int, int]
    ) -> int:
        batch_size = self.train_config.batch_size
        cache_key = (batch_size, *patch_size_xyz)
        cached = self._feature_map_elements_cache.get(cache_key)
        if cached is not None:
            return cached

        unique_tensor_ids: set[int] = set()
        feature_map_elements = 0

        def record_output(
            _module: torch.nn.Module,
            _args: tuple[object, ...],
            output: object,
        ) -> None:
            nonlocal feature_map_elements
            for tensor in _flatten_tensors(output):
                if id(tensor) in unique_tensor_ids:
                    continue
                unique_tensor_ids.add(id(tensor))
                feature_map_elements += tensor.numel()

        hooks = []
        for name, module in self.model.named_modules():
            if not name or "." in name:
                continue
            hooks.append(module.register_forward_hook(record_output))

        x_size, y_size, z_size = patch_size_xyz
        dummy_input = torch.zeros(
            (
                batch_size,
                self.train_config.input_channels,
                z_size,
                y_size,
                x_size,
            ),
            device="meta",
        )

        try:
            with torch.no_grad():
                model_output = self.model(dummy_input)
            for tensor in _flatten_tensors(model_output):
                if id(tensor) in unique_tensor_ids:
                    continue
                unique_tensor_ids.add(id(tensor))
                feature_map_elements += tensor.numel()
        finally:
            for hook in hooks:
                hook.remove()

        self._feature_map_elements_cache[cache_key] = feature_map_elements
        return feature_map_elements

    def _estimate_augmentation_overhead_bytes(
        self, patch_size_xyz: tuple[int, int, int]
    ) -> int:
        patch_voxels = int(np.prod(np.asarray(patch_size_xyz, dtype=np.int64)))
        image_bytes = (
            self.train_config.batch_size
            * patch_voxels
            * self.train_config.input_channels
            * self.activation_element_bytes
        )
        mask_bytes = 0
        if self.train_config.segmentation_output:
            mask_bytes = (
                self.train_config.batch_size
                * patch_voxels
                * self.activation_element_bytes
            )

        spatial_overhead = 0
        if (
            self.train_config.zoom_augmentation_device == "gpu"
            and self.train_config.prob_zoom > 0
        ):
            spatial_overhead = max(spatial_overhead, image_bytes + mask_bytes)
        if (
            self.train_config.rotation_augmentation_device == "gpu"
            and self.train_config.prob_rotation > 0
        ):
            spatial_overhead = max(spatial_overhead, image_bytes + mask_bytes)
        if (
            self.train_config.mirroring_augmentation_device == "gpu"
            and self.train_config.prob_mirroring > 0
        ):
            spatial_overhead = max(spatial_overhead, image_bytes // 2 + mask_bytes // 2)
        if (
            self.train_config.cutout_augmentation_device == "gpu"
            and self.train_config.prob_cutout > 0
        ):
            spatial_overhead = max(spatial_overhead, image_bytes // 2 + mask_bytes // 2)

        intensity_overhead = 0
        intensity_gpu_enabled = (
            (
                self.train_config.intensity_noise_augmentation_device == "gpu"
                and self.train_config.intensity_noise_prob > 0
            )
            or (
                self.train_config.intensity_blur_augmentation_device == "gpu"
                and self.train_config.intensity_blur_prob > 0
            )
            or (
                self.train_config.intensity_mult_brightness_augmentation_device == "gpu"
                and self.train_config.intensity_mult_brightness_prob > 0
            )
            or (
                self.train_config.intensity_add_brightness_augmentation_device == "gpu"
                and self.train_config.intensity_add_brightness_prob > 0
            )
            or (
                self.train_config.intensity_contrast_augmentation_device == "gpu"
                and self.train_config.intensity_contrast_prob > 0
            )
            or (
                self.train_config.intensity_lowres_augmentation_device == "gpu"
                and self.train_config.intensity_lowres_prob > 0
            )
            or (
                self.train_config.intensity_gamma_augmentation_device == "gpu"
                and self.train_config.intensity_gamma_prob > 0
            )
            or (
                self.train_config.channel_dropout_augmentation_device == "gpu"
                and any(
                    probability > 0
                    for probability in self.train_config.channel_dropout_probabilities
                )
            )
            or (
                self.train_config.gin_augmentation_device == "gpu"
                and self.train_config.gin_prob > 0
            )
        )
        if intensity_gpu_enabled:
            intensity_overhead = image_bytes

        return max(spatial_overhead, intensity_overhead)

    def _estimate_loss_computation_bytes(
        self, patch_size_xyz: tuple[int, int, int]
    ) -> int:
        x_size, y_size, z_size = patch_size_xyz
        dummy_input = torch.zeros(
            (
                self.train_config.batch_size,
                self.train_config.input_channels,
                z_size,
                y_size,
                x_size,
            ),
            device="meta",
        )
        with torch.no_grad():
            model_output = self.model(dummy_input)

        raw_prediction_shapes = [
            tuple(tensor.shape) for tensor in model_output.raw_predictions
        ]
        total_anchors = int(
            sum(np.prod(shape[2:], dtype=np.int64) for shape in raw_prediction_shapes)
        )
        batch_size = self.train_config.batch_size
        activation_bytes = self.activation_element_bytes
        integer_bytes = self.integer_element_bytes
        boolean_bytes = self.boolean_element_bytes
        detect_layer = self.model.l23
        num_classes = int(detect_layer.nc)
        reg_channels = int(detect_layer.reg_channels)
        output_channels = int(detect_layer.no)
        max_gt_boxes = self.estimated_boxes_per_image

        loss_bytes = 0

        loss_bytes += batch_size * total_anchors * output_channels * activation_bytes
        loss_bytes += batch_size * total_anchors * reg_channels * activation_bytes
        loss_bytes += batch_size * total_anchors * num_classes * activation_bytes
        if detect_layer.separate_objectness_head:
            loss_bytes += batch_size * total_anchors * activation_bytes

        loss_bytes += total_anchors * 4 * activation_bytes
        loss_bytes += batch_size * total_anchors * 12 * activation_bytes
        loss_bytes += batch_size * max_gt_boxes * 7 * activation_bytes
        loss_bytes += batch_size * max_gt_boxes * boolean_bytes

        assigner_matrix_elements = batch_size * max_gt_boxes * total_anchors
        loss_bytes += assigner_matrix_elements * activation_bytes * 4
        loss_bytes += assigner_matrix_elements * boolean_bytes
        loss_bytes += assigner_matrix_elements
        loss_bytes += batch_size * total_anchors * num_classes * integer_bytes
        loss_bytes += batch_size * total_anchors * num_classes * activation_bytes
        loss_bytes += batch_size * total_anchors * integer_bytes
        loss_bytes += batch_size * total_anchors * boolean_bytes
        loss_bytes += batch_size * total_anchors * 12 * activation_bytes

        if (
            self.train_config.segmentation_output
            and model_output.prototype_masks is not None
        ):
            prototype_voxels = int(
                np.prod(model_output.prototype_masks.shape[1:], dtype=np.int64)
            )
            per_mask_anchor_count = min(self.train_config.tal_topk, total_anchors)
            estimated_masks_per_batch = batch_size * max_gt_boxes
            if model_output.mask_coefficients is not None:
                loss_bytes += (
                    int(np.prod(model_output.mask_coefficients.shape, dtype=np.int64))
                    * activation_bytes
                )
            loss_bytes += (
                estimated_masks_per_batch * prototype_voxels * activation_bytes
            )
            loss_bytes += (
                estimated_masks_per_batch
                * per_mask_anchor_count
                * prototype_voxels
                * activation_bytes
                * 2
            )

        return loss_bytes

    def estimate(self, patch_size_xyz: tuple[int, int, int]) -> PatchMemoryBreakdown:
        feature_map_elements = self._collect_feature_map_elements(patch_size_xyz)
        feature_map_bytes = feature_map_elements * self.activation_element_bytes
        patch_voxels = int(np.prod(np.asarray(patch_size_xyz, dtype=np.int64)))
        input_patch_bytes = (
            self.train_config.batch_size
            * patch_voxels
            * self.train_config.input_channels
            * self.activation_element_bytes
        )
        return PatchMemoryBreakdown(
            feature_map_bytes=feature_map_bytes,
            feature_gradient_bytes=feature_map_bytes,
            parameter_bytes=self.parameter_bytes,
            parameter_gradient_bytes=self.parameter_bytes,
            optimizer_state_bytes=self.optimizer_state_bytes,
            ema_state_bytes=self.ema_state_bytes,
            input_patch_bytes=input_patch_bytes,
            augmentation_overhead_bytes=self._estimate_augmentation_overhead_bytes(
                patch_size_xyz
            ),
            loss_computation_bytes=self._estimate_loss_computation_bytes(
                patch_size_xyz
            ),
            buffer_bytes=self.buffer_bytes,
        )


def estimate_patch_memory(  # noqa: PLR0913
    patch_size_xyz: tuple[int, int, int],
    *,
    scale: ModelScale,
    batch_size: int,
    input_channels: int,
    num_classes: int,
    segmentation_output: bool,
    include_stride4_output: bool,
    prototype_masks: int,
    estimated_boxes_per_image: int = 1,
    data_path: Path | None = None,
    enable_image_label_head: bool | None = None,
    reference_config: dict[str, Any] | DefaultConfig | None = None,
) -> PatchMemoryBreakdown:
    estimator = ModelMemoryEstimator(
        _build_estimation_config(
            scale=scale,
            batch_size=batch_size,
            input_channels=input_channels,
            num_classes=num_classes,
            segmentation_output=segmentation_output,
            include_stride4_output=include_stride4_output,
            prototype_masks=prototype_masks,
            data_path=data_path,
            enable_image_label_head=enable_image_label_head,
            reference_config=reference_config,
        ),
        estimated_boxes_per_image=estimated_boxes_per_image,
    )
    return estimator.estimate(patch_size_xyz)


def estimate_patch_memory_gb(  # noqa: PLR0913
    patch_size_xyz: tuple[int, int, int],
    *,
    scale: ModelScale,
    batch_size: int,
    input_channels: int,
    num_classes: int,
    segmentation_output: bool,
    include_stride4_output: bool,
    prototype_masks: int,
    estimated_boxes_per_image: int = 1,
    data_path: Path | None = None,
    enable_image_label_head: bool | None = None,
    reference_config: dict[str, Any] | DefaultConfig | None = None,
) -> float:
    memory_breakdown = estimate_patch_memory(
        patch_size_xyz,
        scale=scale,
        batch_size=batch_size,
        input_channels=input_channels,
        num_classes=num_classes,
        segmentation_output=segmentation_output,
        include_stride4_output=include_stride4_output,
        prototype_masks=prototype_masks,
        estimated_boxes_per_image=estimated_boxes_per_image,
        data_path=data_path,
        enable_image_label_head=enable_image_label_head,
        reference_config=reference_config,
    )
    return memory_breakdown.total_gb


def _extract_channel_count(
    dataset_yaml: dict[str, Any], fingerprint: dict[str, Any]
) -> int:
    counts = []
    normalization = dataset_yaml.get("normalization")
    if (isinstance(normalization, dict) and "type" not in normalization) or isinstance(
        normalization, list
    ):
        counts.append(len(normalization))

    foreground_stats = fingerprint.get("foreground_channel_stats")
    if isinstance(foreground_stats, list):
        counts.append(len(foreground_stats))

    return max(counts or [1])


def _estimate_boxes_per_image(dataset_path: Path) -> int:
    labels_dir = dataset_path / "labels"
    if not labels_dir.exists():
        return 1

    counts = []
    for label_path in labels_dir.glob("*.txt"):
        lines = label_path.read_text(encoding="utf-8").splitlines()
        counts.append(sum(1 for line in lines if line.strip()))

    if not counts:
        return 1

    return max(1, int(np.ceil(np.percentile(np.asarray(counts, dtype=np.float64), 90))))


def _canonicalize_normalization_tag(value: str) -> NormalizationTag:
    key = "".join(char for char in value if char.isalnum()).lower()
    if key not in _CANONICAL_NORMALIZATION_TAGS:
        msg = (
            f"Unsupported dataset normalization tag '{value}'. Expected CT, MRGlobal or "
            "MRPerChannel."
        )
        raise ValueError(msg)
    return _CANONICAL_NORMALIZATION_TAGS[key]


def _parse_normalization_mapping(  # noqa: PLR0911
    value: object,
    n_channels: int,
) -> dict[int, NormalizationTag] | None:
    if isinstance(value, dict):
        if "type" in value:
            return None
        mapping: dict[int, NormalizationTag] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_value, str):
                return None
            mapping[int(raw_key)] = _canonicalize_normalization_tag(raw_value)
        if mapping:
            return {
                channel: mapping.get(channel, "MRPerChannel")
                for channel in range(n_channels)
            }
        return None

    if isinstance(value, list):
        if len(value) != n_channels:
            return None
        return {
            channel: _canonicalize_normalization_tag(str(raw_value))
            for channel, raw_value in enumerate(value)
        }

    if isinstance(value, str):
        tag = _canonicalize_normalization_tag(value)
        return dict.fromkeys(range(n_channels), tag)

    return None


def resolve_dataset_normalization_tags(
    dataset_yaml: dict[str, Any],
    n_channels: int,
) -> dict[int, NormalizationTag]:
    for key in ("normalization", "modalities", "modality", "channel_names"):
        if key not in dataset_yaml:
            continue
        mapping = _parse_normalization_mapping(dataset_yaml[key], n_channels)
        if mapping is not None:
            return mapping

    LOGGER.warning(
        "No dataset normalization mapping found in dataset.yaml; defaulting all %d channels to "
        "MRPerChannel.",
        n_channels,
    )
    return dict.fromkeys(range(n_channels), "MRPerChannel")


def build_runtime_normalization_config(
    normalization_tags: dict[int, NormalizationTag],
    fingerprint: dict[str, Any],
) -> dict[str, Any] | list[dict[str, Any]]:
    stats = fingerprint.get("foreground_channel_stats")
    if not isinstance(stats, list) or len(stats) < len(normalization_tags):
        msg = "Dataset fingerprint does not contain per-channel intensity statistics."
        raise ValueError(msg)

    ordered_tags = [
        normalization_tags[channel] for channel in sorted(normalization_tags)
    ]
    if ordered_tags and all(tag == "MRGlobal" for tag in ordered_tags):
        return {"type": "zscore"}

    if any(tag == "MRGlobal" for tag in ordered_tags):
        LOGGER.warning(
            "MRGlobal was requested only for a subset of channels. Falling back to per-channel "
            "zscore for those channels because the training config supports shared zscore only for "
            "all channels together."
        )

    runtime_entries: list[dict[str, Any]] = []
    for channel, tag in enumerate(ordered_tags):
        if tag == "CT":
            channel_stats = stats[channel]
            std = float(channel_stats.get("std", 1.0) or 1.0)
            runtime_entries.append(
                {
                    "type": "ct",
                    "mean": float(channel_stats.get("mean", 0.0)),
                    "std": max(std, 1e-6),
                    "clip_min": float(
                        channel_stats.get("p00_5", channel_stats.get("min", 0.0))
                    ),
                    "clip_max": float(
                        channel_stats.get("p99_5", channel_stats.get("max", 1.0))
                    ),
                }
            )
        else:
            runtime_entries.append({"type": "zscore"})

    return runtime_entries[0] if len(runtime_entries) == 1 else runtime_entries


def _preferred_patch_lower_bound(
    representative_large_object_xyz: np.ndarray,
    upper_bound_xyz: np.ndarray,
) -> np.ndarray:
    preferred = representative_large_object_xyz * 1.5
    return np.minimum(_align_up(preferred), upper_bound_xyz)


def propose_patch_size(  # noqa: PLR0913, PLR0912, PLR0915
    fingerprint: dict[str, Any],
    *,
    scale: ModelScale,
    gpu_ram_limit_gb: float,
    batch_size: int,
    input_channels: int,
    num_classes: int,
    segmentation_output: bool,
    include_stride4_output: bool,
    prototype_masks: int,
    estimated_boxes_per_image: int = 1,
    data_path: Path | None = None,
    enable_image_label_head: bool | None = None,
    reference_config: dict[str, Any] | DefaultConfig | None = None,
) -> tuple[PatchSize, float]:
    median_resampled = fingerprint.get("median_resampled_imagesize")
    if not isinstance(median_resampled, list) or len(median_resampled) != 3:
        msg = "Dataset fingerprint is missing median_resampled_imagesize."
        raise ValueError(msg)

    upper_bound_xyz = _align_up(np.asarray(median_resampled, dtype=np.int64))
    representative_large_object_xyz = np.asarray(fingerprint.get("box_size_p95_vox"), dtype=np.float64)
    preferred_lower_bound = _preferred_patch_lower_bound(representative_large_object_xyz, upper_bound_xyz)

    estimator = ModelMemoryEstimator(
        _build_estimation_config(
            scale=scale,
            batch_size=batch_size,
            input_channels=input_channels,
            num_classes=num_classes,
            segmentation_output=segmentation_output,
            include_stride4_output=include_stride4_output,
            prototype_masks=prototype_masks,
            data_path=data_path,
            enable_image_label_head=enable_image_label_head,
            reference_config=reference_config,
        ),
        estimated_boxes_per_image=estimated_boxes_per_image,
    )
    memory_cache: dict[tuple[int, int, int], float] = {}

    def estimate_candidate(candidate: tuple[int, int, int]) -> float:
        cached = memory_cache.get(candidate)
        if cached is not None:
            return cached
        estimated_memory = estimator.estimate(candidate).total_gb
        memory_cache[candidate] = estimated_memory
        return estimated_memory

    min_patch = tuple(int(value) for value in _MIN_PATCH_SIZE)
    if estimate_candidate(min_patch) > gpu_ram_limit_gb:
        fallback_memory = estimate_candidate(min_patch)
        LOGGER.warning(
            "GPU RAM limit %.2f GB is below the estimated requirement for the minimal %s patch at "
            "batch size %d (estimated %.2f GB). Proposing the minimal stride-compatible patch "
            "anyway.",
            gpu_ram_limit_gb,
            min_patch,
            batch_size,
            fallback_memory,
        )
        return PatchSize.from_xyz(min_patch), fallback_memory

    def candidate_from_factor(factor: float) -> tuple[int, int, int]:
        candidate = _align_up(
            np.maximum(_MIN_PATCH_SIZE, upper_bound_xyz.astype(np.float64) * factor)
        )
        candidate = np.minimum(candidate, upper_bound_xyz)
        return tuple(int(value) for value in candidate)

    def rank_candidate(
        candidate: tuple[int, int, int], estimated_memory: float
    ) -> tuple[int, float, float, float, int, float]:
        return _candidate_rank(
            np.asarray(candidate, dtype=np.int64),
            target_ratio_xyz=upper_bound_xyz,
            representative_object_xyz=representative_large_object_xyz,
            preferred_lower_bound_xyz=preferred_lower_bound.astype(np.float64),
            estimated_memory_gb=estimated_memory,
        )

    high_candidate = tuple(int(value) for value in upper_bound_xyz)
    if estimate_candidate(high_candidate) <= gpu_ram_limit_gb:
        best_candidate = high_candidate
    else:
        low_factor = 0.0
        high_factor = 1.0
        best_candidate = min_patch
        for _ in range(12):
            mid_factor = (low_factor + high_factor) / 2.0
            candidate = candidate_from_factor(mid_factor)
            estimated_memory = estimate_candidate(candidate)
            if estimated_memory <= gpu_ram_limit_gb:
                best_candidate = candidate
                low_factor = mid_factor
            else:
                high_factor = mid_factor

    best_candidate_array = np.asarray(best_candidate, dtype=np.int64)
    best_memory_gb = estimate_candidate(best_candidate)
    best_rank = rank_candidate(best_candidate, best_memory_gb)
    improved = True
    while improved:
        improved = False
        next_candidate: tuple[int, int, int] | None = None
        next_rank: tuple[int, float, float, float, int, float] | None = None
        next_memory_gb = float("inf")
        for axis in range(3):
            if best_candidate_array[axis] + _PATCH_STRIDE > upper_bound_xyz[axis]:
                continue
            candidate_array = best_candidate_array.copy()
            candidate_array[axis] += _PATCH_STRIDE
            candidate = tuple(int(value) for value in candidate_array)
            estimated_memory = estimate_candidate(candidate)
            if estimated_memory > gpu_ram_limit_gb:
                continue
            candidate_rank = rank_candidate(candidate, estimated_memory)
            if next_candidate is None or candidate_rank > next_rank:
                next_candidate = candidate
                next_rank = candidate_rank
                next_memory_gb = estimated_memory

        if next_candidate is not None and next_rank > best_rank:
            best_candidate = next_candidate
            best_candidate_array = np.asarray(best_candidate, dtype=np.int64)
            best_memory_gb = next_memory_gb
            best_rank = next_rank
            improved = True

    return PatchSize.from_xyz(best_candidate), best_memory_gb


def _default_output_path(
    dataset_path: Path, model_scale: ModelScale, gpu_ram_limit_gb: float
) -> Path:
    gpu_label = f"{gpu_ram_limit_gb:g}".replace(".", "_")
    return dataset_path / f"proposed_model_config_{model_scale}_{gpu_label}gb.yaml"


def propose_model_config(
    dataset_path: Path,
    model_scale: ModelScale,
    gpu_ram_limit_gb: float,
    *,
    output_path: Path | None = None,
    workers: int = 0,
    reference_config_path: Path = _DEFAULT_REFERENCE_CONFIG,
) -> ProposedModelConfigResult:
    if gpu_ram_limit_gb <= 0:
        msg = "gpu_ram_limit_gb must be greater than zero."
        raise ValueError(msg)

    dataset_path = Path(dataset_path)
    dataset_yaml_path = dataset_path / "dataset.yaml"
    if not dataset_yaml_path.exists():
        msg = f"Missing dataset.yaml at {dataset_yaml_path}"
        raise FileNotFoundError(msg)

    fingerprint_path = dataset_path / "dataset_fingerprint.json"
    analysis_ran = False
    if not fingerprint_path.exists():
        LOGGER.info("dataset_fingerprint.json missing. Running analyze-dataset first.")
        fingerprint_path = analyze_dataset_fn(
            dataset_path=dataset_path, workers=workers
        )
        analysis_ran = True

    dataset_yaml = _load_yaml(dataset_yaml_path)
    dataset_cfg = DatasetConfig.model_validate(dataset_yaml)
    fingerprint = _load_json(fingerprint_path)
    reference_config = _load_yaml(reference_config_path)

    input_channels = _extract_channel_count(dataset_yaml, fingerprint)
    normalization_tags = resolve_dataset_normalization_tags(
        dataset_yaml, input_channels
    )
    runtime_normalization = build_runtime_normalization_config(
        normalization_tags, fingerprint
    )
    enable_image_label_head = bool(dataset_yaml.get("image_label_tasks")) and bool(
        reference_config.get("enable_image_label_head", False)
    )
    enable_tabular_image_label_fusion = enable_image_label_head and bool(
        reference_config.get("enable_tabular_image_label_fusion", False)
    )

    masks_dir = dataset_path / "masks"
    has_masks = bool(
        fingerprint.get("n_images_with_masks", 0)
        or (masks_dir.exists() and any(masks_dir.iterdir()))
    )
    estimated_boxes_per_image = _estimate_boxes_per_image(dataset_path)

    batch_size = int(reference_config.get("batch_size", 2))
    patch_size, estimated_memory = propose_patch_size(
        fingerprint,
        scale=model_scale,
        gpu_ram_limit_gb=gpu_ram_limit_gb,
        batch_size=batch_size,
        input_channels=input_channels,
        num_classes=dataset_cfg.nc,
        segmentation_output=has_masks,
        include_stride4_output=bool(
            reference_config.get("include_stride4_output", False)
        ),
        prototype_masks=int(reference_config.get("prototype_masks", 8)),
        estimated_boxes_per_image=estimated_boxes_per_image,
        data_path=dataset_path,
        enable_image_label_head=enable_image_label_head,
        reference_config=reference_config,
    )
    if estimated_memory > gpu_ram_limit_gb and batch_size > 1:
        LOGGER.info(
            "Reducing batch size from %d to 1 because the minimal stride-compatible patch is still "
            "estimated at %.2f GB.",
            batch_size,
            estimated_memory,
        )
        batch_size = 1
        patch_size, estimated_memory = propose_patch_size(
            fingerprint,
            scale=model_scale,
            gpu_ram_limit_gb=gpu_ram_limit_gb,
            batch_size=batch_size,
            input_channels=input_channels,
            num_classes=dataset_cfg.nc,
            segmentation_output=has_masks,
            include_stride4_output=bool(
                reference_config.get("include_stride4_output", False)
            ),
            prototype_masks=int(reference_config.get("prototype_masks", 8)),
            estimated_boxes_per_image=estimated_boxes_per_image,
            data_path=dataset_path,
            enable_image_label_head=enable_image_label_head,
            reference_config=reference_config,
        )

    proposed_config = dict(reference_config)
    proposed_config.update(
        {
            "input_channels": input_channels,
            "model_scale": model_scale,
            "data": dataset_path.as_posix(),
            "batch_size": batch_size,
            "patch_size": {
                "x": patch_size.x,
                "y": patch_size.y,
                "z": patch_size.z,
            },
            "normalization": runtime_normalization,
            "segmentation_output": has_masks,
            "enable_image_label_head": enable_image_label_head,
            "enable_tabular_image_label_fusion": enable_tabular_image_label_fusion,
            "experiment_name": (
                f"{dataset_path.name}_{model_scale}_{patch_size.x}"
                f"x{patch_size.y}x{patch_size.z}"
            ),
        }
    )
    proposed_config["channel_dropout_probabilities"] = [0.0] * input_channels

    difference_channels = tuple(
        proposed_config.get("validation_plot_difference_channels", [0, 0])
    )
    if (
        len(difference_channels) != 2
        or max(difference_channels, default=0) >= input_channels
    ):
        proposed_config["validation_plot_difference_channels"] = [
            min(1, input_channels - 1),
            0,
        ]

    output_path = output_path or _default_output_path(
        dataset_path, model_scale, gpu_ram_limit_gb
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    DefaultConfig.model_validate({**proposed_config, "num_classes": dataset_cfg.nc})
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(proposed_config, file, sort_keys=False)

    return ProposedModelConfigResult(
        output_path=output_path,
        fingerprint_path=fingerprint_path,
        analysis_ran=analysis_ran,
        estimated_memory_gb=estimated_memory,
    )
