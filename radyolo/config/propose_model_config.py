from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml

from radyolo.config.config import (
    DatasetConfig,
    DefaultConfig,
    get_dataset_config,
    input_channels_from_dataset,
)
from radyolo.config.spatial import PatchSize, VoxelSize
from radyolo.data.analyze_dataset import analyze_dataset as analyze_dataset_fn

LOGGER = logging.getLogger(__name__)

ModelScale = Literal["n", "s", "m", "l", "x"]
NormalizationTag = Literal["CT", "MRGlobal", "MRPerChannel"]
TrainingModeSelection = Literal["automatic", "wholeimage", "patchwise"]
ResolvedTrainingMode = Literal["wholeimage", "patchwise"]

_DEFAULT_REFERENCE_CONFIG = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
_PATCH_STRIDE = 32
_MIN_PATCH_SIZE = np.array([32, 32, 32], dtype=np.int64)
_TARGET_PATCH_VOLUME = 256**3
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
    """Result information about proposed model config."""

    output_path: Path
    fingerprint_path: Path
    analysis_ran: bool
    training_mode: ResolvedTrainingMode


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and require a mapping at the top level."""
    with path.open(encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        msg = f"Expected YAML mapping in {path}, but found {type(loaded).__name__}."
        raise TypeError(msg)
    return loaded


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file and require an object at the top level."""
    with path.open(encoding="utf-8") as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        msg = f"Expected JSON object in {path}, but found {type(loaded).__name__}."
        raise TypeError(msg)
    return loaded


def _align_up(values: np.ndarray, stride: int = _PATCH_STRIDE) -> np.ndarray:
    """Round each axis up to the stride while enforcing the minimum patch size."""
    aligned = np.ceil(values.astype(np.float64) / stride).astype(np.int64) * stride
    return np.maximum(aligned, _MIN_PATCH_SIZE)


def _flatten_tensors(value: object) -> list[torch.Tensor]:
    """Collect all tensors nested inside arbitrary containers or objects."""
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


def _canonicalize_normalization_tag(value: str) -> NormalizationTag:
    """Map dataset normalization aliases to the supported canonical tag names."""
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
    """Parse dataset.yaml normalization metadata into a per-channel tag mapping."""
    if isinstance(value, dict):
        if "type" in value:
            return None
        mapping: dict[int, NormalizationTag] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_value, str):
                return None
            mapping[int(raw_key)] = _canonicalize_normalization_tag(raw_value)
        if mapping:
            return {channel: mapping.get(channel, "MRPerChannel") for channel in range(n_channels)}
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
    dataset_yaml: DatasetConfig | dict[str, Any],
    n_channels: int,
) -> dict[int, NormalizationTag]:
    """Resolve per-channel normalization tags from supported dataset.yaml fields."""
    if isinstance(dataset_yaml, DatasetConfig):
        normalization_candidates: dict[str, object] = {"normalization": dataset_yaml.normalization}
    else:
        normalization_candidates = dataset_yaml

    for key in ("normalization", "modalities", "modality", "channel_names"):
        if key not in normalization_candidates:
            continue
        mapping = _parse_normalization_mapping(normalization_candidates[key], n_channels)
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
    """Convert normalization tags plus fingerprint stats into train-config entries."""
    stats = fingerprint.get("foreground_channel_stats")
    if not isinstance(stats, list) or len(stats) < len(normalization_tags):
        msg = "Dataset fingerprint does not contain per-channel intensity statistics."
        raise ValueError(msg)

    ordered_tags = [normalization_tags[channel] for channel in sorted(normalization_tags)]
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
                    "clip_min": float(channel_stats.get("p00_5", channel_stats.get("min", 0.0))),
                    "clip_max": float(channel_stats.get("p99_5", channel_stats.get("max", 1.0))),
                }
            )
        else:
            runtime_entries.append({"type": "zscore"})

    return runtime_entries[0] if len(runtime_entries) == 1 else runtime_entries


def propose_patch_size(
    fingerprint: dict[str, Any],
) -> PatchSize:
    """Propose a stride-aligned patch size capped by the median resampled image size."""
    median_xyz = np.maximum(
        np.asarray(fingerprint["median_resampled_imagesize"], dtype=np.float64),
        1.0,
    )
    median_cap = _align_up(median_xyz)
    scale_factor = float((_TARGET_PATCH_VOLUME / np.prod(median_xyz)) ** (1.0 / 3.0))
    scaled_xyz = np.maximum(median_xyz * scale_factor, _MIN_PATCH_SIZE.astype(np.float64))
    aligned_patch = _align_up(scaled_xyz)
    capped_candidate = np.minimum(aligned_patch, median_cap)
    return PatchSize.from_xyz(capped_candidate)


def _resolve_training_mode(
    selection: TrainingModeSelection,
    patch_size: PatchSize,
    fingerprint: dict[str, Any],
) -> tuple[ResolvedTrainingMode, VoxelSize | None]:
    """Choose the final training mode and optional voxel size from the proposal inputs."""
    patch_xyz = np.asarray(patch_size.get_xyz(), dtype=np.float64)

    if selection == "wholeimage":
        return ("wholeimage", None)

    voxel_size = VoxelSize.from_xyz(fingerprint["median_voxel_size_mm"])
    if selection == "patchwise":
        return ("patchwise", voxel_size)

    median_image_size = PatchSize.from_xyz(
        np.ceil(fingerprint["median_resampled_imagesize"]).astype(np.int64)
    )
    median_xyz = np.asarray(median_image_size.get_xyz(), dtype=np.float64)
    if np.all(patch_xyz >= median_xyz):
        return ("wholeimage", None)
    return ("patchwise", voxel_size)


def _default_output_path(dataset_path: Path, model_scale: ModelScale) -> Path:
    return dataset_path / f"proposed_model_config_{model_scale}.yaml"


def propose_model_config(
    dataset_path: Path,
    model_scale: ModelScale,
    *,
    training_mode: TrainingModeSelection = "automatic",
    output_path: Path | None = None,
    workers: int = 0,
    reference_config_path: Path = _DEFAULT_REFERENCE_CONFIG,
) -> ProposedModelConfigResult:
    """Generate a dataset-specific training config proposal from dataset metadata."""
    dataset_path = Path(dataset_path)
    dataset_yaml_path = dataset_path / "dataset.yaml"
    if not dataset_yaml_path.exists():
        msg = f"Missing dataset.yaml at {dataset_yaml_path}"
        raise FileNotFoundError(msg)

    fingerprint_path = dataset_path / "dataset_fingerprint.json"
    analysis_ran = False
    if not fingerprint_path.exists():
        LOGGER.info("dataset_fingerprint.json missing. Running analyze-dataset first.")
        fingerprint_path = analyze_dataset_fn(dataset_path=dataset_path, workers=workers)
        analysis_ran = True

    dataset_cfg = get_dataset_config(dataset_path)
    fingerprint = _load_json(fingerprint_path)
    reference_config = _load_yaml(reference_config_path)

    input_channels = input_channels_from_dataset(dataset_cfg) or 1
    normalization_tags = resolve_dataset_normalization_tags(dataset_cfg, input_channels)
    runtime_normalization = build_runtime_normalization_config(normalization_tags, fingerprint)
    enable_image_label_head = bool(dataset_cfg.image_label_tasks) and bool(
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

    batch_size = int(reference_config.get("batch_size", 2))
    patch_size = propose_patch_size(
        fingerprint,
    )
    resolved_training_mode, proposed_voxel_size = _resolve_training_mode(
        training_mode,
        patch_size,
        fingerprint,
    )

    proposed_config = dict(reference_config)
    proposed_config.update(
        {
            "input_channels": input_channels,
            "model_scale": model_scale,
            "data": dataset_path.as_posix(),
            "batch_size": batch_size,
            "training_mode": resolved_training_mode,
            "patch_size": patch_size.model_dump(),
            "voxel_size": None if proposed_voxel_size is None else proposed_voxel_size.model_dump(),
            "normalization": runtime_normalization,
            "segmentation_output": has_masks,
            "enable_image_label_head": enable_image_label_head,
            "enable_tabular_image_label_fusion": enable_tabular_image_label_fusion,
            "experiment_name": (
                f"{dataset_path.name}_{model_scale}_{patch_size.x}x{patch_size.y}x{patch_size.z}"
            ),
        }
    )
    proposed_config["channel_dropout_probabilities"] = [0.0] * input_channels

    difference_channels = tuple(proposed_config.get("validation_plot_difference_channels", [0, 0]))
    if len(difference_channels) != 2 or max(difference_channels, default=0) >= input_channels:
        proposed_config["validation_plot_difference_channels"] = [
            min(1, input_channels - 1),
            0,
        ]

    output_path = output_path or _default_output_path(dataset_path, model_scale)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    DefaultConfig.model_validate({**proposed_config, "num_classes": dataset_cfg.nc})
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(proposed_config, file, sort_keys=False)

    return ProposedModelConfigResult(
        output_path=output_path,
        fingerprint_path=fingerprint_path,
        analysis_ran=analysis_ran,
        training_mode=resolved_training_mode,
    )
