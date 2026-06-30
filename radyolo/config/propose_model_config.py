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

LOGGER = logging.getLogger(__name__)

ModelScale = Literal["n", "s", "m", "l", "x"]
NormalizationTag = Literal["CT", "MRGlobal", "MRPerChannel"]

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
    output_path: Path
    fingerprint_path: Path
    analysis_ran: bool


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


def _extract_channel_count(dataset_yaml: dict[str, Any], fingerprint: dict[str, Any]) -> int:
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
    median_resampled = fingerprint.get("median_resampled_imagesize")
    if not isinstance(median_resampled, list) or len(median_resampled) != 3:
        msg = "Dataset fingerprint is missing median_resampled_imagesize."
        raise ValueError(msg)

    median_xyz = np.maximum(np.asarray(median_resampled, dtype=np.float64), 1.0)
    median_cap = _align_up(median_xyz)
    scale_factor = float((_TARGET_PATCH_VOLUME / np.prod(median_xyz)) ** (1.0 / 3.0))
    scaled_xyz = np.maximum(median_xyz * scale_factor, _MIN_PATCH_SIZE.astype(np.float64))

    down = _align_down(scaled_xyz)
    up = _align_up(scaled_xyz)

    target_ratio = median_xyz / np.cbrt(np.prod(median_xyz))

    def candidate_score(candidate_xyz: np.ndarray) -> tuple[float, float, int]:
        candidate = np.maximum(candidate_xyz.astype(np.float64), 1.0)
        candidate_ratio = candidate / np.cbrt(np.prod(candidate))
        ratio_error = float(np.linalg.norm(np.log(candidate_ratio / target_ratio)))
        volume = int(np.prod(candidate_xyz.astype(np.int64)))
        volume_error = abs(np.log(max(volume, 1) / _TARGET_PATCH_VOLUME))
        return (volume_error, ratio_error, -volume)

    candidates = [
        (x_candidate, y_candidate, z_candidate)
        for x_candidate in (int(down[0]), int(up[0]))
        for y_candidate in (int(down[1]), int(up[1]))
        for z_candidate in (int(down[2]), int(up[2]))
    ]

    unique_candidates = sorted(set(candidates))
    best_candidate = min(
        unique_candidates,
        key=lambda candidate: candidate_score(np.asarray(candidate, dtype=np.int64)),
    )
    capped_candidate = np.minimum(np.asarray(best_candidate, dtype=np.int64), median_cap)
    return PatchSize.from_xyz(capped_candidate)


def _default_output_path(dataset_path: Path, model_scale: ModelScale) -> Path:
    return dataset_path / f"proposed_model_config_{model_scale}.yaml"


def propose_model_config(
    dataset_path: Path,
    model_scale: ModelScale,
    *,
    output_path: Path | None = None,
    workers: int = 0,
    reference_config_path: Path = _DEFAULT_REFERENCE_CONFIG,
) -> ProposedModelConfigResult:
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

    dataset_yaml = _load_yaml(dataset_yaml_path)
    dataset_cfg = DatasetConfig.model_validate(dataset_yaml)
    fingerprint = _load_json(fingerprint_path)
    reference_config = _load_yaml(reference_config_path)

    input_channels = _extract_channel_count(dataset_yaml, fingerprint)
    normalization_tags = resolve_dataset_normalization_tags(dataset_yaml, input_channels)
    runtime_normalization = build_runtime_normalization_config(normalization_tags, fingerprint)
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

    batch_size = int(reference_config.get("batch_size", 2))
    patch_size = propose_patch_size(
        fingerprint,
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
    )
