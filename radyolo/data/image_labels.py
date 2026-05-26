from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import yaml

type ImageLabelValue = int | float | bool | None
type ImageLabelTaskType = Literal["multiclass", "survival"]


@dataclass(frozen=True)
class ImageLabelTaskSpec:
    type: ImageLabelTaskType
    classes: dict[int, str] | None = None
    unit: str | None = None
    masking: str | None = None
    survival_bin_thresholds: tuple[float, ...] | None = None


def _coerce_int_keys(mapping: object, *, context: str) -> dict[int, str]:
    if not isinstance(mapping, dict):
        msg = f"{context} must be a mapping of integer class ids to names."
        raise TypeError(msg)

    coerced: dict[int, str] = {}
    for raw_key, raw_value in mapping.items():
        try:
            key = int(raw_key)
        except (TypeError, ValueError) as exc:
            msg = f"{context} contains a non-integer class id: {raw_key!r}."
            raise ValueError(msg) from exc
        if not isinstance(raw_value, str) or not raw_value:
            msg = f"{context} class {key} must have a non-empty string name."
            raise ValueError(msg)
        coerced[key] = raw_value
    return coerced


def _validate_multiclass_names(classes: dict[int, str], *, task_name: str) -> None:
    if not classes:
        msg = f"image_label_tasks.{task_name}.classes must not be empty."
        raise ValueError(msg)
    actual_keys = set(classes)
    expected_keys = set(range(len(classes)))
    if actual_keys != expected_keys:
        msg = (
            f"image_label_tasks.{task_name}.classes must use contiguous class ids 0-"
            f"{len(classes) - 1}, got {sorted(actual_keys)}."
        )
        raise ValueError(msg)


def load_image_label_task_specs(dataset_path: Path) -> dict[str, ImageLabelTaskSpec]:
    dataset_yaml_path = Path(dataset_path) / "dataset.yaml"
    if not dataset_yaml_path.exists():
        return {}

    parsed = yaml.safe_load(dataset_yaml_path.read_text(encoding="utf-8")) or {}
    task_specs = parsed.get("image_label_tasks")
    if task_specs is None:
        return {}
    if not isinstance(task_specs, dict):
        msg = "dataset.yaml field 'image_label_tasks' must be a mapping of task names to specs."
        raise TypeError(msg)

    specs: dict[str, ImageLabelTaskSpec] = {}
    for task_name, raw_spec in task_specs.items():
        if not isinstance(task_name, str) or not task_name:
            msg = "image_label_tasks keys must be non-empty strings."
            raise ValueError(msg)
        if not isinstance(raw_spec, dict):
            msg = f"image_label_tasks.{task_name} must be a mapping."
            raise TypeError(msg)

        task_type = raw_spec.get("type")
        if task_type not in {"multiclass", "survival"}:
            msg = (
                f"image_label_tasks.{task_name}.type must be 'multiclass' or 'survival', "
                f"got {task_type!r}."
            )
            raise ValueError(msg)

        classes = None
        if task_type == "multiclass":
            classes = _coerce_int_keys(
                raw_spec.get("classes"),
                context=f"image_label_tasks.{task_name}.classes",
            )
            _validate_multiclass_names(classes, task_name=task_name)

        unit = raw_spec.get("unit")
        if unit is not None and not isinstance(unit, str):
            msg = f"image_label_tasks.{task_name}.unit must be a string when provided."
            raise TypeError(msg)

        masking = raw_spec.get("masking")
        if task_type == "survival":
            if not isinstance(masking, str) or not masking:
                msg = (
                    f"image_label_tasks.{task_name}.masking must be a non-empty string for "
                    "survival tasks."
                )
                raise ValueError(msg)
        elif masking is not None:
            msg = f"image_label_tasks.{task_name}.masking is only supported for survival tasks."
            raise ValueError(msg)

        specs[task_name] = ImageLabelTaskSpec(
            type=task_type,
            classes=classes,
            unit=unit,
            masking=masking,
        )

    return specs


def resolve_survival_task_specs(
    dataset_path: Path,
    task_specs: dict[str, ImageLabelTaskSpec],
    *,
    num_bins: int,
) -> dict[str, ImageLabelTaskSpec]:
    if num_bins < 1:
        msg = f"Survival task bin count must be at least 1, got {num_bins}."
        raise ValueError(msg)

    unresolved_survival_tasks = {
        task_name: task_spec
        for task_name, task_spec in task_specs.items()
        if task_spec.type == "survival" and task_spec.survival_bin_thresholds is None
    }
    if not unresolved_survival_tasks:
        return dict(task_specs)

    labels_image_path = Path(dataset_path) / "labels_image"
    if not labels_image_path.exists():
        msg = (
            "Survival image-label tasks require labeled JSON files under labels_image/ to "
            "derive bin thresholds."
        )
        raise ValueError(msg)

    values_by_task = {task_name: [] for task_name in unresolved_survival_tasks}
    for label_path in sorted(labels_image_path.glob("*.json")):
        image_labels = load_case_image_labels(dataset_path, label_path.stem, task_specs)
        for task_name in unresolved_survival_tasks:
            task_value = image_labels.get(task_name)
            if task_value is None:
                continue
            values_by_task[task_name].append(float(task_value))

    resolved_specs = dict(task_specs)
    for task_name, task_spec in unresolved_survival_tasks.items():
        task_values = values_by_task[task_name]
        if not task_values:
            msg = (
                f"Survival task '{task_name}' requires at least one labeled value to derive "
                "bin thresholds."
            )
            raise ValueError(msg)
        max_time = max(task_values)
        if max_time < 0.0:
            msg = f"Survival task '{task_name}' contains a negative survival time."
            raise ValueError(msg)

        if num_bins == 1 or max_time == 0.0:
            bin_thresholds = tuple(0.0 for _ in range(num_bins))
        else:
            step_size = max_time / float(num_bins - 1)
            bin_thresholds = tuple(step_size * float(bin_index) for bin_index in range(num_bins))

        resolved_specs[task_name] = replace(
            task_spec,
            survival_bin_thresholds=bin_thresholds,
        )
    return resolved_specs


def _validate_numeric_scalar(value: object, *, context: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{context} must be a numeric scalar."
        raise TypeError(msg)
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        msg = f"{context} must be finite."
        raise ValueError(msg)
    return value


def _validate_boolean_scalar(value: object, *, context: str) -> bool:
    if not isinstance(value, bool):
        msg = f"{context} must be a boolean scalar."
        raise TypeError(msg)
    return value


def _coerce_image_label_value(
    task_name: str,
    raw_value: object,
    task_spec: ImageLabelTaskSpec | None,
) -> ImageLabelValue:
    context = f"labels_image task '{task_name}'"
    if raw_value is None:
        return None

    if task_spec is None and isinstance(raw_value, bool):
        return _validate_boolean_scalar(raw_value, context=context)

    raw_value = _validate_numeric_scalar(raw_value, context=context)

    if task_spec is None:
        return int(raw_value) if isinstance(raw_value, int) else float(raw_value)

    if task_spec.type == "multiclass":
        numeric_value = float(raw_value)
        if not numeric_value.is_integer():
            msg = f"{context} must be an integer class id for multiclass tasks."
            raise ValueError(msg)
        class_id = int(numeric_value)
        if task_spec.classes is not None and class_id not in task_spec.classes:
            msg = (
                f"{context} has class id {class_id}, but valid ids are "
                f"{sorted(task_spec.classes)}."
            )
            raise ValueError(msg)
        return class_id

    return float(raw_value)


def load_case_image_labels(
    dataset_path: Path,
    case_name: str,
    task_specs: dict[str, ImageLabelTaskSpec] | None = None,
) -> dict[str, ImageLabelValue]:
    labels_image_path = Path(dataset_path) / "labels_image" / f"{case_name}.json"
    if not labels_image_path.exists():
        return {}

    parsed = json.loads(labels_image_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        msg = f"Image label file must contain a JSON object: {labels_image_path}"
        raise TypeError(msg)

    specs = task_specs or {}
    masking_fields = {
        task_spec.masking
        for task_spec in specs.values()
        if task_spec.masking is not None
    }
    image_labels: dict[str, ImageLabelValue] = {}
    for task_name, raw_value in parsed.items():
        if not isinstance(task_name, str) or not task_name:
            msg = f"Image label file contains an invalid task name: {labels_image_path}"
            raise ValueError(msg)
        if specs and task_name not in specs and task_name not in masking_fields:
            msg = (
                f"Image label task '{task_name}' in {labels_image_path} is not declared in "
                "dataset.yaml image_label_tasks."
            )
            raise KeyError(msg)
        if task_name in masking_fields and task_name not in specs:
            if raw_value is None:
                image_labels[task_name] = None
                continue
            image_labels[task_name] = _validate_boolean_scalar(
                raw_value,
                context=f"labels_image task '{task_name}'",
            )
            continue
        image_labels[task_name] = _coerce_image_label_value(
            task_name,
            raw_value,
            specs.get(task_name),
        )
    return image_labels
