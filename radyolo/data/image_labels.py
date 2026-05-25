from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

type ImageLabelValue = int | float | None
type ImageLabelTaskType = Literal["multiclass", "regression"]


@dataclass(frozen=True)
class ImageLabelTaskSpec:
    type: ImageLabelTaskType
    classes: dict[int, str] | None = None
    unit: str | None = None


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
        if task_type not in {"multiclass", "regression"}:
            msg = (
                f"image_label_tasks.{task_name}.type must be 'multiclass' or 'regression', "
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

        specs[task_name] = ImageLabelTaskSpec(type=task_type, classes=classes, unit=unit)

    return specs


def _validate_numeric_scalar(value: object, *, context: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{context} must be a numeric scalar."
        raise TypeError(msg)
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        msg = f"{context} must be finite."
        raise ValueError(msg)
    return value


def _coerce_image_label_value(
    task_name: str,
    raw_value: object,
    task_spec: ImageLabelTaskSpec | None,
) -> ImageLabelValue:
    context = f"labels_image task '{task_name}'"
    if raw_value is None:
        return None

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
    image_labels: dict[str, ImageLabelValue] = {}
    for task_name, raw_value in parsed.items():
        if not isinstance(task_name, str) or not task_name:
            msg = f"Image label file contains an invalid task name: {labels_image_path}"
            raise ValueError(msg)
        if specs and task_name not in specs:
            msg = (
                f"Image label task '{task_name}' in {labels_image_path} is not declared in "
                "dataset.yaml image_label_tasks."
            )
            raise KeyError(msg)
        image_labels[task_name] = _coerce_image_label_value(
            task_name,
            raw_value,
            specs.get(task_name),
        )
    return image_labels
