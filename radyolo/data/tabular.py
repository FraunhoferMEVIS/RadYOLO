from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import yaml


def load_tabular_feature_names(dataset_path: Path) -> tuple[str, ...]:
    dataset_yaml_path = Path(dataset_path) / "dataset.yaml"
    if not dataset_yaml_path.exists():
        return ()

    parsed = yaml.safe_load(dataset_yaml_path.read_text(encoding="utf-8")) or {}
    feature_names = parsed.get("tabular_feature_names")
    if feature_names is None:
        return ()
    if not isinstance(feature_names, list):
        msg = "dataset.yaml field 'tabular_feature_names' must be a list of feature names."
        raise TypeError(msg)

    resolved_names: list[str] = []
    for feature_name in feature_names:
        if not isinstance(feature_name, str) or not feature_name:
            msg = "dataset.yaml field 'tabular_feature_names' must contain non-empty strings."
            raise ValueError(msg)
        resolved_names.append(feature_name)

    deduplicated_names = tuple(dict.fromkeys(resolved_names))
    if len(deduplicated_names) != len(resolved_names):
        msg = "dataset.yaml field 'tabular_feature_names' must not contain duplicates."
        raise ValueError(msg)
    return deduplicated_names


def _coerce_tabular_value(
    value: object,
    *,
    case_name: str,
    feature_name: str,
    source_path: Path,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = (
            "Tabular feature values must be numeric scalars or null, but "
            f"'{feature_name}' in {source_path} for case '{case_name}' has value {value!r}."
        )
        raise TypeError(msg)
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        msg = (
            "Tabular feature values must be finite, but "
            f"'{feature_name}' in {source_path} for case '{case_name}' has value {value!r}."
        )
        raise ValueError(msg)
    return numeric_value


def load_case_tabular_inputs(
    dataset_path: Path,
    case_name: str,
    feature_names: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    num_features = len(feature_names)
    values = torch.zeros(num_features, dtype=torch.float32)
    observed_mask = torch.zeros(num_features, dtype=torch.bool)
    if num_features == 0:
        return values, observed_mask

    tabular_path = Path(dataset_path) / "inputs_tabular" / f"{case_name}.json"
    if not tabular_path.exists():
        return values, observed_mask

    parsed = json.loads(tabular_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        msg = f"Tabular input file must contain a JSON object: {tabular_path}"
        raise TypeError(msg)

    for feature_index, feature_name in enumerate(feature_names):
        raw_value = parsed.get(feature_name)
        if raw_value is None:
            continue
        values[feature_index] = _coerce_tabular_value(
            raw_value,
            case_name=case_name,
            feature_name=feature_name,
            source_path=tabular_path,
        )
        observed_mask[feature_index] = True

    return values, observed_mask
