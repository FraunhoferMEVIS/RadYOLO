from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib
import logging
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from ray import tune
from ray.tune import RunConfig, Tuner
from ray.tune.search.basic_variant import BasicVariantGenerator


def _ensure_project_root_on_sys_path(script_path: Path | None = None) -> Path:
    project_root = (script_path or Path(__file__)).resolve().parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


def _get_cfg_loader() -> Any:
    return importlib.import_module("radyolo.config.config").get_cfg


def _get_default_config_model() -> Any:
    return importlib.import_module("radyolo.config.config").DefaultConfig


_ensure_project_root_on_sys_path()

LOGGER = logging.getLogger(__name__)
RNG = np.random.default_rng()
TRIAL_PROGRESS_FILENAME = "trial_progress.yaml"

SUPPORTED_DISTRIBUTIONS = {
    "choice",
    "uniform",
    "loguniform",
    "one_minus_loguniform",
    "quniform",
    "qloguniform",
    "randint",
    "qrandint",
    "grid_search",
}

DEFAULT_TRAINING_COMMAND = ["uv", "run", "-m", "radyolo", "train"]


def _is_numeric_search_value(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_discrete_initial_point_value(
    value: object,
    spec: Mapping[str, object],
) -> str | None:
    values = spec.get("values", [])
    if value in values:
        return None
    return f"template default {value!r} is not in {values!r}"


def _validate_numeric_initial_point_bounds(
    distribution_type: str,
    value: float,
    spec: Mapping[str, object],
) -> str | None:
    lower = spec.get("lower")
    upper = spec.get("upper")

    if lower is not None and value < float(lower):
        return f"template default {value!r} is below lower bound {lower!r}"

    if upper is None:
        return None

    upper_value = float(upper)
    if distribution_type in {"randint", "qrandint"} and value >= upper_value:
        return f"template default {value!r} must be < upper bound {upper!r}"
    if distribution_type not in {"randint", "qrandint"} and value > upper_value:
        return f"template default {value!r} is above upper bound {upper!r}"
    return None


def _validate_numeric_initial_point_q(
    distribution_type: str,
    value: float,
    spec: Mapping[str, object],
) -> str | None:
    if distribution_type in {"randint", "qrandint"} and int(value) != value:
        return f"template default {value!r} must be an integer for {distribution_type}"

    q = spec.get("q")
    if q is None:
        return None

    q_value = float(q)
    if q_value <= 0:
        return f"distribution {distribution_type} requires q > 0"

    lower_value = float(spec.get("lower", 0.0))
    steps = (value - lower_value) / q_value
    if math.isclose(steps, round(steps), rel_tol=0.0, abs_tol=1e-9):
        return None
    return f"template default {value!r} does not align to q={q!r}"


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune RadYOLO hyperparameters from a tune_config.yaml file. "
            "Each sampled configuration is trained on all configured datasets, "
            "and the dataset metrics are aggregated into a single objective."
        )
    )
    parser.add_argument(
        "tune_config",
        type=Path,
        help="Path to tune_config.yaml",
    )
    parser.add_argument(
        "--resume-errored",
        action="store_true",
        help="Resume errored trials when restoring an existing Ray Tune experiment.",
    )
    return parser.parse_args()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        msg = f"Expected mapping at {path}, got {type(payload).__name__}"
        raise TypeError(msg)
    return payload


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _get_effective_template_config(template_config: dict[str, Any]) -> dict[str, Any]:
    default_config_model = _get_default_config_model()
    validation_payload = copy.deepcopy(template_config)
    validation_payload["num_classes"] = max(1, int(validation_payload.get("num_classes", 1)))
    effective_template_config = default_config_model.model_validate(
        validation_payload
    ).model_dump(mode="python")
    effective_template_config.pop("num_classes", None)
    return effective_template_config


def _build_distribution(spec: Any) -> tuple[Any, str | None]:
    if not isinstance(spec, dict) or "type" not in spec:
        return copy.deepcopy(spec), None

    distribution_type = str(spec["type"])
    cast_type = spec.get("cast")

    if distribution_type not in SUPPORTED_DISTRIBUTIONS:
        msg = (
            f"Unsupported search space type '{distribution_type}'. "
            f"Supported types: {sorted(SUPPORTED_DISTRIBUTIONS)}"
        )
        raise ValueError(msg)

    if distribution_type == "choice":
        values = spec.get("values")
        if not isinstance(values, list) or not values:
            msg = "choice requires a non-empty list in 'values'"
            raise ValueError(msg)
        return tune.choice(values), cast_type

    if distribution_type == "grid_search":
        values = spec.get("values")
        if not isinstance(values, list) or not values:
            msg = "grid_search requires a non-empty list in 'values'"
            raise ValueError(msg)
        return tune.grid_search(values), cast_type

    lower = spec.get("lower")
    upper = spec.get("upper")
    if lower is None or upper is None:
        msg = f"{distribution_type} requires both 'lower' and 'upper'"
        raise ValueError(msg)

    if distribution_type == "uniform":
        return tune.uniform(lower, upper), cast_type
    if distribution_type == "loguniform":
        return tune.loguniform(lower, upper), cast_type
    if distribution_type == "one_minus_loguniform":
        if not (0.0 < lower < upper < 1.0):
            msg = "one_minus_loguniform requires 0 < lower < upper < 1"
            raise ValueError(msg)

        lower_complement = 1.0 - float(upper)
        upper_complement = 1.0 - float(lower)
        return (
            tune.sample_from(
                lambda _: float(
                    1.0
                    - np.exp(
                        RNG.uniform(
                            np.log(lower_complement),
                            np.log(upper_complement),
                        )
                    )
                )
            ),
            cast_type,
        )
    if distribution_type == "quniform":
        q = spec.get("q")
        if q is None:
            msg = "quniform requires 'q'"
            raise ValueError(msg)
        return tune.quniform(lower, upper, q), cast_type
    if distribution_type == "qloguniform":
        q = spec.get("q")
        if q is None:
            msg = "qloguniform requires 'q'"
            raise ValueError(msg)
        return tune.qloguniform(lower, upper, q), cast_type
    if distribution_type == "randint":
        return tune.randint(int(lower), int(upper)), cast_type
    if distribution_type == "qrandint":
        q = spec.get("q")
        if q is None:
            msg = "qrandint requires 'q'"
            raise ValueError(msg)
        return tune.qrandint(int(lower), int(upper), int(q)), cast_type

    msg = f"Unhandled distribution type '{distribution_type}'"
    raise ValueError(msg)


def _build_param_space(search_space_cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    param_space: dict[str, Any] = {}
    casts: dict[str, str] = {}
    for key, spec in search_space_cfg.items():
        distribution, cast_hint = _build_distribution(spec)
        param_space[key] = distribution
        if cast_hint is not None:
            if cast_hint not in {"int", "float", "bool", "str"}:
                msg = f"Unsupported cast '{cast_hint}' for parameter '{key}'"
                raise ValueError(msg)
            casts[key] = cast_hint
    return param_space, casts


def _validate_initial_point_value(value: object, spec: object) -> str | None:
    if not isinstance(spec, Mapping) or "type" not in spec:
        return None

    distribution_type = str(spec["type"])
    if distribution_type in {"choice", "grid_search"}:
        return _validate_discrete_initial_point_value(value, spec)

    if distribution_type == "one_minus_loguniform":
        return (
            "distribution uses tune.sample_from and cannot be seeded via "
            "points_to_evaluate"
        )

    if not _is_numeric_search_value(value):
        return f"template default {value!r} must be numeric for {distribution_type}"

    numeric_value = float(value)
    bounds_error = _validate_numeric_initial_point_bounds(
        distribution_type,
        numeric_value,
        spec,
    )
    if bounds_error is not None:
        return bounds_error

    return _validate_numeric_initial_point_q(distribution_type, numeric_value, spec)


def _build_initial_evaluation_point(
    template_config: dict[str, Any],
    effective_template_config: dict[str, Any],
    search_space_cfg: dict[str, Any],
    cast_hints: dict[str, str],
) -> dict[str, Any]:
    initial_point: dict[str, Any] = {}
    missing_keys: list[str] = []
    skipped_keys: list[str] = []
    for key, spec in search_space_cfg.items():
        if key in effective_template_config:
            candidate_value = copy.deepcopy(effective_template_config[key])
        elif key in template_config:
            candidate_value = copy.deepcopy(template_config[key])
        else:
            missing_keys.append(key)
            continue

        validation_error = _validate_initial_point_value(candidate_value, spec)
        if validation_error is not None:
            skipped_keys.append(f"{key} ({validation_error})")
            continue

        initial_point[key] = candidate_value

    if missing_keys:
        LOGGER.warning(
            "Search-space keys without defaults in the template or validated runtime config "
            "will not be pinned for the first trial: %s",
            ", ".join(sorted(missing_keys)),
        )

    if skipped_keys:
        LOGGER.warning(
            "Skipping incompatible template defaults for the first seeded trial: %s",
            ", ".join(sorted(skipped_keys)),
        )

    return _cast_sampled(initial_point, cast_hints)


def _cast_sampled(sampled: dict[str, Any], casts: dict[str, str]) -> dict[str, Any]:
    casted: dict[str, Any] = {}
    for key, value in sampled.items():
        cast_type = casts.get(key)
        if cast_type == "int":
            casted[key] = int(value)
        elif cast_type == "float":
            casted[key] = float(value)
        elif cast_type == "bool":
            casted[key] = bool(value)
        elif cast_type == "str":
            casted[key] = str(value)
        else:
            casted[key] = value
    return casted


def _aggregate(values: list[float], method: str) -> float:
    if not values:
        return float("nan")
    values_arr = np.asarray(values, dtype=np.float64)
    if method == "mean":
        return float(np.mean(values_arr))
    if method == "median":
        return float(np.median(values_arr))
    if method == "min":
        return float(np.min(values_arr))
    if method == "max":
        return float(np.max(values_arr))
    msg = f"Unsupported aggregation method '{method}'"
    raise ValueError(msg)


def _best_rolling_average_and_epoch(
    series: pd.Series,
    epochs: pd.Series,
    window_size: int,
) -> tuple[float, int]:
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        msg = "Metric column has no numeric values"
        raise ValueError(msg)

    aligned_epochs = pd.to_numeric(epochs.loc[numeric_series.index], errors="coerce")
    if aligned_epochs.isna().any():
        aligned_epochs = pd.Series(numeric_series.index.to_numpy(), index=numeric_series.index)

    if len(numeric_series) < window_size:
        best_avg = float(numeric_series.mean())
        best_epoch = int(aligned_epochs.iloc[-1])
        return best_avg, best_epoch

    rolling = numeric_series.rolling(window=window_size, min_periods=window_size).mean()
    valid_rolling = rolling.dropna()
    if valid_rolling.empty:
        msg = "Rolling metric computation produced no valid values"
        raise ValueError(msg)

    best_index = valid_rolling.idxmax()
    best_avg = float(valid_rolling.loc[best_index])
    best_epoch = int(aligned_epochs.loc[best_index])
    return best_avg, best_epoch


def _extract_metric_from_csv(
    csv_path: Path,
    primary_metric_column: str,
    secondary_metric_column: str,
    rolling_window_epochs: int,
) -> tuple[float, float, int, int]:
    if not csv_path.exists():
        msg = f"Missing expected metrics file: {csv_path}"
        raise FileNotFoundError(msg)

    frame = pd.read_csv(csv_path)
    if frame.empty:
        msg = f"Metrics file has no rows: {csv_path}"
        raise ValueError(msg)

    if primary_metric_column not in frame.columns:
        msg = (
            f"Primary metric column '{primary_metric_column}' not found in {csv_path}. "
            f"Available columns: {list(frame.columns)}"
        )
        raise KeyError(msg)
    if secondary_metric_column not in frame.columns:
        msg = (
            f"Secondary metric column '{secondary_metric_column}' not found in {csv_path}. "
            f"Available columns: {list(frame.columns)}"
        )
        raise KeyError(msg)

    epoch_column = frame["epoch"] if "epoch" in frame.columns else pd.Series(frame.index)
    primary_best_avg, primary_best_epoch = _best_rolling_average_and_epoch(
        frame[primary_metric_column],
        epoch_column,
        rolling_window_epochs,
    )
    secondary_best_avg, secondary_best_epoch = _best_rolling_average_and_epoch(
        frame[secondary_metric_column],
        epoch_column,
        rolling_window_epochs,
    )

    return primary_best_avg, secondary_best_avg, primary_best_epoch, secondary_best_epoch


def _normalize_name(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)


def _compute_run_config_signature(run_config: Mapping[str, object]) -> str:
    payload = yaml.safe_dump(dict(run_config), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_trial_progress(trial_dir: Path) -> dict[str, Any]:
    progress_path = trial_dir / TRIAL_PROGRESS_FILENAME
    if not progress_path.exists():
        return {"datasets": {}}

    progress = _read_yaml(progress_path)
    datasets = progress.get("datasets")
    if not isinstance(datasets, dict):
        progress["datasets"] = {}
    return progress


def _write_trial_progress(trial_dir: Path, progress: Mapping[str, object]) -> None:
    progress_path = trial_dir / TRIAL_PROGRESS_FILENAME
    with progress_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(progress), handle, sort_keys=False)


def _get_dataset_progress_entry(
    trial_progress: Mapping[str, object],
    dataset_name: str,
) -> dict[str, Any] | None:
    datasets = trial_progress.get("datasets")
    if not isinstance(datasets, Mapping):
        return None
    entry = datasets.get(dataset_name)
    if not isinstance(entry, dict):
        return None
    return entry


def _update_dataset_progress_entry(
    trial_progress: dict[str, Any],
    dataset_name: str,
    updates: Mapping[str, object],
) -> dict[str, Any]:
    datasets = trial_progress.setdefault("datasets", {})
    if not isinstance(datasets, dict):
        msg = "trial progress datasets entry must be a mapping"
        raise TypeError(msg)

    entry = copy.deepcopy(datasets.get(dataset_name, {}))
    entry.update(copy.deepcopy(dict(updates)))
    entry["updated_at"] = dt.datetime.now(tz=dt.UTC).isoformat()
    datasets[dataset_name] = entry
    return entry


def _load_existing_run_config_signature(config_path: Path) -> str | None:
    if not config_path.exists():
        return None

    try:
        payload = _read_yaml(config_path)
    except Exception:
        LOGGER.warning("Failed to read existing training config at %s", config_path)
        return None
    return _compute_run_config_signature(payload)


def _get_cached_completed_dataset_metrics(
    progress_entry: Mapping[str, object] | None,
    config_signature: str,
) -> tuple[float, float, int, int, float] | None:
    if progress_entry is None:
        return None
    if progress_entry.get("config_signature") != config_signature:
        return None
    if progress_entry.get("status") != "completed":
        return None

    required_keys = (
        "primary_metric",
        "secondary_metric",
        "primary_best_epoch",
        "secondary_best_epoch",
        "training_minutes",
    )
    if any(key not in progress_entry for key in required_keys):
        return None

    return (
        float(progress_entry["primary_metric"]),
        float(progress_entry["secondary_metric"]),
        int(progress_entry["primary_best_epoch"]),
        int(progress_entry["secondary_best_epoch"]),
        float(progress_entry["training_minutes"]),
    )


def _should_resume_dataset_run(
    progress_entry: Mapping[str, object] | None,
    config_signature: str,
    last_checkpoint_path: Path,
    existing_config_path: Path,
) -> bool:
    if not last_checkpoint_path.exists():
        return False

    if progress_entry is not None and progress_entry.get("config_signature") == config_signature:
        return str(progress_entry.get("status", "")).lower() not in {
            "completed",
            "training_completed",
        }

    return _load_existing_run_config_signature(existing_config_path) == config_signature


def _restore_completed_dataset_metrics(
    progress_entry: Mapping[str, object] | None,
    config_signature: str,
    csv_path: Path,
    primary_metric_column: str,
    secondary_metric_column: str,
    rolling_window_epochs: int,
    trial_progress: dict[str, Any],
    trial_dir: Path,
    dataset_name: str,
) -> tuple[float, float, int, int, float] | None:
    cached_metrics = _get_cached_completed_dataset_metrics(progress_entry, config_signature)
    if cached_metrics is not None:
        return cached_metrics

    if progress_entry is None:
        return None
    if progress_entry.get("config_signature") != config_signature:
        return None
    if progress_entry.get("status") != "training_completed":
        return None

    (
        primary_metric,
        secondary_metric,
        primary_best_epoch,
        secondary_best_epoch,
    ) = _extract_metric_from_csv(
        csv_path,
        primary_metric_column,
        secondary_metric_column,
        rolling_window_epochs,
    )
    training_minutes = float(progress_entry.get("training_minutes", 0.0))
    _update_dataset_progress_entry(
        trial_progress,
        dataset_name,
        {
            "config_signature": config_signature,
            "status": "completed",
            "primary_metric": float(primary_metric),
            "secondary_metric": float(secondary_metric),
            "primary_best_epoch": int(primary_best_epoch),
            "secondary_best_epoch": int(secondary_best_epoch),
            "training_minutes": training_minutes,
        },
    )
    _write_trial_progress(trial_dir, trial_progress)
    return (
        float(primary_metric),
        float(secondary_metric),
        int(primary_best_epoch),
        int(secondary_best_epoch),
        training_minutes,
    )


def _alphabetical_restore_trial_sort_key(trial: object) -> tuple[str, str]:
    trial_id = str(getattr(trial, "trial_id", "")).lower()
    local_path = str(getattr(trial, "local_path", "")).lower()
    return trial_id, local_path


def _patch_ray_restore_trial_order() -> None:
    from ray.tune.execution.tune_controller import TuneController
    from ray.tune.experiment import Trial
    from ray.tune.tune_config import ResumeConfig

    if getattr(TuneController, "_radyolo_alphabetical_restore_patch", False):
        return

    def _requeue_restored_trials_alphabetically(
        self: Any,
        trials: list[Trial],
        resume_config: ResumeConfig,
    ) -> None:
        for trial in sorted(trials, key=_alphabetical_restore_trial_sort_key):
            if trial.status == Trial.ERROR:
                resume_type = resume_config.errored
            elif trial.status == Trial.TERMINATED:
                resume_type = resume_config.finished
            else:
                resume_type = resume_config.unfinished

            trial_to_add = None
            if resume_type == ResumeConfig.ResumeType.RESUME:
                trial_to_add = trial
                trial_to_add.run_metadata.error_filename = None
                trial_to_add.run_metadata.pickled_error_filename = None
                trial_to_add.set_status(Trial.PENDING)
            elif resume_type == ResumeConfig.ResumeType.RESTART:
                trial_to_add = trial.reset()
                trial_to_add.restore_path = None
            elif resume_type == ResumeConfig.ResumeType.SKIP:
                trial_to_add = trial
                if trial_to_add.status != Trial.ERROR:
                    trial_to_add.set_status(Trial.TERMINATED)
            else:
                msg = f"Unknown resume type: {resume_type}"
                raise ValueError(msg)

            self.add_trial(trial_to_add)

    TuneController._requeue_restored_trials = _requeue_restored_trials_alphabetically
    TuneController._radyolo_alphabetical_restore_patch = True


def _as_positive_patch_size(value: Any, *, field_name: str) -> dict[str, int]:
    if isinstance(value, dict):
        expected_axes = ("x", "y", "z")
        missing_axes = [axis for axis in expected_axes if axis not in value]
        if missing_axes:
            msg = (
                f"'{field_name}' must define axes x, y, z; "
                f"missing {', '.join(missing_axes)}"
            )
            raise ValueError(msg)
        ordered_values = [value[axis] for axis in expected_axes]
        axis_labels = list(expected_axes)
    elif isinstance(value, (list, tuple)):
        if len(value) != 3:
            msg = f"'{field_name}' must be a 3-item sequence or {{x, y, z}} mapping"
            raise ValueError(msg)
        ordered_values = list(value)
        axis_labels = ["x", "y", "z"]
    else:
        msg = f"'{field_name}' must be a 3-item sequence or {{x, y, z}} mapping"
        raise TypeError(msg)

    parsed: dict[str, int] = {}
    for axis_label, item in zip(axis_labels, ordered_values, strict=True):
        if isinstance(item, bool):
            msg = f"'{field_name}.{axis_label}' must be an integer, got bool"
            raise TypeError(msg)
        try:
            int_item = int(item)
        except (TypeError, ValueError) as exc:
            msg = f"'{field_name}.{axis_label}' must be an integer"
            raise TypeError(msg) from exc
        if int_item <= 0:
            msg = f"'{field_name}.{axis_label}' must be > 0"
            raise ValueError(msg)
        parsed[axis_label] = int_item
    return parsed


def _build_dataset_train_overrides(dataset_entry: dict[str, Any], *, index: int) -> dict[str, Any]:
    merged_overrides: dict[str, Any] = {}

    raw_overrides = dataset_entry.get("overrides", {})
    if raw_overrides is None:
        raw_overrides = {}
    if not isinstance(raw_overrides, dict):
        msg = f"Dataset entry #{index} key 'overrides' must be a mapping"
        raise TypeError(msg)
    merged_overrides.update(copy.deepcopy(raw_overrides))

    if "normalization" in dataset_entry:
        merged_overrides["normalization"] = copy.deepcopy(dataset_entry["normalization"])

    if "resampling" in dataset_entry:
        resampling_cfg = dataset_entry["resampling"]
        if not isinstance(resampling_cfg, dict):
            msg = f"Dataset entry #{index} key 'resampling' must be a mapping"
            raise TypeError(msg)

        if "patch_size" not in resampling_cfg:
            msg = f"Dataset entry #{index} 'resampling' must define 'patch_size'"
            raise KeyError(msg)

        merged_overrides["patch_size"] = _as_positive_patch_size(
            resampling_cfg["patch_size"],
            field_name=f"datasets[{index - 1}].resampling.patch_size",
        )

    return merged_overrides


def _run_training(command: list[str], config_path: Path, *, cwd: Path | None = None) -> int:
    full_command = [*command, str(config_path)]
    LOGGER.info("Running training command: %s", " ".join(full_command))
    completed = subprocess.run(full_command, check=False, cwd=str(cwd) if cwd else None)  # noqa: S603
    return int(completed.returncode)


def _run_trial(
    sampled_hyperparameters: dict[str, Any],
    *,
    tune_settings: dict[str, Any],
    template_config: dict[str, Any],
    cast_hints: dict[str, str],
) -> None:
    sampled = _cast_sampled(sampled_hyperparameters, cast_hints)
    tune_context = tune.get_context()
    trial_id = tune_context.get_trial_id()

    output_root = Path(tune_settings["output_root"]).resolve()
    experiment_name = str(tune_settings["experiment_name"])
    trial_dir = output_root / experiment_name / "trial_runs" / f"trial_{trial_id}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial_progress = _load_trial_progress(trial_dir)
    trial_progress["trial_id"] = trial_id
    trial_progress["sampled_hyperparameters"] = copy.deepcopy(sampled)
    _write_trial_progress(trial_dir, trial_progress)

    metrics_cfg = tune_settings["metrics"]
    datasets = tune_settings["datasets"]
    command = DEFAULT_TRAINING_COMMAND
    training_cwd = Path(tune_settings["training_working_directory"]).resolve()
    common_overrides = tune_settings.get("common_train_overrides", {})
    failure_metric = float(metrics_cfg.get("failure_metric", 0.0))

    primary_metric_column = str(metrics_cfg.get("primary_column", "metrics/mAP10-95"))
    secondary_metric_column = str(metrics_cfg.get("secondary_column", "metrics/dice"))
    rolling_window_epochs = int(metrics_cfg.get("rolling_window_epochs", 20))
    if rolling_window_epochs <= 0:
        msg = "metrics.rolling_window_epochs must be a positive integer"
        raise ValueError(msg)
    dataset_aggregation = str(metrics_cfg.get("dataset_aggregation", "mean"))

    dataset_primary_metrics: list[float] = []
    dataset_secondary_metrics: list[float] = []
    dataset_primary_best_epochs: list[int] = []
    dataset_secondary_best_epochs: list[int] = []
    report_payload: dict[str, float | int] = {}

    for dataset in datasets:
        dataset_name = _normalize_name(str(dataset.get("name", Path(str(dataset["data"])).name)))
        dataset_dir = trial_dir / dataset_name
        output_folder_base = dataset_dir / "runs"
        train_output_dir = output_folder_base / "train"
        results_csv_path = train_output_dir / "results.csv"
        last_checkpoint_path = train_output_dir / "weights_last.pt"
        output_folder_base.mkdir(parents=True, exist_ok=True)

        run_config = _deep_merge(template_config, common_overrides)
        run_config.update(sampled)
        run_config.update(dataset.get("train_overrides", {}))

        run_config["data"] = str(Path(dataset["data"]).resolve())
        run_config["fold"] = int(dataset.get("fold", run_config.get("fold", 0)))
        run_config["output_folder_base"] = str(output_folder_base)
        run_config["experiment_name"] = "train"
        run_config["exist_ok"] = True

        config_path = dataset_dir / "train_config.yaml"
        config_signature = _compute_run_config_signature(run_config)
        progress_entry = _get_dataset_progress_entry(trial_progress, dataset_name)
        primary_metric_key = f"dataset_metric/{dataset_name}"
        secondary_metric_key = f"dataset_metric_secondary/{dataset_name}"
        primary_epoch_key = f"dataset_best_epoch/{dataset_name}"
        secondary_epoch_key = f"dataset_best_epoch_secondary/{dataset_name}"
        training_minutes_key = f"dataset_training_minutes/{dataset_name}"

        restored_metrics = _restore_completed_dataset_metrics(
            progress_entry,
            config_signature,
            results_csv_path,
            primary_metric_column,
            secondary_metric_column,
            rolling_window_epochs,
            trial_progress,
            trial_dir,
            dataset_name,
        )
        if restored_metrics is not None:
            (
                primary_metric,
                secondary_metric,
                primary_best_epoch,
                secondary_best_epoch,
                training_minutes,
            ) = restored_metrics
            report_payload[f"dataset_failed/{dataset_name}"] = 0
        else:
            run_config["resume"] = _should_resume_dataset_run(
                progress_entry,
                config_signature,
                last_checkpoint_path,
                config_path,
            )
            with config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(run_config, handle, sort_keys=False)

            _update_dataset_progress_entry(
                trial_progress,
                dataset_name,
                {
                    "config_signature": config_signature,
                    "status": "running",
                },
            )
            _write_trial_progress(trial_dir, trial_progress)

            training_start = time.perf_counter()
            return_code = _run_training(command, config_path, cwd=training_cwd)
            training_minutes = (time.perf_counter() - training_start) / 60.0
            if return_code != 0:
                LOGGER.warning(
                    "Training failed for dataset '%s' in trial %s with exit code %d",
                    dataset_name,
                    trial_id,
                    return_code,
                )
                primary_metric = failure_metric
                secondary_metric = failure_metric
                primary_best_epoch = -1
                secondary_best_epoch = -1
                report_payload[f"dataset_failed/{dataset_name}"] = 1
                _update_dataset_progress_entry(
                    trial_progress,
                    dataset_name,
                    {
                        "config_signature": config_signature,
                        "status": "interrupted",
                        "training_minutes": float(training_minutes),
                    },
                )
                _write_trial_progress(trial_dir, trial_progress)
            else:
                _update_dataset_progress_entry(
                    trial_progress,
                    dataset_name,
                    {
                        "config_signature": config_signature,
                        "status": "training_completed",
                        "training_minutes": float(training_minutes),
                    },
                )
                _write_trial_progress(trial_dir, trial_progress)
                try:
                    (
                        primary_metric,
                        secondary_metric,
                        primary_best_epoch,
                        secondary_best_epoch,
                    ) = _extract_metric_from_csv(
                        results_csv_path,
                        primary_metric_column,
                        secondary_metric_column,
                        rolling_window_epochs,
                    )
                    report_payload[f"dataset_failed/{dataset_name}"] = 0
                    _update_dataset_progress_entry(
                        trial_progress,
                        dataset_name,
                        {
                            "config_signature": config_signature,
                            "status": "completed",
                            "primary_metric": float(primary_metric),
                            "secondary_metric": float(secondary_metric),
                            "primary_best_epoch": int(primary_best_epoch),
                            "secondary_best_epoch": int(secondary_best_epoch),
                            "training_minutes": float(training_minutes),
                        },
                    )
                    _write_trial_progress(trial_dir, trial_progress)
                except Exception:
                    LOGGER.exception(
                        "Failed metric extraction for dataset '%s' in trial %s",
                        dataset_name,
                        trial_id,
                    )
                    primary_metric = failure_metric
                    secondary_metric = failure_metric
                    primary_best_epoch = -1
                    secondary_best_epoch = -1
                    report_payload[f"dataset_failed/{dataset_name}"] = 1

        if math.isnan(primary_metric):
            primary_metric = failure_metric
            primary_best_epoch = -1
            report_payload[f"dataset_failed/{dataset_name}"] = 1

        if math.isnan(secondary_metric):
            secondary_metric = failure_metric
            secondary_best_epoch = -1
            report_payload[f"dataset_failed/{dataset_name}"] = 1

        dataset_primary_metrics.append(float(primary_metric))
        dataset_secondary_metrics.append(float(secondary_metric))
        dataset_primary_best_epochs.append(int(primary_best_epoch))
        dataset_secondary_best_epochs.append(int(secondary_best_epoch))
        report_payload[primary_metric_key] = float(primary_metric)
        report_payload[secondary_metric_key] = float(secondary_metric)
        report_payload[primary_epoch_key] = int(primary_best_epoch)
        report_payload[secondary_epoch_key] = int(secondary_best_epoch)
        report_payload[training_minutes_key] = float(training_minutes)

    aggregate_primary_metric = _aggregate(dataset_primary_metrics, dataset_aggregation)
    aggregate_secondary_metric = _aggregate(dataset_secondary_metrics, dataset_aggregation)

    report_payload["dataset_metric_mean"] = float(np.mean(dataset_primary_metrics))
    report_payload["dataset_metric_std"] = float(np.std(dataset_primary_metrics))
    report_payload["dataset_metric_aggregate"] = float(aggregate_primary_metric)
    report_payload["dataset_metric_secondary_mean"] = float(np.mean(dataset_secondary_metrics))
    report_payload["dataset_metric_secondary_std"] = float(np.std(dataset_secondary_metrics))
    report_payload["dataset_metric_secondary_aggregate"] = float(aggregate_secondary_metric)
    report_payload["dataset_best_epoch_mean"] = float(np.mean(dataset_primary_best_epochs))
    report_payload["dataset_best_epoch_secondary_mean"] = float(
        np.mean(dataset_secondary_best_epochs)
    )
    report_payload["num_datasets"] = len(dataset_primary_metrics)

    tune.report(report_payload)


def _validate_dataset_entries(datasets: Any) -> list[dict[str, Any]]:
    if not isinstance(datasets, list) or not datasets:
        msg = "tune_config.yaml requires a non-empty 'datasets' list"
        raise ValueError(msg)

    validated: list[dict[str, Any]] = []
    for idx, entry in enumerate(datasets, start=1):
        if not isinstance(entry, dict):
            msg = f"Dataset entry #{idx} must be a mapping"
            raise TypeError(msg)
        if "data" not in entry:
            msg = f"Dataset entry #{idx} is missing required key 'data'"
            raise KeyError(msg)
        validated_entry = copy.deepcopy(entry)
        validated_entry["train_overrides"] = _build_dataset_train_overrides(entry, index=idx)
        validated.append(validated_entry)
    return validated


def _resolve_training_working_directory(raw_cwd: Any, tune_config_path: Path) -> Path:
    if raw_cwd is None:
        # Default to repository root (parent of hyperparameter_tuning/) to avoid
        # inheriting Ray's very long trial working directories on Windows.
        return tune_config_path.resolve().parent.parent

    resolved = Path(str(raw_cwd)).expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        msg = f"Configured training_working_directory does not exist or is not a directory: {resolved}"
        raise FileNotFoundError(msg)
    return resolved


def _short_trial_dirname_creator(trial: Any) -> str:
    trial_id = getattr(trial, "trial_id", "trial")
    return f"t_{trial_id}"


def _load_tune_settings(
    config_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    dict[str, Any],
    dict[str, Any],
]:
    tune_cfg = _read_yaml(config_path)

    template_path = tune_cfg.get("template_config")
    if template_path is None:
        msg = "tune_config.yaml is missing required key 'template_config'"
        raise KeyError(msg)

    template_config_path = Path(str(template_path)).expanduser().resolve()
    template_config = _read_yaml(template_config_path)
    effective_template_config = _get_effective_template_config(template_config)

    search_space_cfg = tune_cfg.get("search_space")
    if not isinstance(search_space_cfg, dict) or not search_space_cfg:
        msg = "tune_config.yaml requires a non-empty 'search_space' mapping"
        raise ValueError(msg)

    param_space, cast_hints = _build_param_space(search_space_cfg)
    initial_evaluation_point = _build_initial_evaluation_point(
        template_config,
        effective_template_config,
        search_space_cfg,
        cast_hints,
    )

    output_root = Path(str(tune_cfg.get("output_root", "hyperparameter_tuning/runs"))).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    experiment_name = str(
        tune_cfg.get(
            "experiment_name",
            f"tune_{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d_%H%M%S')}",
        )
    )

    metrics_cfg = tune_cfg.get("metrics", {})
    if not isinstance(metrics_cfg, dict):
        msg = "'metrics' must be a mapping"
        raise TypeError(msg)

    tune_settings = {
        "output_root": str(output_root),
        "experiment_name": experiment_name,
        "datasets": _validate_dataset_entries(tune_cfg.get("datasets")),
        "metrics": metrics_cfg,
        "training_working_directory": str(
            _resolve_training_working_directory(
                tune_cfg.get("training_working_directory"),
                config_path,
            )
        ),
        "common_train_overrides": tune_cfg.get("common_train_overrides", {}),
    }

    return (
        tune_cfg,
        template_config,
        param_space,
        cast_hints,
        tune_settings,
        initial_evaluation_point,
    )


def main() -> int:
    _setup_logging()
    args = _parse_args()

    (
        tune_cfg,
        template_config,
        param_space,
        cast_hints,
        tune_settings,
        initial_evaluation_point,
    ) = _load_tune_settings(args.tune_config)

    tune_options = tune_cfg.get("tune", {})
    if not isinstance(tune_options, dict):
        msg = "'tune' must be a mapping"
        raise TypeError(msg)

    resources = tune_options.get("resources", {"cpu": 4, "gpu": 1})
    num_samples = int(tune_options.get("num_samples", 50))
    max_concurrent_trials = int(tune_options.get("max_concurrent_trials", 2))
    use_short_trial_dirnames = bool(
        tune_options.get("use_short_trial_dirnames", os.name == "nt")
    )

    metric_name = str(tune_options.get("metric", "dataset_metric_aggregate"))
    metric_mode = str(tune_options.get("mode", "max"))

    trainable = tune.with_parameters(
        _run_trial,
        tune_settings=tune_settings,
        template_config=template_config,
        cast_hints=cast_hints,
    )
    trainable_with_resources = tune.with_resources(trainable, resources)

    output_root = Path(tune_settings["output_root"])
    experiment_name = str(tune_settings["experiment_name"])
    experiment_dir = output_root / experiment_name

    if Tuner.can_restore(str(experiment_dir)):
        _patch_ray_restore_trial_order()
        LOGGER.info("Restoring existing Ray Tune experiment at %s", experiment_dir)
        tuner = Tuner.restore(
            str(experiment_dir),
            trainable=trainable_with_resources,
            param_space=param_space,
            resume_errored=args.resume_errored,
        )
    else:
        LOGGER.info(
            "Seeding first tuning trial with template defaults for search-space keys: %s",
            initial_evaluation_point,
        )
        tuner = Tuner(
            trainable=trainable_with_resources,
            param_space=param_space,
            tune_config=tune.TuneConfig(
                metric=metric_name,
                mode=metric_mode,
                search_alg=BasicVariantGenerator(
                    points_to_evaluate=[initial_evaluation_point],
                ),
                num_samples=num_samples,
                max_concurrent_trials=max_concurrent_trials,
                trial_dirname_creator=(
                    _short_trial_dirname_creator if use_short_trial_dirnames else None
                ),
            ),
            run_config=RunConfig(
                name=experiment_name,
                storage_path=str(output_root),
                verbose=1,
            ),
        )

    results = tuner.fit()
    try:
        best = results.get_best_result(metric=metric_name, mode=metric_mode)
    except RuntimeError:
        LOGGER.exception(
            "No best trial found for metric '%s'. All trials may have failed before reporting.",
            metric_name,
        )
        return 1

    LOGGER.info("Best trial id: %s", best.metrics.get("trial_id", "n/a"))
    LOGGER.info("Best %s: %.6f", metric_name, float(best.metrics[metric_name]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
