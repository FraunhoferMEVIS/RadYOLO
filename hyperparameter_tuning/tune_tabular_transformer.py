from __future__ import annotations  # noqa: I001

import torch

import argparse
import copy
import datetime as dt
import importlib
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import yaml
from ray import tune
from ray.tune import RunConfig, Tuner
from ray.tune.search.basic_variant import BasicVariantGenerator
from torch import nn
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


def _ensure_project_root_on_sys_path(script_path: Path | None = None) -> Path:
    project_root = (script_path or Path(__file__)).resolve().parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


PROJECT_ROOT = _ensure_project_root_on_sys_path()

from hyperparameter_tuning.tune_learning_parameters import (  # noqa: E402
    _aggregate,
    _build_initial_evaluation_point,
    _build_param_space,
    _cast_sampled,
    _deep_merge,
    _extract_metric_from_csv,
    _get_effective_template_config,
    _metric_report_key,
    _normalize_name,
    _patch_ray_restore_trial_order,
    _read_yaml,
    _resolve_training_working_directory,
    _setup_logging,
    _short_trial_dirname_creator,
)
from radyolo.config.config import DefaultConfig  # noqa: E402
from radyolo.data.augmentations import apply_tabular_mask_augmentation  # noqa: E402
from radyolo.data.data import _load_split_cases  # noqa: E402
from radyolo.data.data_splits import DEFAULT_SPLIT_FILE_NAME  # noqa: E402
from radyolo.data.image_labels import (  # noqa: E402
    ImageLabelTaskSpec,
    ImageLabelValue,
    load_case_image_labels,
    load_image_label_task_specs,
    resolve_survival_task_specs,
)
from radyolo.data.tabular import (  # noqa: E402
    load_case_tabular_inputs,
    load_tabular_feature_names,
)
from radyolo.models.layers import ModelOutput  # noqa: E402
from radyolo.models.model import TabularTransformerEncoder  # noqa: E402
from radyolo.training.loss import (  # noqa: E402
    ImageLabelClassificationLoss,
    ImageLabelSurvivalLoss,
)
from radyolo.training.optimizer import build_optimizer  # noqa: E402
from radyolo.utils.util import select_device  # noqa: E402
from radyolo.validation.validate import Validator  # noqa: E402

LOGGER = logging.getLogger(__name__)
TABULAR_IGNORED_SEARCH_KEYS = {"model_scale", "prototype_layer"}
DEFAULT_METRIC_COLUMNS = [
    "metrics/image_label_balanced_accuracy",
    "metrics/image_label_multiclass_roc_auc",
    "metrics/image_label_c_index",
]


class CachedTabularSample(dict[str, object]):
    pass


class CachedTabularDataset(Dataset[CachedTabularSample]):
    def __init__(
        self,
        dataset_path: Path,
        case_names: list[str],
        task_specs: dict[str, ImageLabelTaskSpec],
        tabular_feature_names: tuple[str, ...],
    ) -> None:
        self.dataset_path = dataset_path
        self.task_specs = dict(task_specs)
        self.tabular_feature_names = tuple(tabular_feature_names)
        self.samples: list[CachedTabularSample] = []
        skipped_without_supervision = 0

        for case_name in case_names:
            image_labels = load_case_image_labels(dataset_path, case_name, self.task_specs)
            if not _has_usable_image_label_supervision(image_labels, self.task_specs):
                skipped_without_supervision += 1
                continue

            tabular_values, tabular_mask = load_case_tabular_inputs(
                dataset_path,
                case_name,
                self.tabular_feature_names,
            )
            self.samples.append(
                CachedTabularSample(
                    case_name=case_name,
                    tabular_values=tabular_values,
                    tabular_mask=tabular_mask,
                    image_labels=copy.deepcopy(image_labels),
                )
            )

        if skipped_without_supervision:
            LOGGER.info(
                "Skipped %d/%d cases without usable image-label supervision in %s",
                skipped_without_supervision,
                len(case_names),
                dataset_path,
            )
        if not self.samples:
            msg = f"No supervised tabular samples found in {dataset_path}"
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> CachedTabularSample:
        sample = self.samples[index]
        return CachedTabularSample(
            case_name=sample["case_name"],
            tabular_values=sample["tabular_values"],
            tabular_mask=sample["tabular_mask"],
            image_labels=copy.deepcopy(sample["image_labels"]),
        )


class TabularImageLabelModel(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        *,
        num_features: int,
        task_specs: dict[str, ImageLabelTaskSpec],
        image_label_head_projection_channels: int,
        image_label_head_mlp_hidden_layers: int,
        image_label_head_mlp_hidden_channels: int,
        image_label_head_dropout: float,
        tabular_transformer_d_model: int,
        tabular_transformer_num_heads: int,
        tabular_transformer_num_layers: int,
        tabular_transformer_feedforward_channels: int,
        tabular_transformer_dropout: float,
    ) -> None:
        super().__init__()
        if not task_specs:
            msg = "Tabular image-label model requires at least one image-label task"
            raise ValueError(msg)
        self.image_label_task_specs = dict(task_specs)
        self.tabular_encoder = TabularTransformerEncoder(
            num_features=num_features,
            d_model=tabular_transformer_d_model,
            num_heads=tabular_transformer_num_heads,
            num_layers=tabular_transformer_num_layers,
            feedforward_channels=tabular_transformer_feedforward_channels,
            dropout=tabular_transformer_dropout,
        )
        self.tabular_projection = nn.Sequential(
            nn.Linear(tabular_transformer_d_model, image_label_head_projection_channels),
            nn.SiLU(),
            nn.Dropout(p=image_label_head_dropout),
        )

        mlp_input_width = image_label_head_projection_channels
        mlp_layers: list[nn.Module] = []
        for _ in range(image_label_head_mlp_hidden_layers):
            mlp_layers.extend(
                [
                    nn.Linear(mlp_input_width, image_label_head_mlp_hidden_channels),
                    nn.SiLU(),
                    nn.Dropout(p=image_label_head_dropout),
                ]
            )
            mlp_input_width = image_label_head_mlp_hidden_channels
        self.shared_mlp = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()

        self.task_heads = nn.ModuleDict()
        for task_name, task_spec in self.image_label_task_specs.items():
            if task_spec.type == "multiclass":
                if task_spec.classes is None:
                    msg = f"Multiclass task '{task_name}' requires declared classes"
                    raise ValueError(msg)
                output_width = len(task_spec.classes)
            elif task_spec.type == "survival":
                if not task_spec.survival_bin_thresholds:
                    msg = f"Survival task '{task_name}' requires resolved bin thresholds"
                    raise ValueError(msg)
                output_width = len(task_spec.survival_bin_thresholds)
            else:
                msg = f"Unsupported image-label task type '{task_spec.type}'"
                raise ValueError(msg)
            self.task_heads[task_name] = nn.Linear(mlp_input_width, output_width)

    def forward(self, tabular_values: torch.Tensor, tabular_mask: torch.Tensor) -> ModelOutput:
        encoded = self.tabular_encoder(
            tabular_values.to(dtype=torch.float32),
            tabular_mask.to(dtype=torch.bool),
        )
        shared_features = self.shared_mlp(self.tabular_projection(encoded))
        return ModelOutput(
            image_label_outputs={
                task_name: task_head(shared_features)
                for task_name, task_head in self.task_heads.items()
            }
        )


class TabularImageLabelLoss:
    def __init__(
        self,
        config: DefaultConfig,
        task_specs: dict[str, ImageLabelTaskSpec],
        device: torch.device,
    ) -> None:
        self.classification_loss = ImageLabelClassificationLoss(
            device=device,
            image_label_tasks=task_specs,
            image_label_multiclass_cce_gain=config.image_label_multiclass_cce_gain,
            image_label_multiclass_cce_class_weights=config.image_label_multiclass_cce_class_weights,
        )
        self.survival_loss = ImageLabelSurvivalLoss(
            device=device,
            image_label_tasks=task_specs,
            image_label_survival_mtlr_gain=config.image_label_survival_mtlr_gain,
            image_label_survival_cox_gain=config.image_label_survival_cox_gain,
        )
        self.loss_names = [
            *self.classification_loss.loss_names,
            *self.survival_loss.loss_names,
        ]
        self.num_loss_terms = len(self.loss_names)
        if self.num_loss_terms == 0:
            msg = "No image-label loss terms are enabled for this tabular tuning run"
            raise ValueError(msg)

    def __call__(
        self,
        model_output: ModelOutput,
        batch: dict[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        classification_total, classification_items = self.classification_loss(model_output, batch)  # type: ignore[arg-type]
        survival_total, survival_items = self.survival_loss(model_output, batch)  # type: ignore[arg-type]
        return (
            classification_total + survival_total,
            torch.cat([classification_items, survival_items]),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune only the RadYOLO tabular transformer and image-label output heads. "
            "The input/output JSON files are cached in memory before each trial trains."
        )
    )
    parser.add_argument(
        "tune_config",
        type=Path,
        help="Path to tune_config_tabular_transformer.yaml",
    )
    parser.add_argument(
        "--resume-errored",
        action="store_true",
        help="Resume errored trials when restoring an existing Ray Tune experiment.",
    )
    return parser.parse_args()


def _resolve_path(raw_path: object, *, base_dir: Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _has_usable_image_label_supervision(
    image_labels: Mapping[str, ImageLabelValue],
    task_specs: Mapping[str, ImageLabelTaskSpec],
) -> bool:
    for task_name, task_spec in task_specs.items():
        task_value = image_labels.get(task_name)
        if task_value is None:
            continue
        if task_spec.type == "survival":
            event_field = task_spec.masking
            if not event_field or image_labels.get(event_field) is None:
                continue
        return True
    return False


def _collate_cached_tabular_samples(
    samples: list[CachedTabularSample],
) -> dict[str, object]:
    return {
        "tabular_values": torch.stack(
            [sample["tabular_values"] for sample in samples]  # type: ignore[list-item]
        ),
        "tabular_mask": torch.stack(
            [sample["tabular_mask"] for sample in samples]  # type: ignore[list-item]
        ),
        "image_labels": tuple(sample["image_labels"] for sample in samples),
        "case_names": tuple(str(sample["case_name"]) for sample in samples),
    }


def _move_batch_to_device(
    batch: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    return {
        **batch,
        "tabular_values": batch["tabular_values"].to(device=device, non_blocking=True),  # type: ignore[union-attr]
        "tabular_mask": batch["tabular_mask"].to(device=device, non_blocking=True),  # type: ignore[union-attr]
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_task_specs(config: DefaultConfig) -> dict[str, ImageLabelTaskSpec]:
    task_specs = load_image_label_task_specs(Path(config.data))
    task_specs = resolve_survival_task_specs(
        Path(config.data),
        task_specs,
        num_bins=config.image_label_survival_num_bins,
    )
    if not task_specs:
        msg = f"Dataset {config.data} does not declare image_label_tasks"
        raise ValueError(msg)
    return task_specs


def _build_cached_datasets(
    config: DefaultConfig,
    task_specs: dict[str, ImageLabelTaskSpec],
    tabular_feature_names: tuple[str, ...],
) -> tuple[CachedTabularDataset, CachedTabularDataset]:
    dataset_path = Path(config.data)
    train_cases = _load_split_cases(
        dataset_path,
        config.fold,
        "train",
        config.split_file_name or DEFAULT_SPLIT_FILE_NAME,
    )
    val_cases = _load_split_cases(
        dataset_path,
        config.fold,
        "val",
        config.split_file_name or DEFAULT_SPLIT_FILE_NAME,
    )
    if not val_cases:
        msg = f"Fold {config.fold} has no validation cases in {config.split_file_name}"
        raise ValueError(msg)

    train_dataset = CachedTabularDataset(
        dataset_path,
        train_cases,
        task_specs,
        tabular_feature_names,
    )
    val_dataset = CachedTabularDataset(
        dataset_path,
        val_cases,
        task_specs,
        tabular_feature_names,
    )
    LOGGER.info(
        "Cached tabular dataset %s: train=%d val=%d features=%d tasks=%d",
        dataset_path,
        len(train_dataset),
        len(val_dataset),
        len(tabular_feature_names),
        len(task_specs),
    )
    return train_dataset, val_dataset


def _build_dataloader(
    dataset: Dataset[CachedTabularSample],
    config: DefaultConfig,
    *,
    shuffle: bool,
    device: torch.device,
    dataloader_workers: int,
) -> DataLoader[dict[str, object]]:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=dataloader_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=dataloader_workers > 0,
        collate_fn=_collate_cached_tabular_samples,
    )


def _build_model(
    config: DefaultConfig,
    task_specs: dict[str, ImageLabelTaskSpec],
    tabular_feature_names: tuple[str, ...],
) -> TabularImageLabelModel:
    if not tabular_feature_names:
        msg = f"Dataset {config.data} does not declare tabular_feature_names"
        raise ValueError(msg)
    return TabularImageLabelModel(
        num_features=len(tabular_feature_names),
        task_specs=task_specs,
        image_label_head_projection_channels=config.image_label_head_projection_channels,
        image_label_head_mlp_hidden_layers=config.image_label_head_mlp_hidden_layers,
        image_label_head_mlp_hidden_channels=config.image_label_head_mlp_hidden_channels,
        image_label_head_dropout=config.image_label_head_dropout,
        tabular_transformer_d_model=config.tabular_transformer_d_model,
        tabular_transformer_num_heads=config.tabular_transformer_num_heads,
        tabular_transformer_num_layers=config.tabular_transformer_num_layers,
        tabular_transformer_feedforward_channels=config.tabular_transformer_feedforward_channels,
        tabular_transformer_dropout=config.tabular_transformer_dropout,
    )


def _linear_lr_lambda(config: DefaultConfig) -> Callable[[int], float]:
    def lr_factor(epoch: int) -> float:
        return (
            max(1 - epoch / max(config.epochs, 1), 0.0) * (1.0 - config.final_learning_rate_ratio)
            + config.final_learning_rate_ratio
        )

    return lr_factor


def _apply_warmup(
    optimizer: torch.optim.Optimizer,
    config: DefaultConfig,
    *,
    iteration: int,
    num_warmup_iterations: int,
    lr_factor: float,
    accumulate: int,
) -> int:
    if num_warmup_iterations < 0 or iteration > num_warmup_iterations:
        return accumulate

    warmup_iteration_range = [0, num_warmup_iterations]
    warmed_accumulate = max(
        2,
        int(
            np.interp(
                iteration,
                warmup_iteration_range,
                [2, config.nominal_batch_size / max(config.batch_size, 1)],
            ).round()
        ),
    )
    for group_index, param_group in enumerate(optimizer.param_groups):
        initial_lr = float(param_group.get("initial_lr", config.initial_learning_rate))
        param_group["lr"] = float(
            np.interp(
                iteration,
                warmup_iteration_range,
                [config.warmup_bias_lr if group_index == 0 else 0.0, initial_lr * lr_factor],
            )
        )
        if "momentum" in param_group:
            param_group["momentum"] = float(
                np.interp(
                    iteration,
                    warmup_iteration_range,
                    [config.warmup_momentum, config.momentum],
                )
            )
    return warmed_accumulate


def _evaluate_image_labels(
    model: TabularImageLabelModel,
    val_loader: DataLoader[dict[str, object]],
    config: DefaultConfig,
    task_specs: dict[str, ImageLabelTaskSpec],
    device: torch.device,
    save_dir: Path,
) -> dict[str, float]:
    model.eval()
    validator = Validator(
        dataloader=val_loader,
        save_dir=save_dir,
        config=config,
        names={},
    )
    validator.image_label_task_specs = dict(task_specs)
    validator.init_metrics()
    with torch.no_grad():
        for raw_batch in val_loader:
            batch = _move_batch_to_device(raw_batch, device)
            with torch.autocast(
                device_type=device.type,
                enabled=config.amp and device.type == "cuda",
            ):
                model_output = model(
                    batch["tabular_values"],  # type: ignore[arg-type]
                    batch["tabular_mask"],  # type: ignore[arg-type]
                )
            validator.update_image_label_metrics(batch, model_output.image_label_outputs)  # type: ignore[arg-type]
    return validator._get_image_label_stats()  # noqa: SLF001


def _write_results_csv(results_csv_path: Path, history: list[dict[str, float | int]]) -> None:
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(results_csv_path, index=False)


def _dump_config_for_artifact(config: DefaultConfig) -> dict[str, Any]:
    return config.model_dump(mode="json")


def _save_checkpoint(
    path: Path,
    *,
    model: TabularImageLabelModel,
    optimizer: torch.optim.Optimizer,
    config: DefaultConfig,
    epoch: int,
    metrics: Mapping[str, float],
    tabular_feature_names: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_config": _dump_config_for_artifact(config),
            "metrics": dict(metrics),
            "tabular_feature_names": tuple(tabular_feature_names),
            "saved_at": dt.datetime.now(tz=dt.UTC).isoformat(),
        },
        path,
    )


def _is_better(value: float, best_value: float | None, mode: str) -> bool:
    if best_value is None:
        return True
    if mode == "min":
        return value < best_value
    return value > best_value


def _train_tabular_dataset(  # noqa: PLR0915
    config: DefaultConfig,
    *,
    output_dir: Path,
    metrics_cfg: Mapping[str, object],
    metric_name: str,
    metric_mode: str,
    dataloader_workers: int,
) -> tuple[dict[str, float], dict[str, int], float]:
    _seed_everything(int(config.seed))
    if config.use_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    device = select_device(config.device)
    task_specs = _load_task_specs(config)
    tabular_feature_names = load_tabular_feature_names(Path(config.data))
    train_dataset, val_dataset = _build_cached_datasets(config, task_specs, tabular_feature_names)
    train_loader = _build_dataloader(
        train_dataset,
        config,
        shuffle=True,
        device=device,
        dataloader_workers=dataloader_workers,
    )
    val_loader = _build_dataloader(
        val_dataset,
        config,
        shuffle=False,
        device=device,
        dataloader_workers=dataloader_workers,
    )

    model = _build_model(config, task_specs, tabular_feature_names).to(device)
    image_label_loss = TabularImageLabelLoss(config, task_specs, device)
    effective_batch_size = max(config.batch_size, 1)
    base_accumulate = max(round(config.nominal_batch_size / effective_batch_size), 1)
    weight_decay = (
        config.weight_decay * effective_batch_size * base_accumulate / config.nominal_batch_size
    )
    optimizer = build_optimizer(
        model=model,
        name=config.optimizer,
        lr=config.initial_learning_rate,
        momentum=config.momentum,
        decay=weight_decay,
        muon_weight=config.muon_weight,
    )
    lr_lambda = _linear_lr_lambda(config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    num_batches = len(train_loader)
    num_warmup_iterations = (
        max(round(config.warmup_epochs * num_batches), 100) if config.warmup_epochs > 0 else -1
    )
    tracked_metric_columns = list(metrics_cfg["tracked_columns"])
    rolling_window_epochs = int(metrics_cfg.get("rolling_window_epochs", 20))
    failure_metric = float(metrics_cfg.get("failure_metric", 0.0))

    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv_path = output_dir / "results.csv"
    best_checkpoint_path = output_dir / "weights_best.pt"
    last_checkpoint_path = output_dir / "weights_last.pt"
    history: list[dict[str, float | int]] = []
    best_metric_value: float | None = None
    best_epoch = -1
    epochs_without_improvement = 0
    training_start = time.perf_counter()
    last_optimizer_step = 0

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_loss_items = torch.zeros(image_label_loss.num_loss_terms, dtype=torch.float64)
        optimized_batches = 0
        accumulate = base_accumulate

        for batch_index, raw_batch in enumerate(train_loader):
            iteration = batch_index + num_batches * epoch
            accumulate = _apply_warmup(
                optimizer,
                config,
                iteration=iteration,
                num_warmup_iterations=num_warmup_iterations,
                lr_factor=float(lr_lambda(epoch)),
                accumulate=accumulate,
            )
            batch = _move_batch_to_device(raw_batch, device)
            batch = apply_tabular_mask_augmentation(
                batch,
                config.tabular_feature_mask_augmentation_prob,
            )
            with torch.autocast(
                device_type=device.type,
                enabled=config.amp and device.type == "cuda",
            ):
                model_output = model(
                    batch["tabular_values"],  # type: ignore[arg-type]
                    batch["tabular_mask"],  # type: ignore[arg-type]
                )
                loss, loss_items = image_label_loss(model_output, batch)
                scaled_loss = loss / accumulate

            if not scaled_loss.requires_grad:
                continue

            scaler.scale(scaled_loss).backward()
            running_loss += float(loss.detach().cpu())
            running_loss_items += loss_items.detach().double().cpu()
            optimized_batches += 1

            should_step = (
                iteration - last_optimizer_step
            ) >= accumulate or batch_index == num_batches - 1
            if should_step:
                scaler.unscale_(optimizer)
                if config.grad_clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                last_optimizer_step = iteration

        scheduler.step()
        validation_metrics = _evaluate_image_labels(
            model,
            val_loader,
            config,
            task_specs,
            device,
            output_dir,
        )
        current_metric = float(validation_metrics.get(metric_name, failure_metric))
        mean_loss = running_loss / max(optimized_batches, 1)
        mean_loss_items = running_loss_items / max(optimized_batches, 1)
        epoch_row: dict[str, float | int] = {
            "epoch": epoch + 1,
            "train/img_loss": mean_loss,
            "lr/pg0": float(optimizer.param_groups[0]["lr"]),
        }
        for loss_name, loss_value in zip(
            image_label_loss.loss_names,
            mean_loss_items.tolist(),
            strict=True,
        ):
            epoch_row[f"train/{loss_name}"] = float(loss_value)
        epoch_row.update({key: float(value) for key, value in validation_metrics.items()})
        history.append(epoch_row)
        _write_results_csv(results_csv_path, history)

        if _is_better(current_metric, best_metric_value, metric_mode):
            best_metric_value = current_metric
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            if config.save:
                _save_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch + 1,
                    metrics=validation_metrics,
                    tabular_feature_names=tabular_feature_names,
                )
        else:
            epochs_without_improvement += 1

        LOGGER.info(
            "Epoch %d/%d tabular loss %.5f %s %.5f best %.5f at epoch %d",
            epoch + 1,
            config.epochs,
            mean_loss,
            metric_name,
            current_metric,
            float(best_metric_value if best_metric_value is not None else failure_metric),
            best_epoch,
        )
        if config.patience >= 0 and epochs_without_improvement >= config.patience:
            LOGGER.info(
                "Stopping tabular training after %d epochs without improvement",
                epochs_without_improvement,
            )
            break

    training_minutes = (time.perf_counter() - training_start) / 60.0
    if config.save:
        last_metrics = {
            key: float(value)
            for key, value in history[-1].items()
            if isinstance(value, (int, float)) and str(key).startswith("metrics/")
        }
        _save_checkpoint(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=int(history[-1]["epoch"]),
            metrics=last_metrics,
            tabular_feature_names=tabular_feature_names,
        )

    metric_values, metric_best_epochs = _extract_metric_from_csv(
        results_csv_path,
        tracked_metric_columns,
        rolling_window_epochs,
    )
    return metric_values, metric_best_epochs, training_minutes


def _validate_tracked_metric_columns(metrics_cfg: Mapping[str, object]) -> list[str]:
    raw_tracked_columns = metrics_cfg.get("tracked_columns", DEFAULT_METRIC_COLUMNS)
    if not isinstance(raw_tracked_columns, list) or not raw_tracked_columns:
        msg = "metrics.tracked_columns must be a non-empty list"
        raise ValueError(msg)
    tracked_columns: list[str] = []
    seen: set[str] = set()
    for index, column in enumerate(raw_tracked_columns):
        if not isinstance(column, str) or not column.strip():
            msg = f"metrics.tracked_columns[{index}] must be a non-empty string"
            raise ValueError(msg)
        normalized = column.strip()
        if normalized in seen:
            msg = f"metrics.tracked_columns contains duplicate entry: {normalized!r}"
            raise ValueError(msg)
        seen.add(normalized)
        tracked_columns.append(normalized)
    return tracked_columns


def _filter_tabular_search_space(search_space_cfg: dict[str, Any]) -> dict[str, Any]:
    filtered = {
        key: copy.deepcopy(value)
        for key, value in search_space_cfg.items()
        if key not in TABULAR_IGNORED_SEARCH_KEYS
    }
    ignored_keys = sorted(set(search_space_cfg) & TABULAR_IGNORED_SEARCH_KEYS)
    if ignored_keys:
        LOGGER.warning(
            "Ignoring search-space keys that do not apply to tabular-only tuning: %s",
            ", ".join(ignored_keys),
        )
    return filtered


def _validate_tabular_dataset_entries(datasets: object) -> list[dict[str, Any]]:
    if not isinstance(datasets, list) or not datasets:
        msg = "tune_config_tabular_transformer.yaml requires a non-empty 'datasets' list"
        raise ValueError(msg)

    validated: list[dict[str, Any]] = []
    for index, entry in enumerate(datasets, start=1):
        if not isinstance(entry, dict):
            msg = f"Dataset entry #{index} must be a mapping"
            raise TypeError(msg)
        if "data" not in entry:
            msg = f"Dataset entry #{index} is missing required key 'data'"
            raise KeyError(msg)
        raw_overrides = entry.get("overrides", {})
        if raw_overrides is None:
            raw_overrides = {}
        if not isinstance(raw_overrides, dict):
            msg = f"Dataset entry #{index} key 'overrides' must be a mapping"
            raise TypeError(msg)
        validated_entry = copy.deepcopy(entry)
        validated_entry["overrides"] = copy.deepcopy(raw_overrides)
        validated.append(validated_entry)
    return validated


def _build_train_config(
    template_config: dict[str, Any],
    sampled: dict[str, Any],
    common_overrides: Mapping[str, object],
    dataset: Mapping[str, object],
    output_folder_base: Path,
) -> DefaultConfig:
    run_config = _deep_merge(template_config, dict(common_overrides))
    run_config.update(sampled)
    run_config.update(copy.deepcopy(dict(dataset.get("overrides", {}))))
    run_config["data"] = str(Path(str(dataset["data"])).expanduser().resolve())
    run_config["fold"] = int(dataset.get("fold", run_config.get("fold", 0)))
    run_config["split_file_name"] = str(
        dataset.get("split_file_name", run_config.get("split_file_name", DEFAULT_SPLIT_FILE_NAME))
    )
    run_config["output_folder_base"] = str(output_folder_base)
    run_config["experiment_name"] = "train"
    run_config["enable_image_label_head"] = True
    run_config["enable_tabular_image_label_fusion"] = True
    run_config["plots"] = False
    run_config["resume"] = False
    run_config["exist_ok"] = True
    run_config["num_classes"] = max(1, int(run_config.get("num_classes", 1)))
    return DefaultConfig.model_validate(run_config)


def _run_trial(  # noqa: PLR0915
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

    metrics_cfg = tune_settings["metrics"]
    tracked_metric_columns = list(metrics_cfg["tracked_columns"])
    dataset_aggregation = str(metrics_cfg.get("dataset_aggregation", "mean"))
    failure_metric = float(metrics_cfg.get("failure_metric", 0.0))
    metric_name = str(tune_settings["objective_metric"])
    metric_mode = str(tune_settings["objective_mode"])
    dataloader_workers = int(tune_settings.get("dataloader_workers", 0))
    common_overrides = tune_settings.get("common_train_overrides", {})

    dataset_metrics_by_name: dict[str, list[float]] = {
        metric_name_: [] for metric_name_ in tracked_metric_columns
    }
    dataset_best_epochs_by_name: dict[str, list[int]] = {
        metric_name_: [] for metric_name_ in tracked_metric_columns
    }
    report_payload: dict[str, float | int | str] = {
        "trial_id": trial_id,
    }

    for dataset in tune_settings["datasets"]:
        dataset_name = _normalize_name(str(dataset.get("name", Path(str(dataset["data"])).name)))
        dataset_dir = trial_dir / dataset_name
        output_folder_base = dataset_dir / "runs"
        output_dir = output_folder_base / "train"
        config_path = dataset_dir / "train_config.yaml"
        training_minutes_key = f"dataset_training_minutes/{dataset_name}"

        try:
            config = _build_train_config(
                template_config,
                sampled,
                common_overrides,
                dataset,
                output_folder_base,
            )
            dataset_dir.mkdir(parents=True, exist_ok=True)
            with config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(_dump_config_for_artifact(config), handle, sort_keys=False)

            metric_values, metric_best_epochs, training_minutes = _train_tabular_dataset(
                config,
                output_dir=output_dir,
                metrics_cfg=metrics_cfg,
                metric_name=metric_name,
                metric_mode=metric_mode,
                dataloader_workers=dataloader_workers,
            )
            report_payload[f"dataset_failed/{dataset_name}"] = 0
        except Exception:
            LOGGER.exception(
                "Tabular tuning failed for dataset '%s' in trial %s",
                dataset_name,
                trial_id,
            )
            metric_values = dict.fromkeys(tracked_metric_columns, failure_metric)
            metric_best_epochs = dict.fromkeys(tracked_metric_columns, -1)
            training_minutes = 0.0
            report_payload[f"dataset_failed/{dataset_name}"] = 1

        for metric_column in tracked_metric_columns:
            metric_value = float(metric_values.get(metric_column, failure_metric))
            best_epoch = int(metric_best_epochs.get(metric_column, -1))
            if math.isnan(metric_value):
                metric_value = failure_metric
                best_epoch = -1
                report_payload[f"dataset_failed/{dataset_name}"] = 1
            dataset_metrics_by_name[metric_column].append(metric_value)
            dataset_best_epochs_by_name[metric_column].append(best_epoch)
            report_payload[_metric_report_key(f"dataset_metric/{dataset_name}", metric_column)] = (
                metric_value
            )
            report_payload[
                _metric_report_key(f"dataset_best_epoch/{dataset_name}", metric_column)
            ] = best_epoch
        report_payload[training_minutes_key] = float(training_minutes)

    for metric_column in tracked_metric_columns:
        metric_values = dataset_metrics_by_name[metric_column]
        metric_best_epochs = dataset_best_epochs_by_name[metric_column]
        aggregate_metric_value = _aggregate(metric_values, dataset_aggregation)
        report_payload[_metric_report_key("dataset_metric_mean", metric_column)] = float(
            np.mean(metric_values)
        )
        report_payload[_metric_report_key("dataset_metric_std", metric_column)] = float(
            np.std(metric_values)
        )
        report_payload[_metric_report_key("dataset_metric_aggregate", metric_column)] = float(
            aggregate_metric_value
        )
        report_payload[_metric_report_key("dataset_best_epoch_mean", metric_column)] = float(
            np.mean(metric_best_epochs)
        )

    report_payload["num_datasets"] = len(tune_settings["datasets"])
    tune.report(report_payload)


def _get_ray_trainable() -> Callable[..., None]:
    if __name__ != "__main__":
        return _run_trial

    module = importlib.import_module("hyperparameter_tuning.tune_tabular_transformer")
    return module._run_trial  # noqa: SLF001


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
    config_dir = config_path.resolve().parent

    template_path = tune_cfg.get("template_config")
    if template_path is None:
        msg = "tune_config_tabular_transformer.yaml is missing required key 'template_config'"
        raise KeyError(msg)
    template_config_path = _resolve_path(template_path, base_dir=config_dir)
    template_config = _read_yaml(template_config_path)
    effective_template_config = _get_effective_template_config(template_config)

    search_space_cfg = tune_cfg.get("search_space")
    if not isinstance(search_space_cfg, dict) or not search_space_cfg:
        msg = "tune_config_tabular_transformer.yaml requires a non-empty 'search_space' mapping"
        raise ValueError(msg)
    search_space_cfg = _filter_tabular_search_space(search_space_cfg)
    if not search_space_cfg:
        msg = "No tabular-applicable search-space entries remain after filtering"
        raise ValueError(msg)

    param_space, cast_hints = _build_param_space(search_space_cfg)
    initial_evaluation_point = _build_initial_evaluation_point(
        template_config,
        effective_template_config,
        search_space_cfg,
        cast_hints,
    )

    output_root = _resolve_path(
        tune_cfg.get("output_root", "hyperparameter_tuning/runs"),
        base_dir=PROJECT_ROOT,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    experiment_name = str(
        tune_cfg.get(
            "experiment_name",
            f"tune_tabular_{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d_%H%M%S')}",
        )
    )

    metrics_cfg = tune_cfg.get("metrics", {})
    if not isinstance(metrics_cfg, dict):
        msg = "'metrics' must be a mapping"
        raise TypeError(msg)
    tracked_metric_columns = _validate_tracked_metric_columns(metrics_cfg)
    metrics_cfg = copy.deepcopy(metrics_cfg)
    metrics_cfg["tracked_columns"] = tracked_metric_columns
    default_objective_metric = _metric_report_key(
        "dataset_metric_aggregate",
        tracked_metric_columns[0],
    )

    tune_options = tune_cfg.get("tune", {})
    if not isinstance(tune_options, dict):
        msg = "'tune' must be a mapping"
        raise TypeError(msg)
    raw_objective_metric = str(tune_options.get("metric", default_objective_metric))
    objective_metric = (
        raw_objective_metric.removeprefix("dataset_metric_aggregate/")
        if raw_objective_metric.startswith("dataset_metric_aggregate/")
        else raw_objective_metric
    )

    tune_settings = {
        "output_root": str(output_root),
        "experiment_name": experiment_name,
        "datasets": _validate_tabular_dataset_entries(tune_cfg.get("datasets")),
        "metrics": metrics_cfg,
        "default_objective_metric": default_objective_metric,
        "objective_metric": objective_metric,
        "objective_mode": str(tune_options.get("mode", "max")),
        "training_working_directory": str(
            _resolve_training_working_directory(
                tune_cfg.get("training_working_directory"),
                config_path,
            )
        ),
        "common_train_overrides": tune_cfg.get("common_train_overrides", {}),
        "dataloader_workers": int(tune_cfg.get("dataloader_workers", 0)),
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
    resources = tune_options.get("resources", {"cpu": 4, "gpu": 1})
    num_samples = int(tune_options.get("num_samples", 50))
    max_concurrent_trials = int(tune_options.get("max_concurrent_trials", 2))
    use_short_trial_dirnames = bool(tune_options.get("use_short_trial_dirnames", os.name == "nt"))

    metric_name = str(tune_options.get("metric", tune_settings["default_objective_metric"]))
    metric_mode = str(tune_options.get("mode", "max"))

    trainable = tune.with_parameters(
        _get_ray_trainable(),
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
            "Seeding first tabular tuning trial with template defaults: %s",
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
            "No best tabular trial found for metric '%s'. "
            "All trials may have failed before reporting.",
            metric_name,
        )
        return 1

    LOGGER.info("Best trial id: %s", best.metrics.get("trial_id", "n/a"))
    LOGGER.info("Best %s: %.6f", metric_name, float(best.metrics[metric_name]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
