from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field

import torch
from tqdm_loggable.tqdm_logging import tqdm_logging


@dataclass(slots=True)
class TaskAlignedEpochWarningStats:
    """Aggregated warning counters from task-aligned assignment."""

    total_gt_objects: int = 0
    alignment_zero_fallback_objects: int = 0
    boxes_without_anchor: int = 0
    target_assignment_fallback_objects: int = 0


@dataclass(slots=True)
class MaskEpochWarningStats:
    """Aggregated warning counters from mask-target preparation."""

    total_objects: int = 0
    empty_after_downsampling: int = 0
    without_assigned_grid_cells: int = 0


@dataclass(slots=True)
class EpochWarningStats:
    """Combined warning counters collected during a training epoch."""

    tal: TaskAlignedEpochWarningStats = field(default_factory=TaskAlignedEpochWarningStats)
    mask: MaskEpochWarningStats = field(default_factory=MaskEpochWarningStats)

    def as_list(self) -> list[int]:
        """Serialize warning counters to a fixed-size integer list."""
        return [
            self.tal.total_gt_objects,
            self.tal.alignment_zero_fallback_objects,
            self.tal.boxes_without_anchor,
            self.tal.target_assignment_fallback_objects,
            self.mask.total_objects,
            self.mask.empty_after_downsampling,
            self.mask.without_assigned_grid_cells,
        ]

    @classmethod
    def from_list(cls, values: list[int]) -> EpochWarningStats:
        """Deserialize warning counters from the list returned by as_list."""
        if len(values) != 7:
            msg = f"Expected 7 warning-stat values, got {len(values)}."
            raise ValueError(msg)

        return cls(
            tal=TaskAlignedEpochWarningStats(
                total_gt_objects=values[0],
                alignment_zero_fallback_objects=values[1],
                boxes_without_anchor=values[2],
                target_assignment_fallback_objects=values[3],
            ),
            mask=MaskEpochWarningStats(
                total_objects=values[4],
                empty_after_downsampling=values[5],
                without_assigned_grid_cells=values[6],
            ),
        )


def format_epoch_warning_summary(stats: EpochWarningStats) -> str | None:
    """Build a human-readable warning summary for epoch-level counters."""
    warning_parts: list[str] = []

    if stats.tal.alignment_zero_fallback_objects:
        warning_parts.append(
            "TaskAlignedAssinger fallback for "
            f"{stats.tal.alignment_zero_fallback_objects}/{stats.tal.total_gt_objects} "
            "reference objects because they had zero alignment with all predictions"
        )
    if stats.tal.target_assignment_fallback_objects:
        warning_parts.append(
            "TaskAlignedAssinger fallback for "
            f"{stats.tal.target_assignment_fallback_objects}/{stats.tal.total_gt_objects} "
            "reference objects becaue they had no assigned anchor "
            "after assignment conflict resolution"
        )
    if stats.tal.boxes_without_anchor:
        warning_parts.append(
            "no anchors found for "
            f"{stats.tal.boxes_without_anchor}/{stats.tal.total_gt_objects} "
            "reference objects"
        )
    if stats.mask.empty_after_downsampling:
        warning_parts.append(
            "empty references for "
            f"{stats.mask.empty_after_downsampling}/{stats.mask.total_objects} masks "
            "which can happen because the segmentation acts on half the input image resolution "
            "or due to data augmentation"
        )
    if stats.mask.without_assigned_grid_cells:
        warning_parts.append(
            f"{stats.mask.without_assigned_grid_cells}/{stats.mask.total_objects} masks "
            "were skipped because they had no assinged grid cell"
        )

    if not warning_parts:
        return None

    warning_parts.append(
        "TaskAlignedAssinger fallbacks or empty masks can be caused by small or overlapping "
        "reference objects - the model can still learn, but increasing the input resolution "
        "might help to improve it"
    )

    return "; ".join(warning_parts) + "."


def get_debug_batch_size(batch: dict) -> int | None:
    """Infer batch size from known batch fields used by the training loop."""
    image = batch.get("img")
    if isinstance(image, torch.Tensor) and image.ndim > 0:
        return int(image.shape[0])
    case_names = batch.get("case_names")
    if isinstance(case_names, (list, tuple)):
        return len(case_names)
    return None


def get_debug_case_preview(batch: dict, limit: int = 3) -> str | None:
    """Return a short comma-separated preview of case names in a batch."""
    case_names = batch.get("case_names")
    if not isinstance(case_names, (list, tuple)) or not case_names:
        return None
    return ", ".join(str(name) for name in case_names[:limit])


def debug_log_training_event(logger: logging.Logger, event: str, **fields: object) -> None:
    """Log a structured debug event when debug logging is enabled."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    details = ", ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    if details:
        logger.debug("%s: %s", event, details)
        return
    logger.debug("%s", event)


def debug_log_batch_received(
    logger: logging.Logger,
    epoch: int,
    batch_index: int,
    batch: dict,
) -> None:
    """Log metadata for a newly received training batch."""
    debug_log_training_event(
        logger,
        "Received training batch",
        epoch=epoch,
        batch_index=batch_index,
        batch_size=get_debug_batch_size(batch),
        cases=get_debug_case_preview(batch),
    )


def debug_log_forward_start(
    logger: logging.Logger,
    epoch: int,
    batch_index: int,
    batch: dict,
) -> None:
    """Log the start of a model forward pass for a training batch."""
    image = batch.get("img")
    image_shape = tuple(image.shape) if isinstance(image, torch.Tensor) else None
    debug_log_training_event(
        logger,
        "Starting forward pass",
        epoch=epoch,
        batch_index=batch_index,
        image_shape=image_shape,
    )


def debug_log_loss_computed(
    logger: logging.Logger,
    epoch: int,
    batch_index: int,
    loss: torch.Tensor,
) -> None:
    """Log the computed loss value for a training batch."""
    loss_value: float | str
    if loss.numel() == 1:
        loss_value = float(loss.detach().float().cpu().item())
    else:
        loss_value = f"tensor(shape={tuple(loss.shape)})"
    debug_log_training_event(
        logger,
        "Computed training loss",
        epoch=epoch,
        batch_index=batch_index,
        loss=loss_value,
    )


def debug_log_backward_start(logger: logging.Logger, epoch: int, batch_index: int) -> None:
    """Log the start of backpropagation for the current batch."""
    debug_log_training_event(
        logger,
        "Starting backward pass",
        epoch=epoch,
        batch_index=batch_index,
    )


def debug_log_weight_update_start(logger: logging.Logger, epoch: int, batch_index: int) -> None:
    """Log the start of optimizer parameter updates for the current batch."""
    debug_log_training_event(
        logger,
        "Starting optimizer step",
        epoch=epoch,
        batch_index=batch_index,
    )


def configure_tqdm_logging(
    *,
    level: int = logging.INFO,
    log_rate: datetime.timedelta | None = None,
    stream_handler_factory: callable | None = None,
) -> logging.Logger:
    """Configure tqdm_loggable handlers and return its logger."""
    tqdm_logging.set_level(level)
    tqdm_logging.set_log_rate(log_rate or datetime.timedelta(minutes=1))

    tqdm_logger = logging.getLogger("tqdm_loggable.tqdm_logging")
    tqdm_logger.handlers.clear()
    handler_factory = stream_handler_factory or logging.StreamHandler
    tqdm_handler = handler_factory()
    tqdm_handler.setLevel(level)
    tqdm_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-9s %(message)s",
            datefmt="[%H:%M:%S%z]",
        )
    )
    tqdm_logger.addHandler(tqdm_handler)
    tqdm_logger.setLevel(level)
    return tqdm_logger
