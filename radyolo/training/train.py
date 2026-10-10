from __future__ import annotations

import datetime
import gc
import logging
import os
import shutil
import socket
import sys
import time
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm_loggable
import tqdm_loggable.utils
from torch.utils.data.distributed import DistributedSampler
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm_loggable.auto import tqdm

from radyolo.config.config import (
    ModelCheckpoint,
    get_dataset_config,
    get_training_config,
    load_model_checkpoint,
    save_model_checkpoint,
)
from radyolo.data.data import (
    BackgroundUndersamplingSampler,
    PatchWiseDataset,
    WholeImageDataset,
    apply_configured_augmentations_to_batch,
    build_dataloader,
    pad_batch_to_common_shape,
    preprocess_batch_images,
)
from radyolo.models.model import (
    HardcodedModel,
    build_model_from_config,
    get_model_min_stride,
)
from radyolo.training.loss import build_detection_loss_from_config
from radyolo.training.optimizer import build_optimizer
from radyolo.utils.logging_utils import (
    EpochWarningStats,
    configure_tqdm_logging,
    debug_log_backward_start,
    debug_log_batch_received,
    debug_log_forward_start,
    debug_log_loss_computed,
    debug_log_weight_update_start,
    format_epoch_warning_summary,
)
from radyolo.utils.torch_utils import EarlyStopping
from radyolo.utils.util import (
    ModelEMA,
    init_seeds,
    parse_cuda_device_ids,
    select_device,
    wrap_model_for_training,
)
from radyolo.validation.nms_optimization import run_nms_threshold_optimization
from radyolo.validation.plotting import plot_results
from radyolo.validation.validate import Validator

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.data import DataLoader

    from radyolo.models.layers import ModelOutput
    from radyolo.training.loss import DetectionSegmentationLoss


LOGGER = logging.getLogger(__name__)
RADYOLO_LOGGER_NAME = "radyolo"
MANAGED_LOGGING_HANDLER_ATTR = "_radyolo_managed_handler"

DDP_LOCAL_RANK_ENV = "RADYOLO_DDP_LOCAL_RANK"
DDP_RANK_ENV = "RADYOLO_DDP_RANK"
DDP_WORLD_SIZE_ENV = "RADYOLO_DDP_WORLD_SIZE"
LOG_LEVEL_ENV = "RADYOLO_LOG_LEVEL"
LOG_FILE_ENV = "RADYOLO_LOG_FILE"


def _find_free_port() -> int:
    """Return an available localhost TCP port for distributed startup."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _select_distributed_backend() -> str:
    """Prefer NCCL when available, otherwise fall back to Gloo."""
    if dist.is_nccl_available():
        return "nccl"
    return "gloo"


def _distributed_train_worker(
    local_rank: int,
    world_size: int,
    config_path: str,
    master_addr: str,
    master_port: int,
) -> None:
    """Initialize worker process environment and run one trainer instance."""
    os.environ[DDP_LOCAL_RANK_ENV] = str(local_rank)
    os.environ[DDP_RANK_ENV] = str(local_rank)
    os.environ[DDP_WORLD_SIZE_ENV] = str(world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    _configure_process_logging()
    LOGGER.info(
        "Started distributed train worker rank=%d/%d with master=%s:%d for config=%s.",
        local_rank,
        world_size,
        master_addr,
        master_port,
        config_path,
    )

    trainer = Trainer(cfg=config_path)
    trainer.train()


class StdoutFilter(logging.Filter):
    """Allow only log records below warning level on stdout."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.WARNING


def _mark_managed_handler(handler: logging.Handler) -> logging.Handler:
    """Tag a logging handler so this module can safely manage its lifecycle."""
    setattr(handler, MANAGED_LOGGING_HANDLER_ATTR, True)
    return handler


def _close_managed_handlers(logger: logging.Logger) -> None:
    """Remove and close handlers previously tagged as managed."""
    for handler in list(logger.handlers):
        if not getattr(handler, MANAGED_LOGGING_HANDLER_ATTR, False):
            continue
        logger.removeHandler(handler)
        handler.close()


def _has_managed_handlers(logger: logging.Logger) -> bool:
    """Check whether the logger currently has managed handlers attached."""
    return any(getattr(handler, MANAGED_LOGGING_HANDLER_ATTR, False) for handler in logger.handlers)


def _get_log_rank_context(rank: int | None = None, world_size: int | None = None) -> str:
    """Build rank prefix for distributed logs, or empty string for single-process runs."""
    if rank is None:
        rank = int(os.environ.get(DDP_RANK_ENV, "0"))
    if world_size is None:
        world_size = int(os.environ.get(DDP_WORLD_SIZE_ENV, "1"))
    if world_size <= 1:
        return ""
    return f"[rank {rank}/{world_size}] "


def _resolve_process_log_level() -> int:
    """Resolve effective process log level from environment settings."""
    configured_level = os.environ.get(LOG_LEVEL_ENV)
    if configured_level is None:
        return logging.INFO
    if configured_level.isdigit():
        return int(configured_level)
    return logging.getLevelNamesMapping().get(configured_level.upper(), logging.INFO)


def _configure_process_logging() -> None:
    """Configure rank-aware stdout/stderr logging handlers for this process."""
    level = _resolve_process_log_level()
    simple_formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-9s %(spacer)s: %(rank_context)s%(message)s",
        datefmt="[%H:%M:%S%z]",
        defaults={
            "spacer": " " * 11,
            "rank_context": _get_log_rank_context(),
        },
    )

    radyolo_logger = logging.getLogger(RADYOLO_LOGGER_NAME)
    if _has_managed_handlers(radyolo_logger):
        _close_managed_handlers(radyolo_logger)

    out_handler = _mark_managed_handler(logging.StreamHandler(sys.stdout))
    out_handler.setLevel(level)
    out_handler.addFilter(StdoutFilter())
    out_handler.setFormatter(simple_formatter)
    radyolo_logger.addHandler(out_handler)

    err_handler = _mark_managed_handler(logging.StreamHandler(sys.stderr))
    err_handler.setLevel(logging.WARNING)
    err_handler.setFormatter(simple_formatter)
    radyolo_logger.addHandler(err_handler)

    radyolo_logger.setLevel(logging.DEBUG)
    configure_tqdm_logging(
        level=logging.INFO,
        stream_handler_factory=lambda: logging.StreamHandler(sys.stdout),
    )


class Trainer:
    """Coordinate end-to-end model training, validation, checkpointing, and logging."""

    def __init__(self, cfg: str | Path) -> None:
        """Load training config and initialize runtime/distributed state."""
        self.config_path = Path(cfg)
        self.config = get_training_config(cfg, {})
        self.in_distributed_worker = DDP_LOCAL_RANK_ENV in os.environ
        self.local_rank = int(os.environ.get(DDP_LOCAL_RANK_ENV, "0"))
        self.rank = int(os.environ.get(DDP_RANK_ENV, "0"))
        self.world_size = int(os.environ.get(DDP_WORLD_SIZE_ENV, "1"))
        output_folder_base = Path(self.config.output_folder_base or "runs")
        output_folder_base.mkdir(parents=True, exist_ok=True)

        self.save_dir = output_folder_base / str(self.config.experiment_name)
        self.save_dir.mkdir(parents=True, exist_ok=self.config.exist_ok)
        self.last_model_path = self.save_dir / "weights_last.pt"
        self.best_model_path = self.save_dir / "weights_best.pt"
        self.save_period = self.config.save_period
        self.csv_path = self.save_dir / "results.csv"

        self.check_resume()

        self.device = select_device(self.config.device)
        self.cuda_device_ids = (
            parse_cuda_device_ids(self.config.device) if self.device.type == "cuda" else []
        )
        self.configured_world_size = max(len(self.cuda_device_ids), 1)
        self.should_launch_distributed = (
            self.configured_world_size > 1 and not self.in_distributed_worker
        )
        if not self.in_distributed_worker:
            self.local_rank = 0
            self.rank = 0
            self.world_size = 1
        self.distributed = self.in_distributed_worker
        self.is_main_process = self.rank == 0
        if self.distributed:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        if self.config.use_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        if not self.should_launch_distributed and (self.is_main_process or self.distributed):
            self._configure_file_logging()

        if self.distributed:
            self._setup_distributed_process_group()

        init_seeds(self.config.seed)

        self.epochs = self.config.epochs or 100
        if self.device.type in {"cpu", "mps"}:
            self.config.workers = 0

        self.model: torch.nn.Module | None = None
        self.loss: DetectionSegmentationLoss | None = None
        self.data = get_dataset_config(self.config.data)

        self.lf: Callable[[int], float] | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.loss_items = torch.ones((0,))

        # Stateful validation fitness: exponentially weighted metric aggregate across epochs.
        self.ewa_fitness = None
        self.ewa_alpha = 0.6
        self.epoch_peak_memory_gb = 0.0

    def _log_epoch_warning_summary(self) -> None:
        """Aggregate and log per-epoch warning counters across distributed workers."""
        if self.loss is None:
            return

        warning_stats = self.loss.pop_epoch_warning_stats()
        warning_counts = torch.tensor(
            warning_stats.as_list(),
            device=self.device,
            dtype=torch.int64,
        )
        if dist.is_available() and dist.is_initialized():
            dist.reduce(warning_counts, dst=0, op=dist.ReduceOp.SUM)

        if not self.is_main_process:
            return

        reduced_warning_stats = EpochWarningStats.from_list(
            [int(value) for value in warning_counts.tolist()]
        )
        summary = format_epoch_warning_summary(reduced_warning_stats)
        if summary is not None:
            LOGGER.warning(summary)

    def _configure_file_logging(self) -> None:
        """Attach a per-run file logger and store its path in the environment."""
        time_now = datetime.datetime.now(tz=datetime.UTC)
        radyolo_logger = logging.getLogger(RADYOLO_LOGGER_NAME)
        log_file_path = self._get_log_file_path(time_now)
        os.environ[LOG_FILE_ENV] = str(log_file_path)
        for handler in list(radyolo_logger.handlers):
            if not (
                getattr(handler, MANAGED_LOGGING_HANDLER_ATTR, False)
                and isinstance(handler, logging.FileHandler)
            ):
                continue
            radyolo_logger.removeHandler(handler)
            handler.close()
        file_handler = logging.FileHandler(
            str(log_file_path),
            encoding="utf-8",
        )
        _mark_managed_handler(file_handler)
        file_handler.setLevel(_resolve_process_log_level())
        file_handler.setFormatter(
            logging.Formatter(
                fmt=(
                    "%(asctime)s %(levelname)- 9s %(module)30.30s:"
                    "%(funcName)-14.14s:%(lineno)- 4d: %(rank_context)s%(message)s"
                ),
                datefmt="[%H:%M:%S%z]",
                defaults={"rank_context": _get_log_rank_context(self.rank, self.world_size)},
            )
        )
        radyolo_logger.addHandler(file_handler)
        radyolo_logger.setLevel(logging.DEBUG)
        LOGGER.info("Experiment folder: %s", self.save_dir.resolve())
        LOGGER.info("Input config path: %s", self.config_path.resolve())
        LOGGER.info("Dataset path: %s", Path(self.config.data).resolve())

    def _get_log_file_path(self, time_now: datetime.datetime) -> Path:
        """Build the log file path for this run and rank."""
        log_stem = time_now.strftime("%Y_%m_%d-%H_%M_%S")
        if self.distributed:
            log_stem = f"{log_stem}-rank{self.rank}"
        return self.save_dir / f"{log_stem}.log"

    def _setup_distributed_process_group(self) -> None:
        """Initialize the torch distributed process group for this worker."""
        if dist.is_initialized():
            return
        backend = _select_distributed_backend()
        LOGGER.info(
            (
                "Rank %d initializing distributed process group with backend=%s, "
                "world_size=%d, local_rank=%d, master=%s:%s."
            ),
            self.rank,
            backend,
            self.world_size,
            self.local_rank,
            os.environ.get("MASTER_ADDR", "<unset>"),
            os.environ.get("MASTER_PORT", "<unset>"),
        )
        dist.init_process_group(
            backend=backend,
            rank=self.rank,
            world_size=self.world_size,
            timeout=datetime.timedelta(minutes=30),
            device_id=self.local_rank,
        )
        LOGGER.info("Rank %d finished distributed process group initialization.", self.rank)

    def _teardown_distributed_process_group(self) -> None:
        """Tear down distributed state when training ends."""
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()

    def _validate_distributed_setup(self) -> None:
        """Validate that configured devices support distributed training."""
        if self.device.type != "cuda":
            msg = "DistributedDataParallel requires explicit CUDA devices, e.g. device='0,1'."
            raise ValueError(msg)

    def _launch_distributed_training(self) -> None:
        """Spawn one training worker per configured CUDA device."""
        self._validate_distributed_setup()
        master_addr = "127.0.0.1"
        master_port = _find_free_port()
        os.environ.setdefault(LOG_LEVEL_ENV, str(_resolve_process_log_level()))
        LOGGER.info(
            "Launching distributed training with %d processes via backend=%s at %s:%d.",
            self.configured_world_size,
            _select_distributed_backend(),
            master_addr,
            master_port,
        )
        mp.spawn(
            _distributed_train_worker,
            args=(
                self.configured_world_size,
                str(self.config_path),
                master_addr,
                master_port,
            ),
            nprocs=self.configured_world_size,
            join=True,
        )

    def _get_per_process_batch_size(self) -> int:
        return self.config.batch_size

    def _get_per_process_workers(self) -> int:
        return self.config.workers

    def _get_effective_batch_size(self) -> int:
        if not self.distributed:
            return self.config.batch_size
        return self.config.batch_size * self.world_size

    def _barrier(self) -> None:
        """Synchronize all workers when distributed training is active."""
        if self.distributed and dist.is_initialized():
            dist.barrier()

    def _get_distributed_collective_device(self) -> torch.device:
        """Return the device used for distributed collective control tensors."""
        if not self.distributed:
            return torch.device("cpu")
        backend = dist.get_backend() if dist.is_initialized() else _select_distributed_backend()
        if backend == "nccl":
            return self.device
        return torch.device("cpu")

    def _broadcast_stop(self, stop: bool) -> bool:
        """Broadcast stop decision from rank 0 to all workers."""
        if not self.distributed:
            return stop
        stop_tensor = torch.tensor(
            int(stop),
            device=self._get_distributed_collective_device(),
        )
        dist.broadcast(stop_tensor, src=0)
        return bool(stop_tensor.item())

    def _distributed_any(self, value: bool) -> bool:
        """Return whether any worker reports True for the given boolean flag."""
        if not self.distributed or not dist.is_initialized():
            return value
        value_tensor = torch.tensor(
            int(value),
            device=self._get_distributed_collective_device(),
        )
        dist.all_reduce(value_tensor, op=dist.ReduceOp.MAX)
        return bool(value_tensor.item())

    def _build_distributed_safe_zero_loss(self, model_output: ModelOutput) -> torch.Tensor:
        """Create a zero-valued surrogate loss that preserves graph connectivity."""
        zero_loss = torch.zeros((), device=self.device)
        if model_output.raw_predictions is not None:
            for tensor in model_output.raw_predictions:
                zero_loss = zero_loss + torch.nan_to_num(tensor).sum() * 0.0
        for tensor in (model_output.mask_coefficients, model_output.prototype_masks):
            if tensor is not None:
                zero_loss = zero_loss + torch.nan_to_num(tensor).sum() * 0.0
        return zero_loss

    def copy_train_config(self) -> None:
        """Copy the active training config into the run directory."""
        shutil.copy2(self.config_path, self.save_dir / "train_config.yaml")

    def _compute_validation_fitness(self, metrics: dict[str, float]) -> float:
        """Combine validation metrics into a single weighted fitness score."""
        map_1095 = float(metrics.get("metrics/mAP10-95", 0.0))
        map_1095_agn = float(metrics.get("metrics/mAP10-95Agn", 0.0))
        dice = float(metrics.get("metrics/dice", 0.0))
        weights = self.config.fitness_metric_weights
        total_weight = sum(weights)
        weighted_metrics = (
            weights[0] * map_1095,
            weights[1] * map_1095_agn,
            weights[2] * dice,
        )
        return sum(weighted_metrics) / total_weight

    def check_resume(self) -> None:
        """Resolve resume behavior and load checkpoint config when available."""
        resume = self.config.resume
        if resume:
            try:
                if not self.last_model_path.exists():
                    self.resume = False
                    return

                checkpoint_config = load_model_checkpoint(self.last_model_path)["train_config"]
                if not Path(checkpoint_config.data).exists():
                    checkpoint_config.data = self.config.data

                resume = True
                self.config = checkpoint_config

            except Exception as e:
                msg = "Resume checkpoint not found, please pass a valid checkpoint to resume from."
                raise FileNotFoundError(msg) from e
        self.resume = resume

    def setup_model(
        self,
        num_classes: int,
    ) -> tuple[ModelCheckpoint | None, HardcodedModel]:
        """Build model and optionally preload weights from the last checkpoint."""
        if isinstance(self.model, HardcodedModel):
            return None, self.model

        checkpoint_path = self.last_model_path
        model_state_dict = None
        checkpoint: ModelCheckpoint | None = None
        if self.resume and checkpoint_path.suffix == ".pt" and checkpoint_path.exists():
            checkpoint = load_model_checkpoint(checkpoint_path)
            model_state_dict = checkpoint["model_state_dict"]

        model = build_model_from_config(
            self.config,
            num_classes=num_classes,
            state_dict=model_state_dict,
        )
        return checkpoint, model

    def _setup_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> tuple[torch.optim.lr_scheduler.LRScheduler, Callable[[int], float]]:
        """Create learning rate scheduler and its lambda policy function."""
        def lf(x: int) -> float:
            return (
                max(1 - x / self.epochs, 0) * (1.0 - self.config.final_learning_rate_ratio)
                + self.config.final_learning_rate_ratio
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lf)
        return scheduler, lf

    def _setup_train(
        self,
    ) -> tuple[
        torch.nn.Module,
        torch.GradScaler,
        torch.utils.data.DataLoader,
        Validator | None,
        ModelEMA,
        int,
        torch.optim.Optimizer,
        bool,
        torch.optim.lr_scheduler.LRScheduler,
        Callable[[int], float],
        EarlyStopping,
        float | None,
        int,
    ]:
        """Initialize all training components and resume state before looping."""
        ckpt, model = self.setup_model(
            self.data.num_classes,
        )
        model = model.to(self.device)
        self.loss = build_detection_loss_from_config(self.config, model, self.device)
        self.loss_names = tuple(self.loss.loss_names)

        # Freeze layers
        freeze_layer_names = [".dfl"]
        for layer_name, layer in model.named_parameters():
            if any(name in layer_name for name in freeze_layer_names):
                LOGGER.info("Freezing layer '%s'", layer_name)
                layer.requires_grad = False
            elif not layer.requires_grad and layer.dtype.is_floating_point:
                LOGGER.warning("setting requires_grad=True for frozen layer %s.", layer_name)
                layer.requires_grad = True

        # Check AMP
        scaler = torch.GradScaler("cuda", enabled=self.config.amp)

        # Dataloaders
        train_loader = self.get_dataloader(
            self.config.data, model, fold=self.config.fold, mode="train"
        )
        if self.config.val and (not self.distributed or self.is_main_process):
            val_loader = self.get_dataloader(
                self.config.data, model, fold=self.config.fold, mode="val"
            )
            validator = self.get_validator(
                val_loader,
                self.save_dir,
                self.data.names,
            )
        else:
            val_loader = None
            validator = None
        ema = ModelEMA(model, decay=self.config.ema_decay, tau=self.config.ema_tau)
        model = wrap_model_for_training(
            model,
            distributed=self.distributed,
            rank=self.rank,
            local_rank=self.local_rank,
        )

        # Optimizer
        effective_batch_size = self._get_effective_batch_size()
        accumulate = max(round(self.config.nominal_batch_size / effective_batch_size), 1)
        weight_decay = (
            self.config.weight_decay
            * effective_batch_size
            * accumulate
            / self.config.nominal_batch_size
        )
        optimizer = build_optimizer(
            model=model,
            name=self.config.optimizer,
            lr=self.config.initial_learning_rate,
            momentum=self.config.momentum,
            decay=weight_decay,
            muon_weight=self.config.muon_weight,
        )

        # Scheduler
        scheduler, lf = self._setup_scheduler(optimizer)
        stopper, stop = EarlyStopping(patience=self.config.patience), False
        best_fitness, start_epoch = self.resume_training(ckpt, optimizer, scaler, ema)
        scheduler.last_epoch = start_epoch - 1
        return (
            model,
            scaler,
            train_loader,
            validator,
            ema,
            accumulate,
            optimizer,
            stop,
            scheduler,
            lf,
            stopper,
            best_fitness,
            start_epoch,
        )

    def train(self) -> None:
        """Run training entrypoint including optional distributed launch."""
        try:
            if self.should_launch_distributed:
                self._launch_distributed_training()
                return

            if self.is_main_process:
                self.copy_train_config()
            (
                model,
                scaler,
                train_loader,
                validator,
                ema,
                accumulate,
                optimizer,
                stop,
                scheduler,
                lf,
                stopper,
                best_fitness,
                start_epoch,
            ) = self._setup_train()

            self._train_loop(
                model=model,
                train_loader=train_loader,
                validator=validator,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                lf=lf,
                ema=ema,
                accumulate=accumulate,
                stopper=stopper,
                stop=stop,
                best_fitness=best_fitness,
                start_epoch=start_epoch,
            )
        finally:
            self._teardown_distributed_process_group()

    def _train_loop(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        validator: Validator | None,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: torch.amp.grad_scaler.GradScaler,
        lf: Callable[[int], float],
        ema: ModelEMA,
        accumulate: int,
        stopper: EarlyStopping,
        stop: bool,
        best_fitness: float | None,
        start_epoch: int,
    ) -> None:
        """Execute epoch/batch optimization loop with logging and validation."""
        num_batches = len(train_loader)
        num_warmup_iterations = (
            max(round(self.config.warmup_epochs * num_batches), 100)
            if self.config.warmup_epochs > 0
            else -1
        )
        train_time_start = time.time()
        self.epoch = start_epoch
        last_opt_step = num_batches * self.epoch
        optimizer.zero_grad()
        while True:
            model.train()
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(self.epoch)
            timing_values = self._create_epoch_timing_values()
            self._reset_epoch_peak_memory()
            self.loss.reset_epoch_warning_stats()

            if self.is_main_process:
                LOGGER.info(
                    "%13s" * (2 + len(self.loss_names)),
                    "Epoch",
                    "GPU_mem",
                    *self.loss_names,
                )
                pbar = tqdm(
                    enumerate(train_loader),
                    total=num_batches,
                    bar_format="{l_bar}{bar:10}{r_bar}",
                    smoothing=0.05,
                )
            else:
                pbar = enumerate(train_loader)
            tloss: torch.Tensor | None = torch.zeros(self.loss.num_loss_terms, dtype=torch.float64)
            loss_length: int = 1
            loss: torch.Tensor | None = None
            previous_gpu_steps_end: float | None = None
            has_previous_batch_in_epoch = False
            with (
                logging_redirect_tqdm(loggers=[logging.root, logging.getLogger("radyolo")])
                if self.is_main_process and tqdm_loggable.utils.is_interactive_session()
                else nullcontext()
            ):
                for i, batch in pbar:
                    debug_log_batch_received(LOGGER, self.epoch + 1, i, batch)
                    run_gpu_augmentations = self.config.uses_gpu_augmentations()
                    # Warmup
                    num_iterations = i + num_batches * self.epoch
                    if num_iterations <= num_warmup_iterations:
                        warmup_iteration_range = [0, num_warmup_iterations]
                        accumulate = max(
                            2,
                            int(
                                np.interp(
                                    num_iterations,
                                    warmup_iteration_range,
                                    [
                                        2,
                                        self.config.nominal_batch_size
                                        / self._get_effective_batch_size(),
                                    ],
                                ).round()
                            ),
                        )
                        for j, x in enumerate(optimizer.param_groups):
                            x["lr"] = np.interp(
                                num_iterations,
                                warmup_iteration_range,
                                [
                                    self.config.warmup_bias_lr if j == 0 else 0.0,
                                    x["initial_lr"] * lf(self.epoch),
                                ],
                            )
                            if "momentum" in x:
                                x["momentum"] = np.interp(
                                    num_iterations,
                                    warmup_iteration_range,
                                    [self.config.warmup_momentum, self.config.momentum],
                                )

                    all_gpu_steps_start = time.perf_counter()
                    if has_previous_batch_in_epoch and previous_gpu_steps_end is not None:
                        timing_values["timing/post_gpu_to_next_gpu_steps_s"] += (
                            all_gpu_steps_start - previous_gpu_steps_end
                        )
                    # Forward
                    with torch.autocast(device_type=self.device.type, enabled=self.config.amp):
                        prepared_batch = self._prepare_training_batch(
                            batch,
                            run_gpu_augmentations,
                        )
                        debug_log_forward_start(LOGGER, self.epoch + 1, i, prepared_batch)
                        skip_due_to_non_finite_input = not torch.isfinite(
                            prepared_batch["img"]
                        ).all()
                        if skip_due_to_non_finite_input:
                            self._log_non_finite_input_batch(prepared_batch)
                        if self._distributed_any(skip_due_to_non_finite_input):
                            continue
                        raw_predictions = model(prepared_batch["img"])
                        if isinstance(raw_predictions, torch.Tensor):
                            raise TypeError(raw_predictions)

                        loss, self.loss_items = cast(
                            "tuple[torch.Tensor, torch.Tensor]",
                            self.loss(
                                raw_predictions,
                                prepared_batch,
                                current_epoch=self.epoch + 1,
                            ),
                        )
                        debug_log_loss_computed(LOGGER, self.epoch + 1, i, loss)

                        skip_due_to_non_finite_loss = (
                            not torch.isfinite(loss) or not torch.isfinite(self.loss_items).all()
                        )
                        if skip_due_to_non_finite_loss:
                            self._log_non_finite_training_batch(
                                batch=prepared_batch,
                                loss=loss,
                                loss_items=self.loss_items,
                            )
                        if self._distributed_any(skip_due_to_non_finite_loss):
                            optimizer.zero_grad()
                            if self.distributed:
                                surrogate_loss = self._build_distributed_safe_zero_loss(
                                    raw_predictions
                                )
                                scaler.scale(surrogate_loss).backward()
                            continue

                        tloss = (
                            tloss * i + self.loss_items.to(dtype=tloss.dtype, device=tloss.device)
                        ) / (i + 1)

                    # Backward
                    debug_log_backward_start(LOGGER, self.epoch + 1, i)
                    scaler.scale(loss).backward()

                    # Optimize
                    if num_iterations - last_opt_step >= accumulate:
                        debug_log_weight_update_start(LOGGER, self.epoch + 1, i)
                        self.optimizer_step(
                            scaler,
                            optimizer,
                            model,
                            ema,
                            self.config.grad_clip_max_norm,
                        )
                        last_opt_step = num_iterations

                    gpu_steps_end = time.perf_counter()
                    timing_values["timing/gpu_steps_total_s"] += gpu_steps_end - all_gpu_steps_start
                    previous_gpu_steps_end = gpu_steps_end
                    has_previous_batch_in_epoch = True
                    self._update_epoch_peak_memory()

                    # Log
                    loss_length = tloss.shape[0] if len(tloss.shape) else 1
                    if self.is_main_process:
                        pbar.set_description(
                            ("%13s%13s" + "%13.3g" * loss_length)
                            % (
                                f"{self.epoch + 1}/{self.epochs}",
                                f"{self.epoch_peak_memory_gb:.3g}G",
                                *(tloss if loss_length > 1 else torch.unsqueeze(tloss, 0)),
                            )
                        )

            ema.advance_updates()
            scheduler.step()
            self._barrier()
            self._log_epoch_warning_summary()

            if self.is_main_process and tloss is not None:
                LOGGER.info(
                    "%13s%13s" + "%13s" * len(self.loss_names),  # noqa: G003
                    "Completed Epoch",
                    "GPU_mem",
                    *self.loss_names,
                )
                LOGGER.info(
                    "%13s%13s" + "%13.3g" * loss_length,  # noqa: G003
                    f"{self.epoch + 1}/{self.epochs}",
                    f"{self.epoch_peak_memory_gb:.3g}G",
                    *(tloss if loss_length > 1 else torch.unsqueeze(tloss, 0)),  # losses
                )

            final_epoch = self.epoch + 1 >= self.epochs

            if self.is_main_process:
                metrics: dict[str, float] = {}
                fitness = 0.0
                if validator is not None and (self.config.val or final_epoch or stop):
                    validation_start = time.perf_counter()
                    metrics, fitness, best_fitness = self.validate(
                        validator=validator,
                        ema=ema,
                        best_fitness=best_fitness,
                    )
                    timing_values["timing/validation_s"] = time.perf_counter() - validation_start

                self.save_metrics(
                    metrics={
                        **(self.label_loss_items(tloss) if tloss is not None else {}),
                        **metrics,
                        "train/peak_memory_gb": float(self.epoch_peak_memory_gb),
                        **timing_values,
                    },
                    train_time_start=train_time_start,
                )

                stop |= stopper(self.epoch + 1, fitness) or final_epoch

                if self.config.save or final_epoch:
                    self.save_model(
                        best_fitness=best_fitness or 0.0,
                        ema=ema,
                        optimizer=optimizer,
                        scaler=scaler,
                        metrics=metrics,
                        fitness=fitness,
                    )

                if self.config.plots:
                    plot_results(self.csv_path)

            stop = self._broadcast_stop(stop)
            self._barrier()

            self._clear_memory()

            if stop:
                break

            self.epoch += 1

        if self.is_main_process:
            seconds = time.time() - train_time_start
            msg = f"\n{self.epoch + 1} epochs completed in {seconds / 3600:.3f} hours."
            LOGGER.info(msg)
            if validator is not None and self.config.val:
                run_nms_threshold_optimization(
                    validator=validator,
                    config=self.config,
                    save_dir=self.save_dir,
                    last_model_path=self.last_model_path,
                    best_model_path=self.best_model_path,
                    device=self.device,
                )

        self._clear_memory()

    @staticmethod
    def _create_epoch_timing_values() -> dict[str, float]:
        """Initialize timing metric accumulators for one epoch."""
        keys = (
            "timing/gpu_steps_total_s",
            "timing/post_gpu_to_next_gpu_steps_s",
            "timing/validation_s",
        )
        return dict.fromkeys(keys, 0.0)

    def _prepare_training_batch(self, batch: dict, run_gpu_augmentations: bool) -> dict:
        """Move batch tensors to target device and apply configured augmentations."""
        preprocess_batch_images(
            batch,
            device=self.device,
            normalization=self.config.normalization,
            target_shape=self.config.patch_size,
            apply_preprocessing=self.config.apply_preprocessing,
            preprocessing_device=self.config.preprocessing_device,
            variable_input_size=self.config.variable_input_size,
        )
        if self.config.segmentation_output:
            batch["masks"] = tuple(
                masks.to(self.device, non_blocking=True) if masks is not None else None
                for masks in batch["masks"]
            )

        if run_gpu_augmentations:
            for key in ("cls", "bboxes", "batch_idx"):
                batch[key] = batch[key].to(self.device, non_blocking=True)
            apply_configured_augmentations_to_batch(
                batch,
                self.config,
                self.config.patch_size,
                placement="gpu",
            )
        if self.config.variable_input_size:
            pad_batch_to_common_shape(batch)
        return batch

    @staticmethod
    def optimizer_step(
        scaler: torch.GradScaler,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        ema: ModelEMA,
        grad_clip_max_norm: float,
    ) -> None:
        """Apply one optimizer step with grad clipping and EMA update."""
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        ema.update(model)

    @staticmethod
    def _has_non_finite_gradients(model: torch.nn.Module) -> bool:
        """Return True when any floating-point gradient contains NaN or Inf."""
        parameters_with_grads = [
            parameter
            for parameter in model.parameters()
            if parameter.grad is not None and parameter.grad.dtype.is_floating_point
        ]
        return any(not torch.isfinite(parameter.grad).all() for parameter in parameters_with_grads)

    @staticmethod
    def _log_non_finite_training_batch(
        batch: dict,
        loss: torch.Tensor,
        loss_items: torch.Tensor,
    ) -> None:
        """Log context for batches skipped due to non-finite training loss."""
        loss_value = (
            loss.detach().float().cpu().item() if loss.numel() == 1 else loss.detach().float().cpu()
        )
        case_names = batch.get("case_names")
        if isinstance(case_names, (list, tuple)):
            case_text = ", ".join(str(name) for name in case_names[:4])
        elif case_names is None:
            case_text = "<unknown>"
        else:
            case_text = str(case_names)
        LOGGER.warning(
            "Skipping batch because training loss became non-finite. loss=%s loss_items=%s "
            "cases=%s",
            loss_value,
            loss_items.detach().float().cpu().tolist(),
            case_text,
        )

    @staticmethod
    def _log_non_finite_input_batch(batch: dict) -> None:
        """Log input tensor diagnostics for batches with non-finite image values."""
        image = batch["img"]
        case_names = batch.get("case_names")
        if isinstance(case_names, (list, tuple)):
            case_text = ", ".join(str(name) for name in case_names[:4])
        elif case_names is None:
            case_text = "<unknown>"
        else:
            case_text = str(case_names)
        nan_count = int(torch.isnan(image).sum().item())
        pos_inf_count = int(torch.isposinf(image).sum().item())
        neg_inf_count = int(torch.isneginf(image).sum().item())
        LOGGER.warning(
            "Skipping batch because input image tensor became non-finite. "
            "nan_count=%s pos_inf_count=%s neg_inf_count=%s cases=%s",
            nan_count,
            pos_inf_count,
            neg_inf_count,
            case_text,
        )

    def _get_memory(self) -> float:
        """Get accelerator memory utilization in GB."""
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
        elif self.device.type == "cpu":
            memory = 0
        else:
            memory = torch.cuda.memory_reserved()
        return memory / 1e9

    def _reset_epoch_peak_memory(self) -> None:
        self.epoch_peak_memory_gb = self._get_memory()

    def _update_epoch_peak_memory(self) -> None:
        self.epoch_peak_memory_gb = max(self.epoch_peak_memory_gb, self._get_memory())

    def _clear_memory(self) -> None:
        """Clear accelerator memory on used platform."""
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()

    def get_dataloader(
        self,
        dataset_path: Path,
        model: torch.nn.Module,
        fold: int = 0,
        mode: Literal["train", "val"] = "train",
    ) -> DataLoader:
        """Build train or validation dataloader with optional distributed/sampling logic."""
        if mode not in {"train", "val"}:
            message = f"get_dataloader() only supports modes train and val, but got {mode}"
            raise ValueError(message)

        dataset_class = (
            PatchWiseDataset if self.config.training_mode == "patchwise" else WholeImageDataset
        )
        foreground_oversampling = 0.0
        background_undersampling = 0.0
        if mode == "train" or self.config.training_mode == "patchwise":
            foreground_oversampling = self.config.foreground_oversampling
        if mode == "train":
            background_undersampling = self.config.background_undersampling

        dataset = dataset_class(
            src_path=dataset_path,
            fold=fold,
            group=mode,
            patch_size=self.config.patch_size,
            augment=mode == "train",
            min_stride=get_model_min_stride(model),
            normalization=self.config.normalization,
            apply_preprocessing=self.config.apply_preprocessing,
            preprocessing_device=self.config.preprocessing_device,
            load_masks=self.config.segmentation_output,
            foreground_oversampling=foreground_oversampling,
            background_undersampling=background_undersampling,
            min_box_crop_fraction=self.config.min_box_crop_fraction,
            config=self.config,
        )
        sampler = None
        shuffle = mode == "train"
        if mode == "train" and background_undersampling > 0.0 and dataset.background_case_indices:
            sampler = BackgroundUndersamplingSampler(
                dataset,
                background_undersampling=background_undersampling,
                shuffle=True,
                seed=int(self.config.seed),
                num_replicas=self.world_size if self.distributed else 1,
                rank=self.rank if self.distributed else 0,
            )
            if len(sampler) == 0:
                msg = (
                    "background_undersampling removed all training samples for the selected split. "
                    "Decrease background_undersampling or ensure the split contains foreground cases."
                )
                raise ValueError(msg)
            shuffle = False
        elif self.distributed and mode == "train":
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            shuffle = False
        return build_dataloader(
            dataset,
            self._get_per_process_batch_size(),
            self._get_per_process_workers(),
            shuffle,
            self.device,
            sampler=sampler,
        )

    def save_model(
        self,
        best_fitness: float,
        ema: ModelEMA,
        optimizer: torch.optim.Optimizer,
        scaler: torch.GradScaler,
        metrics: dict[str, float],
        fitness: float,
    ) -> None:
        """Persist latest checkpoint and optional best/periodic snapshots."""
        is_best = best_fitness == fitness
        save_paths = [self.last_model_path]
        if is_best:
            save_paths.append(self.best_model_path)
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            save_paths.append(self.save_dir / f"epoch{self.epoch}.pt")
        save_model_checkpoint(
            save_paths,
            self.epoch,
            best_fitness,
            ema.ema,
            ema.updates,
            optimizer,
            scaler,
            self.config,
            {
                **metrics,
                "metrics/fitness": float(self.ewa_fitness) if self.ewa_fitness is not None else 0.0,
            },
        )
        if is_best:
            self._save_best_validation_artifacts()

    def _save_best_validation_artifacts(self) -> None:
        """Copy current validation artifacts into *_best snapshots."""
        validation_summary = self.save_dir / "validation_summary.png"
        if validation_summary.exists():
            shutil.copy2(validation_summary, self.save_dir / "validation_summary_best.png")

        validation_samples = self.save_dir / "validation_samples"
        validation_samples_best = self.save_dir / "validation_samples_best"
        if not validation_samples.exists():
            return

        with suppress(FileNotFoundError):
            shutil.rmtree(validation_samples_best)
        shutil.copytree(validation_samples, validation_samples_best)

    def get_validator(
        self,
        test_loader: DataLoader,
        save_dir: Path,
        names: dict[int, str],
    ) -> Validator:
        """Create the validation runner for this training session."""
        return Validator(
            test_loader,
            save_dir=save_dir,
            config=self.config,
            names=names,
        )

    def validate(
        self,
        validator: Validator,
        ema: ModelEMA,
        best_fitness: float | None,
    ) -> tuple[dict[str, float], float, float]:
        """Run validation, update smoothed fitness, and track best fitness."""
        metrics = validator(self, model=cast("HardcodedModel", ema.ema))
        current_fitness = self._compute_validation_fitness(metrics)
        if self.ewa_fitness is None:
            self.ewa_fitness = current_fitness
        else:
            self.ewa_fitness = (
                self.ewa_alpha * float(self.ewa_fitness) + (1.0 - self.ewa_alpha) * current_fitness
            )
        metrics["metrics/fitness"] = float(self.ewa_fitness)
        fitness = float(self.ewa_fitness)
        if not best_fitness or best_fitness < fitness:
            best_fitness = fitness
        return metrics, fitness, best_fitness

    @overload
    def label_loss_items(
        self, loss_items: torch.Tensor, prefix: str = "train"
    ) -> dict[str, float]: ...

    @overload
    def label_loss_items(self, loss_items: None = None, prefix: str = "train") -> list[str]: ...

    def label_loss_items(
        self, loss_items: torch.Tensor | None = None, prefix: str = "train"
    ) -> dict[str, float] | list[str]:
        """Return labeled loss keys or a key-to-value mapping for logging."""
        keys: list[str] = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            rounded_loss_items = [round(float(x.detach()), 5) for x in loss_items]
            return dict(zip(keys, rounded_loss_items, strict=True))
        return keys

    def resume_training(
        self,
        checkpoint: ModelCheckpoint | None,
        optimizer: torch.optim.Optimizer,
        scaler: torch.GradScaler,
        ema: ModelEMA,
        best_fitness: float | None = None,
    ) -> tuple[float | None, int]:
        """Loads the state dicts for optimizer, scaler and model ema to resume training."""
        if checkpoint is None:
            return best_fitness, 0
        start_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_fitness = checkpoint["best_fitness"]
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        ema.ema.load_state_dict(checkpoint["model_state_dict"])
        ema.updates = checkpoint["ema_updates"]
        train_metrics = checkpoint.get("train_metrics", {})
        self.ewa_fitness = train_metrics.get("metrics/fitness")

        LOGGER.info(
            "Resuming training %s from epoch %d to %d total epochs.",
            self.last_model_path,
            start_epoch + 1,
            self.epochs,
        )
        if self.epochs < start_epoch:
            LOGGER.info(
                "%s has been trained for %d epochs. Fine-tuning for %d more epochs.",
                self.model,
                checkpoint["epoch"],
                self.epochs,
            )
        return best_fitness, start_epoch

    def save_metrics(self, metrics: dict, train_time_start: float) -> None:
        """Append one epoch of metrics to the CSV training log."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2
        header = (
            ""
            if self.csv_path.exists()
            else (("%s," * n % ("epoch", "time", *keys)).rstrip(",") + "\n")
        )
        t = time.time() - train_time_start
        try:
            with self.csv_path.open("a", encoding="utf-8") as file:
                line = ("%.6g," * n % (self.epoch + 1, t, *vals)).rstrip(",")
                file.write(header + line + "\n")
        except OSError as exc:
            LOGGER.warning("Could not save metrics CSV %s: %s", self.csv_path, exc)
