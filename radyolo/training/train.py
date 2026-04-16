from __future__ import annotations

import datetime
import gc
import logging
import os
import shutil
import socket
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm_loggable
import tqdm_loggable.utils
import yaml
from torch.utils.data.distributed import DistributedSampler
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm_loggable.auto import tqdm

from radyolo.config.config import (
    DatasetConfig,
    ModelCheckpoint,
    get_cfg,
    load_model_checkpoint,
    save_model_checkpoint,
)
from radyolo.data.data import (
    PatchWiseDataset,
    WholeImageDataset,
    apply_configured_augmentations_to_batch,
    build_dataloader,
)
from radyolo.models.model import HardcodedModel, build_model_from_config, get_model_min_stride
from radyolo.training.loss import build_detection_loss_from_config
from radyolo.utils.torch_utils import EarlyStopping
from radyolo.utils.util import (
    ModelEMA,
    init_seeds,
    parse_cuda_device_ids,
    select_device,
    wrap_model_for_training,
)
from radyolo.validation.plotting import plot_results
from radyolo.validation.validate import Validator

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.data import DataLoader

    from radyolo.training.loss import v8DetectionLoss


LOGGER = logging.getLogger(__name__)
RADYOLO_LOGGER_NAME = "radyolo"
MANAGED_LOGGING_HANDLER_ATTR = "_radyolo_managed_handler"

DDP_LOCAL_RANK_ENV = "RADYOLO_DDP_LOCAL_RANK"
DDP_RANK_ENV = "RADYOLO_DDP_RANK"
DDP_WORLD_SIZE_ENV = "RADYOLO_DDP_WORLD_SIZE"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _select_distributed_backend() -> str:
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
    worker_config = get_cfg(config_path, {})
    _configure_process_logging(worker_config.verbose)
    os.environ[DDP_LOCAL_RANK_ENV] = str(local_rank)
    os.environ[DDP_RANK_ENV] = str(local_rank)
    os.environ[DDP_WORLD_SIZE_ENV] = str(world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
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
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.WARNING


def _mark_managed_handler(handler: logging.Handler) -> logging.Handler:
    setattr(handler, MANAGED_LOGGING_HANDLER_ATTR, True)
    return handler


def _has_managed_handlers(logger: logging.Logger) -> bool:
    return any(getattr(handler, MANAGED_LOGGING_HANDLER_ATTR, False) for handler in logger.handlers)


def _configure_process_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    simple_formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-9s %(spacer)s: %(message)s",
        datefmt="[%H:%M:%S%z]",
        defaults={"spacer": " " * 11},
    )

    radyolo_logger = logging.getLogger(RADYOLO_LOGGER_NAME)
    if not _has_managed_handlers(radyolo_logger):
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


class Trainer:
    def __init__(self, cfg: str | Path) -> None:
        self.config_path = Path(cfg)
        self.config = get_cfg(cfg, {})
        self.in_distributed_worker = DDP_LOCAL_RANK_ENV in os.environ
        self.local_rank = int(os.environ.get(DDP_LOCAL_RANK_ENV, "0"))
        self.rank = int(os.environ.get(DDP_RANK_ENV, "0"))
        self.world_size = int(os.environ.get(DDP_WORLD_SIZE_ENV, "1"))
        output_folder_base = Path(self.config.output_folder_base or "runs")
        output_folder_base.mkdir(parents=True, exist_ok=True)

        self.save_dir = output_folder_base / str(self.config.experiment_name)
        self.wdir = self.save_dir / "weights"
        self.save_dir.mkdir(parents=True, exist_ok=self.config.exist_ok)
        self.wdir.mkdir(parents=True, exist_ok=self.config.exist_ok)
        self.last_model_path = self.wdir / "last.pt"
        self.best_model_path = self.wdir / "best.pt"
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

        if not self.should_launch_distributed and self.is_main_process:
            self._configure_file_logging()

        if self.distributed:
            self._setup_distributed_process_group()

        init_seeds(self.config.seed, deterministic=self.config.deterministic)

        self.epochs = self.config.epochs or 100
        if self.device.type in {"cpu", "mps"}:
            self.config.workers = 0

        self.model: torch.nn.Module | None = None
        self.loss: v8DetectionLoss | None = None
        self.data = self.get_dataset_config(self.config.data)

        self.lf: Callable[[int], float] | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.loss_items = torch.ones((0,))

        # Stateful validation fitness: exponentially weighted metric aggregate across epochs.
        self.ewa_fitness = None
        self.ewa_alpha = 0.6

    def _configure_file_logging(self) -> None:
        time_now = datetime.datetime.now(tz=datetime.UTC)
        radyolo_logger = logging.getLogger(RADYOLO_LOGGER_NAME)
        file_handler = logging.FileHandler(
            str(self.save_dir.joinpath(time_now.strftime("%Y_%m_%d-%H_%M_%S")).with_suffix(".log")),
            encoding="utf-8",
        )
        file_handler.setFormatter(
            logging.Formatter(
                fmt=(
                    "%(asctime)s %(levelname)- 9s %(module)30.30s:"
                    "%(funcName)-14.14s:%(lineno)- 4d: %(message)s"
                ),
                datefmt="[%H:%M:%S%z]",
            )
        )
        radyolo_logger.addHandler(file_handler)
        radyolo_logger.setLevel(logging.DEBUG)
        LOGGER.info("Experiment folder: %s", self.save_dir.resolve())
        LOGGER.info("Input config path: %s", self.config_path.resolve())
        LOGGER.info("Dataset path: %s", Path(self.config.data).resolve())

    def _setup_distributed_process_group(self) -> None:
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
            timeout=datetime.timedelta(minutes=10),
        )
        LOGGER.info("Rank %d finished distributed process group initialization.", self.rank)

    def _teardown_distributed_process_group(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()

    def _validate_distributed_setup(self) -> None:
        if self.device.type != "cuda":
            msg = "DistributedDataParallel requires explicit CUDA devices, e.g. device='0,1'."
            raise ValueError(msg)
        if self.config.batch_size < self.configured_world_size:
            msg = (
                "batch_size must be at least the number of distributed processes; got "
                f"batch_size={self.config.batch_size} and world_size={self.configured_world_size}."
            )
            raise ValueError(msg)
        if self.config.batch_size % self.configured_world_size != 0:
            msg = (
                "batch_size must be divisible by the number of distributed processes; got "
                f"batch_size={self.config.batch_size} and world_size={self.configured_world_size}."
            )
            raise ValueError(msg)

    def _launch_distributed_training(self) -> None:
        self._validate_distributed_setup()
        master_addr = "127.0.0.1"
        master_port = _find_free_port()
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
        if not self.distributed:
            return self.config.batch_size
        return self.config.batch_size // self.world_size

    def _get_per_process_workers(self) -> int:
        if not self.distributed or self.config.workers == 0:
            return self.config.workers
        return max(self.config.workers // self.world_size, 1)

    def _barrier(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.barrier()

    def _broadcast_stop(self, stop: bool) -> bool:
        if not self.distributed:
            return stop
        stop_tensor = torch.tensor(int(stop), device="cpu")
        dist.broadcast(stop_tensor, src=0)
        return bool(stop_tensor.item())

    def copy_train_config(self) -> None:
        shutil.copy2(self.config_path, self.save_dir / "train_config.yaml")

    def _compute_validation_fitness(self, metrics: dict[str, float]) -> float:
        map_1095 = float(metrics.get("metrics/mAP10-95", 0.0))
        map_1095_agn = float(metrics.get("metrics/mAP10-95Agn", 0.0))
        dice = float(metrics.get("metrics/dice", 0.0))
        weights = self.config.fitness_metric_weights
        total_weight = sum(weights)
        return (
            weights[0] * map_1095 + weights[1] * map_1095_agn + weights[2] * dice
        ) / total_weight

    def check_resume(self) -> None:
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
        if isinstance(self.model, HardcodedModel):
            return None, self.model

        checkpoint_path = self.last_model_path
        model_state_dict = None
        checkpoint: ModelCheckpoint | None = None
        if self.resume and checkpoint_path.suffix == ".pt" and checkpoint_path.exists():
            checkpoint = load_model_checkpoint(checkpoint_path)
            self.config = type(self.config).model_validate(checkpoint["train_config"].model_dump())
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
        ckpt, model = self.setup_model(
            self.data.nc,
        )
        model = model.to(self.device)
        self.loss = build_detection_loss_from_config(self.config, model, self.device)

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
        accumulate = max(round(self.config.nominal_batch_size / self.config.batch_size), 1)
        weight_decay = (
            self.config.weight_decay
            * self.config.batch_size
            * accumulate
            / self.config.nominal_batch_size
        )
        optimizer = self.build_optimizer(
            model=model,
            name=self.config.optimizer,
            lr=self.config.initial_learning_rate,
            momentum=self.config.momentum,
            decay=weight_decay,
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

            if self.is_main_process:
                LOGGER.info(
                    "%11s" * (3 + len(self.loss_names)),
                    "Epoch",
                    "GPU_mem",
                    *self.loss_names,
                    "Size",
                )
                pbar = tqdm(
                    enumerate(train_loader),
                    total=num_batches,
                    bar_format="{l_bar}{bar:10}{r_bar}",
                    smoothing=0.05,
                )
            else:
                pbar = enumerate(train_loader)
            tloss: torch.Tensor | None = torch.zeros(5, dtype=torch.float64)
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
                                    [2, self.config.nominal_batch_size / self.config.batch_size],
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
                    with torch.autocast(self.device.type):
                        prepared_batch = self._prepare_training_batch(
                            batch,
                            run_gpu_augmentations,
                        )
                        raw_predictions = model(prepared_batch["img"])
                        if isinstance(raw_predictions, torch.Tensor):
                            raise TypeError(raw_predictions)
                        if self.loss is None:
                            msg = "Detection loss is not initialized"
                            raise RuntimeError(msg)

                        loss, self.loss_items = cast(
                            "tuple[torch.Tensor, torch.Tensor]",
                            self.loss(
                                raw_predictions,
                                prepared_batch,
                                current_epoch=self.epoch + 1,
                            ),
                        )

                        tloss = (
                            tloss * i + self.loss_items.to(dtype=tloss.dtype, device=tloss.device)
                        ) / (i + 1)

                    # Backward
                    deterministic_restore = torch.are_deterministic_algorithms_enabled()
                    try:
                        torch.use_deterministic_algorithms(False)
                        scaler.scale(loss).backward()
                    finally:
                        torch.use_deterministic_algorithms(deterministic_restore)

                    # Optimize
                    if num_iterations - last_opt_step >= accumulate:
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

                    # Log
                    loss_length = tloss.shape[0] if len(tloss.shape) else 1
                    if self.is_main_process:
                        pbar.set_description(
                            ("%11s" * 2 + "%11.4g" * (loss_length))
                            % (
                                f"{self.epoch + 1}/{self.epochs}",
                                f"{self._get_memory():.3g}G",
                                *(tloss if loss_length > 1 else torch.unsqueeze(tloss, 0)),
                            )
                        )

            scheduler.step()
            self._barrier()

            if loss is None:
                message = "Detected None loss"
                raise RuntimeError(message)

            if self.is_main_process and tloss is not None:
                LOGGER.info(
                    "%15s" * 1 + "%11s" * (2 + len(self.loss_names)),  # noqa: G003
                    "Completed Epoch",
                    "GPU_mem",
                    *self.loss_names,
                    "Loss",
                )
                LOGGER.info(
                    "%15s%11s" + "%11.4g" * (loss_length + 1),  # noqa: G003
                    f"{self.epoch + 1}/{self.epochs}",
                    f"{self._get_memory():.3g}G",
                    *(tloss if loss_length > 1 else torch.unsqueeze(tloss, 0)),  # losses
                    loss.detach(),
                )

            final_epoch = self.epoch + 1 >= self.epochs

            if self.is_main_process:
                metrics: dict[str, float] = {}
                fitness = 0.0
                if validator is not None and (self.config.val or final_epoch or stop):
                    metrics, fitness, best_fitness = self.validate(
                        validator=validator,
                        ema=ema,
                        best_fitness=best_fitness,
                    )

                self.save_metrics(
                    metrics={
                        **(self.label_loss_items(tloss) if tloss is not None else {}),
                        **metrics,
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

        self._clear_memory()

    @staticmethod
    def _create_epoch_timing_values() -> dict[str, float]:
        keys = (
            "timing/gpu_steps_total_s",
            "timing/post_gpu_to_next_gpu_steps_s",
        )
        return dict.fromkeys(keys, 0.0)

    def _prepare_training_batch(self, batch: dict, run_gpu_augmentations: bool) -> dict:
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
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
        return batch

    @staticmethod
    def optimizer_step(
        scaler: torch.GradScaler,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        ema: ModelEMA,
        grad_clip_max_norm: float,
    ) -> None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        ema.update(model)

    def _get_memory(self) -> float:
        """Get accelerator memory utilization in GB."""
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
        elif self.device.type == "cpu":
            memory = 0
        else:
            memory = torch.cuda.memory_reserved()
        return memory / 1e9

    def _clear_memory(self) -> None:
        """Clear accelerator memory on used platform."""
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()

    @staticmethod
    def get_dataset_config(
        dataset_folder: Path,
    ) -> DatasetConfig:
        """Reduction of get_dataset(), check_file() and validation included in Pydantic model."""
        dataset_config_path = dataset_folder / "dataset.yaml"
        with Path(dataset_config_path).open() as f:
            yaml_config = yaml.safe_load(f)
        return DatasetConfig.model_validate(yaml_config)

    def get_dataloader(
        self,
        dataset_path: Path,
        model: torch.nn.Module,
        fold: int = 0,
        mode: Literal["train", "val"] = "train",
    ) -> DataLoader:
        if mode not in {"train", "val"}:
            message = f"get_dataloader() only supports modes train and val, but got {mode}"
            raise ValueError(message)

        dataset_class = (
            PatchWiseDataset if self.config.training_mode == "patchwise" else WholeImageDataset
        )
        foreground_oversampling = 0.0
        if mode == "train" or self.config.training_mode == "patchwise":
            foreground_oversampling = self.config.foreground_oversampling

        dataset = dataset_class(
            src_path=dataset_path,
            fold=fold,
            group=mode,
            patch_size=self.config.patch_size,
            augment=mode == "train",
            min_stride=get_model_min_stride(model),
            normalization=self.config.normalization,
            apply_preprocessing=self.config.apply_preprocessing,
            load_masks=self.config.segmentation_output,
            foreground_oversampling=foreground_oversampling,
            min_box_crop_fraction=self.config.min_box_crop_fraction,
            config=self.config,
        )
        sampler = None
        shuffle = mode == "train"
        if self.distributed and mode == "train":
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
        save_paths = [self.last_model_path]
        if best_fitness == fitness:
            save_paths.append(self.best_model_path)
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            save_paths.append(self.wdir / f"epoch{self.epoch}.pt")
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

    def get_validator(
        self,
        test_loader: DataLoader,
        save_dir: Path,
        names: dict[int, str],
    ) -> Validator:
        self.loss_names = "box_loss", "cls_obj_loss", "dfl_loss", "mask_loss", "cls_sep_loss"
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
        metrics = validator(self, ema=ema)
        current_fitness = self._compute_validation_fitness(metrics)
        if self.ewa_fitness is None:
            self.ewa_fitness = current_fitness
        else:
            self.ewa_fitness = self.ewa_alpha * float(self.ewa_fitness) + (
                1.0 - self.ewa_alpha
            ) * current_fitness
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
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2
        header = (
            ""
            if self.csv_path.exists()
            else (("%s," * n % ("epoch", "time", *keys)).rstrip(",") + "\n")
        )
        t = time.time() - train_time_start
        with self.csv_path.open("a") as f:
            line = ("%.6g," * n % (self.epoch + 1, t, *vals)).rstrip(",")
            f.write(header + line + "\n")

    def build_optimizer(
        self,
        model: torch.nn.Module,
        name: str = "auto",
        lr: float = 0.0001,
        momentum: float = 0.9,
        decay: float = 1e-5,
    ) -> torch.optim.Optimizer:
        weights_with_decay, weights_without_decay, bias_weights = [], [], []
        available_normalization_layers = tuple(
            v for k, v in torch.nn.__dict__.items() if "Norm" in k
        )
        name_ = name

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:
                    bias_weights.append(param)
                elif isinstance(module, available_normalization_layers):
                    weights_without_decay.append(param)
                else:
                    weights_with_decay.append(param)

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
        name_ = {x.lower(): x for x in optimizers}.get(name_.lower())
        if name_ in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(torch.optim, name, torch.optim.Adam)(
                bias_weights, lr=lr, betas=(momentum, 0.999), weight_decay=0.0
            )
        elif name_ == "RMSProp":
            optimizer = torch.optim.RMSprop(bias_weights, lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = torch.optim.SGD(bias_weights, lr=lr, momentum=momentum, nesterov=True)
        else:
            msg = f"Optimizer '{name_}' not found in list of available optimizers {optimizers}."
            raise NotImplementedError(msg)

        optimizer.add_param_group({"params": weights_with_decay, "weight_decay": decay})
        optimizer.add_param_group({"params": weights_without_decay, "weight_decay": 0.0})
        return optimizer
