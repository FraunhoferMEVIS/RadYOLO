from __future__ import annotations

import datetime
import gc
import io
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm_loggable
import tqdm_loggable.utils
import yaml
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm_loggable.auto import tqdm

from yolo11_3d.nn.data import build_dataloader
from yolo11_3d.ultralytics.utils.plotting import plot_results
from yolo11_3d.ultralytics.utils.torch_utils import EarlyStopping

from .config import (
    DatasetConfig,
    DefaultConfig,
    ModelCheckpoint,
    get_cfg,
    model_checkpoint_from_torch,
)
from .data import NIfTIDatasetWithMasks
from .model import HardcodedModel
from .validate import Validator

from .util import ModelEMA, init_seeds, select_device

if TYPE_CHECKING:
    from collections.abc import Callable

    from .validate import Validator

LOGGER = logging.getLogger(__name__)
LOGGER_YOLO = logging.getLogger("yolo." + __name__)


class YoloFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.name.startswith("yolo.")


class Trainer:
    def __init__(self, cfg: str | Path) -> None:
        self.args = get_cfg(cfg, {})
        run_dir = Path() / (self.args.project or "runs")
        if not run_dir.is_dir():
            msg = f"{run_dir} does not exist or is not a directory!"
            raise RuntimeError(msg)
        
        self.save_dir = run_dir / (str(self.args.name) or time_now.strftime("%Y_%m_%d-%H_%M_%S"))
        self.wdir = self.save_dir / "weights"
        self.save_dir.mkdir(exist_ok=self.args.exist_ok)
        self.wdir.mkdir(exist_ok=self.args.exist_ok)
        self.last_model_path = self.wdir / "last.pt"
        self.best_model_path = self.wdir / "best.pt"
        self.save_period = self.args.save_period
        self.csv_path = self.save_dir / "results.csv"

        self.check_resume()

        self.device = select_device(self.args.device)
        if self.args.use_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        init_seeds(self.args.seed, deterministic=self.args.deterministic)

        self.epochs = self.args.epochs or 100
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0

        self.model: torch.nn.Module | None = None
        self.data = self.get_dataset_config(self.args.data)

        self.lf: Callable[[int], float] | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.loss_items = torch.ones((0,))

        time_now = datetime.datetime.now(tz=datetime.UTC)

        file_handler = logging.FileHandler(
            str(self.save_dir.joinpath(time_now.strftime("%Y_%m_%d-%H_%M_%S")).with_suffix(".log")),
            encoding="utf-8",
        )
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)- 9s %(module)30.30s:%(funcName)-14.14s:%(lineno)- 4d: %(message)s",
                datefmt="[%H:%M:%S%z]",
            )
        )
        file_handler.addFilter(YoloFilter())
        logging.root.addHandler(file_handler)

        # Stateful metric: exponentially weighted mAP10-95 across epochs (persisted on resume)
        self.ewa_mAP1095 = None
        self.ewa_alpha = 0.6

    def check_resume(self) -> None:
        resume = self.args.resume
        if resume:
            try:
                if not self.last_model_path.exists():
                    self.resume = False
                    return

                ckpt_args = model_checkpoint_from_torch(self.last_model_path)["train_args"]
                if not Path(ckpt_args.data).exists():
                    ckpt_args.data = self.args.data

                message = f"Resuming from model checkpoint {self.last_model_path}..."
                print(message)
                LOGGER.info(message)
                resume = True
                self.args = ckpt_args

            except Exception as e:
                msg = "Resume checkpoint not found, please pass a valid checkpoint to resume from."
                raise FileNotFoundError(msg) from e
        self.resume = resume

    def setup_model(
        self,
        num_classes: int,
        device: torch.device,
        input_channels: int = 3,
        scale: Literal["n", "s", "m", "l", "x"] = "n",
    ) -> tuple[ModelCheckpoint | None, HardcodedModel]:
        if isinstance(self.model, HardcodedModel):
            return None, self.model

        checkpoint_path = self.last_model_path
        weights_ema = None
        checkpoint: ModelCheckpoint | None = None
        if self.resume and checkpoint_path.suffix == ".pt" and checkpoint_path.exists():
            checkpoint = model_checkpoint_from_torch(checkpoint_path)
            self.args = self.args.model_copy(update=checkpoint["train_args"].model_dump())
            weights_ema = checkpoint["ema"].to(self.device).float()

        model = HardcodedModel(
            input_channels=input_channels,
            num_classes=num_classes,
            scale=scale,
            box=self.args.box,
            cls=self.args.cls,
            dfl=self.args.dfl,
            mask_gain=self.args.mask_gain,
            class_separate_gain=self.args.class_separate_gain,
            device=device,
            enable_output_masks=self.args.segmentation_output,
            num_prototype_masks=self.args.prototype_masks,
            higher_resolution_prototypes=self.args.higher_resolution_prototypes,
            separate_objectness_head=self.args.separate_objectness_head,
            include_stride4_output=self.args.include_stride4_output,
            objectness_loss=self.args.objectness_loss,
        )
        if weights_ema:
            model.load(weights_ema)
        return checkpoint, model

    def _setup_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> tuple[torch.optim.lr_scheduler.LRScheduler, Callable[[int], float]]:
        def lf(x: int) -> float:
            return max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lf)
        return scheduler, lf

    def _setup_train(
        self,
    ) -> tuple[
        HardcodedModel,
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
            input_channels=self.args.input_channels,
            scale=self.args.model_scale,
            device=self.device,
        )
        model = model.to(self.device)

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
        scaler = torch.GradScaler("cuda", enabled=self.args.amp)

        # Dataloaders
        train_loader = self.get_dataloader(
            self.args.data, model, fold=self.args.fold, mode="train"
        )
        if self.args.val:
            val_loader = self.get_dataloader(
                self.args.data, model, fold=self.args.fold, mode="val"
            )
            validator = self.get_validator(
                val_loader,
                self.save_dir,
                self.data.names,
            )
        else:
            val_loader = None
            validator = None
        ema = ModelEMA(model)

        # Optimizer
        accumulate = max(round(self.args.nbs / self.args.batch_size), 1)
        weight_decay = self.args.weight_decay * self.args.batch_size * accumulate / self.args.nbs
        optimizer = self.build_optimizer(
            model=model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
        )

        # Scheduler
        scheduler, lf = self._setup_scheduler(optimizer)
        stopper, stop = EarlyStopping(patience=self.args.patience), False
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

    def _train_loop(
        self,
        model: HardcodedModel,
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
        args = self.args
        num_batches = len(train_loader)
        num_warmup_iterations = (
            max(round(args.warmup_epochs * num_batches), 100) if args.warmup_epochs > 0 else -1
        )
        train_time_start = time.time()
        self.epoch = start_epoch
        last_opt_step = num_batches * self.epoch
        optimizer.zero_grad()
        while True:
            model.train()

            LOGGER_YOLO.info(
                "%11s" * (3 + len(self.loss_names)),
                "Epoch",
                "GPU_mem",
                *self.loss_names,
                "Size",
            )
            pbar = tqdm(
                enumerate(train_loader), total=num_batches, bar_format="{l_bar}{bar:10}{r_bar}"
            )
            tloss: torch.Tensor | None = None
            loss_length: int = 1
            loss: torch.Tensor | None = None
            with (
                logging_redirect_tqdm(loggers=[logging.root, logging.getLogger("yolo11-3d")])
                if tqdm_loggable.utils.is_interactive_session()
                else nullcontext()
            ):
                for i, batch in pbar:
                    # Warmup
                    num_iterations = i + num_batches * self.epoch
                    if num_iterations <= num_warmup_iterations:
                        warmup_iteration_range = [0, num_warmup_iterations]
                        accumulate = max(
                            1,
                            int(
                                np.interp(
                                    num_iterations,
                                    warmup_iteration_range,
                                    [1, args.nbs / self.args.batch_size],
                                ).round()
                            ),
                        )
                        for j, x in enumerate(optimizer.param_groups):
                            x["lr"] = np.interp(
                                num_iterations,
                                warmup_iteration_range,
                                [
                                    args.warmup_bias_lr if j == 0 else 0.0,
                                    x["initial_lr"] * lf(self.epoch),
                                ],
                            )
                            if "momentum" in x:
                                x["momentum"] = np.interp(
                                    num_iterations,
                                    warmup_iteration_range,
                                    [args.warmup_momentum, args.momentum],
                                )

                    # Forward
                    with torch.autocast(self.device.type):
                        batch["img"] = batch["img"].to(self.device, non_blocking=True)
                        if self.args.segmentation_output:
                            batch["masks"] = tuple(masks.to(self.device, non_blocking=True) if masks is not None else None
                                                   for masks in batch["masks"])
                        
                        loss, self.loss_items = cast(
                            "tuple[torch.Tensor, torch.Tensor]", model(batch)
                        )

                        tloss = (
                            (tloss * i + self.loss_items) / (i + 1)
                            if tloss is not None
                            else self.loss_items
                        )

                    # Backward
                    deterministic_restore = torch.are_deterministic_algorithms_enabled()
                    try:
                        torch.use_deterministic_algorithms(False)
                        scaler.scale(loss).backward()
                    finally:
                        torch.use_deterministic_algorithms(deterministic_restore)

                    # Optimize
                    if num_iterations - last_opt_step >= accumulate:
                        self.optimizer_step(scaler, optimizer, model, ema)
                        last_opt_step = num_iterations

                    # Log
                    loss_length = tloss.shape[0] if len(tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (1 + loss_length))
                        % (
                            f"{self.epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",
                            *(tloss if loss_length > 1 else torch.unsqueeze(tloss, 0)),  # losses
                            batch["img"].shape[-1],  # imgsz
                        )
                    )

            scheduler.step()

            if loss is None:
                raise RuntimeError

            if tloss is not None:
                LOGGER.warning(
                    "%15s" * 1 + "%11s" * (2 + len(self.loss_names)),
                    "Completed Epoch",
                    "GPU_mem",
                    *self.loss_names,
                    "Loss",
                )
                LOGGER.warning(
                    "%15s%11s" + "%11.4g" * (loss_length + 1),
                    f"{self.epoch + 1}/{self.epochs}",
                    f"{self._get_memory():.3g}G",
                    *(tloss if loss_length > 1 else torch.unsqueeze(tloss, 0)),  # losses
                    loss,
                )

            final_epoch = self.epoch + 1 >= self.epochs
            ema.update_attr(
                model,
                include=["yaml", "nc", "args", "names", "stride", "class_weights", "num_classes"],
            )

            if validator is not None and (self.args.val or final_epoch or stop):
                metrics, fitness, best_fitness = self.validate(
                    validator=validator,
                    ema=ema,
                    best_fitness=best_fitness,
                    amp=self.device.type != "cpu",
                )
                self.save_metrics(
                    metrics={
                        **(self.label_loss_items(tloss) if tloss is not None else {}),
                        **metrics,
                    },
                    train_time_start=train_time_start,
                )
            else:
                metrics = {}
                fitness = 0.0

            stop |= stopper(self.epoch + 1, fitness) or final_epoch

            if args.save or final_epoch:
                self.save_model(
                    epoch=self.epoch,
                    best_fitness=best_fitness or 0.0,
                    ema=ema,
                    optimizer=optimizer,
                    scaler=scaler,
                    train_args=args,
                    metrics=metrics,
                    fitness=fitness,
                )

            if args.plots:
                plot_results(self.csv_path)

            self._clear_memory()

            if stop:
                break

            self.epoch += 1

        seconds = time.time() - train_time_start
        msg = f"\n{self.epoch + 1} epochs completed in {seconds / 3600:.3f} hours."
        LOGGER.info(msg)

        self._clear_memory()

    @staticmethod
    def optimizer_step(
        scaler: torch.GradScaler,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        ema: ModelEMA,
    ) -> None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
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
        """Reduction of get_dataset(), check_det_dataset(), check_file() and validation included in Pydantic model."""
        dataset_config_path = dataset_folder / "dataset.yaml"
        with Path(dataset_config_path).open() as f:
            yaml_config = yaml.safe_load(f)
        dataset_config = DatasetConfig.model_validate(yaml_config)
        return dataset_config

    def get_dataloader(
        self,
        dataset_path: Path,
        model: torch.nn.Module,
        fold: int = 0,
        mode: Literal["train", "val"] = "train",
    ) -> DataLoader:
        assert mode in {"train", "val"}
        dataset = NIfTIDatasetWithMasks(
            src_path=dataset_path,
            fold=fold,
            group=mode,
            imgsz=tuple(self.args.imgsz),
            augment=self.args.augment and mode == "train",
            min_stride=int(
                model.stride.min() if model and isinstance(model.stride, torch.Tensor) else 8
            ),
            pad=0.0 if mode == "train" else 0.5,
            normalization=self.args.normalization,
            config=self.args,
        )
        shuffle = mode == "train"
        data_loader = build_dataloader(
            dataset, 
            self.args.batch_size, 
            self.args.workers, 
            shuffle, 
            self.device,
        )
        return data_loader

    def save_model(
        self,
        epoch: int,
        best_fitness: float,
        ema: ModelEMA,
        optimizer: torch.optim.Optimizer,
        scaler: torch.GradScaler,
        train_args: DefaultConfig,
        metrics: dict[str, float],
        fitness: float,
    ):
        buffer = io.BytesIO()
        checkpoint = ModelCheckpoint(
            epoch=epoch,
            best_fitness=best_fitness,
            model=None,
            ema=ema.ema,
            updates=ema.updates,
            optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),
            train_args=train_args,
            train_metrics={**metrics, "metrics/mAP-ewa(B)": float(self.ewa_mAP1095) if self.ewa_mAP1095 is not None else 0.0},
            train_results={},
            date=datetime.datetime.now(tz=datetime.UTC),
        )
        torch.save(dict(checkpoint), buffer)
        serialized_checkpoint = buffer.getvalue()

        self.last_model_path.write_bytes(serialized_checkpoint)
        if best_fitness == fitness:
            self.best_model_path.write_bytes(serialized_checkpoint)
        if (self.save_period > 0) and (epoch % self.save_period == 0):
            (self.wdir / f"epoch{epoch}.pt").write_bytes(serialized_checkpoint)

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
            args=self.args,
            names=names,
        )

    def validate(
        self,
        validator: Validator,
        ema: ModelEMA,
        best_fitness: float | None,
        amp: bool,
    ) -> tuple[dict[str, float], float, float]:
        metrics = validator(self, ema=ema, amp=amp)
        # Update stateful EWA using per-epoch mAP 0.1–0.95
        map_val = metrics.get("metrics/mAP10-95(B)")
        if map_val is not None:
            if self.ewa_mAP1095 is None:
                self.ewa_mAP1095 = float(map_val)
            else:
                self.ewa_mAP1095 = self.ewa_alpha * float(self.ewa_mAP1095) + (1.0 - self.ewa_alpha) * float(map_val)
            metrics["metrics/mAP-ewa(B)"] = float(self.ewa_mAP1095)
        # Use EWA as new fitness metric if available, otherwise fall back to original
        fitness: float = (
            float(self.ewa_mAP1095)
            if self.ewa_mAP1095 is not None
            else 0.0
        )
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
            rounded_loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, rounded_loss_items, strict=True))
        return keys

    def resume_training(
        self,
        ckpt: ModelCheckpoint | None,
        optimizer: torch.optim.Optimizer,
        scaler: torch.GradScaler,
        ema: ModelEMA,
        best_fitness: float | None = None,
    ) -> tuple[float | None, int]:
        if ckpt is None:  # or not self.resume:
            return best_fitness, 0
        start_epoch = ckpt["epoch"] + 1
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            best_fitness = ckpt["best_fitness"]
        if ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        if ckpt["ema"] is not None:
            ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
            ema.updates = ckpt["updates"]
        # Restore stateful EWA metric if present in saved train_metrics
        try:
            ewa_saved = ckpt.get("train_metrics", {}).get("metrics/mAP-ewa(B)")
            if isinstance(ewa_saved, (int, float)):
                self.ewa_mAP1095 = float(ewa_saved)
        except Exception:
            pass
        assert start_epoch > 0
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
                ckpt["epoch"],
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
