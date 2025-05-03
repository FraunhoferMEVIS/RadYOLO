from __future__ import annotations

import datetime
import gc
import logging
import math
import time
from copy import copy, deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import torch
import yaml
from tqdm import tqdm

from yolo11_3d.nn.config import DatasetConfig, DefaultConfig, ModelCheckpoint, get_cfg
from yolo11_3d.nn.data import BaseDataset, DummyDataset, MIPDataset, NIfTIDataset
from yolo11_3d.nn.model import Hardcoded3DModel, HardcodedModel
from yolo11_3d.nn.util import ModelEMA, init_seeds, select_device
from yolo11_3d.ultralytics.data.build import InfiniteDataLoader, build_dataloader
from yolo11_3d.ultralytics.utils.checks import check_imgsz
from yolo11_3d.ultralytics.utils.torch_utils import convert_optimizer_state_dict_to_fp16

if TYPE_CHECKING:
    from collections.abc import Callable

LOGGER = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg: str | Path) -> None:
        self.args = get_cfg(cfg, {})
        self.device = select_device(self.args.device)

        init_seeds(self.args.seed, deterministic=self.args.deterministic)

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs or 100
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0

        self.model_path = self.args.model.with_suffix(".pt")
        self.model: torch.nn.Module | None = None
        self.data, self.trainset, self.testset = self.get_dataset_configs(self.args.data)
        self.ema: ModelEMA | None = None

        self.lf: Callable[[int], float] | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        # self.loss_names = ["Loss"]
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"

        run_dir = Path() / "runs"
        if not run_dir.is_dir():
            msg = f"{run_dir} does not exist or is not a directory!"
            raise RuntimeError(msg)
        self.wdir = run_dir / datetime.datetime.now(tz=datetime.UTC).strftime("%Y_%m_%d-%H_%M_%S")
        self.wdir.mkdir()
        self.last_model_path = self.wdir / "last.pt"
        self.best_model_path = self.wdir / "best.pt"
        self.save_period = self.args.save_period

    def setup_model(
        self,
        num_classes: int,
        device: torch.device,
        input_channels: int = 3,
        scale: Literal["n", "s", "m", "l", "x"] = "n",
    ) -> tuple[ModelCheckpoint | None, torch.nn.Module]:
        if isinstance(self.model, torch.nn.Module):
            return None, self.model

        checkpoint_path = self.model_path
        weights_ema = None
        checkpoint: ModelCheckpoint | None = None
        if checkpoint_path.suffix == ".pt" and checkpoint_path.exists():
            checkpoint = ModelCheckpoint.model_validate(
                torch.load(checkpoint_path, weights_only=False)
            )
            self.args = self.args.model_copy(update=checkpoint.train_args.model_dump())
            weights_ema = checkpoint.ema.to(self.device).float()
        elif isinstance(self.args.pretrained, Path):
            raise NotImplementedError
            weights_ema = torch.load(self.args.pretrained)

        # model = HardcodedModel(
        #     input_channels=input_channels,
        #     num_classes=num_classes,
        #     scale=scale,
        #     box=self.args.box,
        #     cls=self.args.cls,
        #     dfl=self.args.dfl,
        #     device=device,
        # )

        model = Hardcoded3DModel(
            input_channels=input_channels,
            num_classes=num_classes,
            scale=scale,
            box=self.args.box,
            cls=self.args.cls,
            dfl=self.args.dfl,
            device=device,
        )
        if weights_ema:
            model.load(weights_ema)
        return checkpoint, model

    def _setup_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> tuple[torch.optim.lr_scheduler.LRScheduler, Callable[[int], float]]:
        if self.args.cos_lr:
            raise NotImplementedError
        else:  # noqa: RET506

            def lf(x: int) -> float:
                return max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lf)
        return scheduler, lf

    def _setup_train(
        self,
        world_size: int,
        device: torch.device,
        args: DefaultConfig,
        batch_size: int,
        epochs: int,
        start_epoch: int = 1,
    ) -> tuple[
        torch.nn.Module,
        bool,
        torch.GradScaler,
        DefaultConfig,
        int,
        int,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader | None,
        None,
        dict[str, int],
        ModelEMA,
        int,
        torch.optim.Optimizer,
        bool,
        torch.optim.lr_scheduler.LRScheduler,
        Callable[[int], float],
    ]:
        ckpt, model = self.setup_model(self.data.nc, input_channels=4, scale="n", device=device)
        model = model.to(device)
        # self.set_model_attributes(model)

        # Freeze layers
        freeze_list = (
            args.freeze
            if isinstance(args.freeze, list)
            else range(args.freeze)
            if isinstance(args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]
        freeze_layer_names = [f"model.{x}" for x in freeze_list] + always_freeze_names
        for layer_name, layer in model.named_parameters():
            if any(name in layer_name for name in freeze_layer_names):
                LOGGER.info("Freezing layer '%s'", layer_name)
                layer.requires_grad = False
            elif not layer.requires_grad and layer.dtype.is_floating_point:
                LOGGER.warning("setting requires_grad=True for frozen layer %s.", layer_name)
                layer.requires_grad = True

        # Check AMP
        amp = args.amp
        if amp:
            LOGGER.warning(
                "Automatic Mixed Precision enabled. Please check if your system fully supports this."
            )
            # amp = check_amp(model) # TODO(lmg): implement check for stable amp support
        scaler = torch.GradScaler("cuda", enabled=amp)
        if world_size > 1:
            raise NotImplementedError

        # Check imgsz
        grid_size = max(
            int(
                model.stride.max()
                if hasattr(model, "stride") and isinstance(model.stride, torch.Tensor)
                else 32
            ),
            32,
        )
        # args.imgsz = check_imgsz(args.imgsz, stride=grid_size, floor=grid_size, max_dim=1)
        stride = grid_size

        # Batch size # TODO: maybe later integrate this
        # if batch_size < 1:
        #     args.batch = batch_size = self.auto_batch()

        # Dataloaders
        train_loader = self.get_dataloader(
            self.trainset, model, batch_size=batch_size, rank=-1, mode="train", device=device
        )
        if self.testset is not None:
            test_loader = self.get_dataloader(
                self.testset, model, batch_size=batch_size * 2, rank=-1, mode="val", device=device
            )
        else:
            test_loader = None
        validator = None  # self.get_validator()
        # metric_keys = validator.metrics.keys + self.label_loss_items(prefix="val")
        metrics: dict[str, int] = {}  # dict(zip(metric_keys, [0] * len(metric_keys), strict=True))
        ema = ModelEMA(model)
        if args.plots:
            raise NotImplementedError

        # Optimizer
        accumulate = max(round(args.nbs / batch_size), 1)
        weight_decay = args.weight_decay * batch_size * accumulate / args.nbs
        iterations = math.ceil(len(train_loader.dataset) / max(batch_size, args.nbs)) * epochs  # pyright: ignore[reportArgumentType]
        optimizer = self.build_optimizer(
            model=model,
            name=args.optimizer,
            lr=args.lr0,
            momentum=args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )

        # Scheduler
        scheduler, lf = self._setup_scheduler(optimizer)
        stop = False
        # self.resume_training(ckpt)
        scheduler.last_epoch = start_epoch - 1
        return (
            model,
            amp,
            scaler,
            args,
            stride,
            batch_size,
            train_loader,
            test_loader,
            validator,
            metrics,
            ema,
            accumulate,
            optimizer,
            stop,
            scheduler,
            lf,
        )

    def train(self) -> None:
        world_size = 1 if torch.cuda.is_available() else 0

        # Setup Train
        (
            model,
            amp,
            scaler,
            args,
            stride,
            batch_size,
            train_loader,
            test_loader,
            validator,
            metrics,
            ema,
            accumulate,
            optimizer,
            stop,
            scheduler,
            lf,
        ) = self._setup_train(
            world_size=world_size,
            device=self.device,
            args=self.args,
            batch_size=self.batch_size,
            epochs=self.epochs,
        )

        self._train_loop(
            epochs=self.epochs,
            batch_size=self.batch_size,
            args=self.args,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            lf=lf,
            amp_device_type="cuda",
            ema=ema,
            accumulate=accumulate,
        )

    def _train_loop(
        self,
        epochs: int,
        batch_size: int,
        args: DefaultConfig,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader | None,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: torch.amp.grad_scaler.GradScaler,
        lf: Callable[[int], float],
        amp_device_type: Literal["cuda", "cpu"],
        ema: ModelEMA,
        accumulate: int,
        # world_size=1,
    ) -> None:
        stop = False
        num_batches = len(train_loader)
        num_warmup_iterations = (
            max(round(args.warmup_epochs * num_batches), 100) if args.warmup_epochs > 0 else -1
        )
        last_opt_step = -1
        epoch_time = None
        epoch_time_start = time.time()
        train_time_start = time.time()
        epoch = 0
        best_fitness: float | None = None
        optimizer.zero_grad()
        while True:
            # epoch_ = epoch
            scheduler.step()
            model.train()
            pbar = tqdm(
                enumerate(train_loader), total=num_batches, bar_format="{l_bar}{bar:10}{r_bar}"
            )
            # if epoch == (epochs - args.close_mosaic):
            #     raise NotImplementedError

            LOGGER.info(
                "\n" + "%11s" * (3 + len(self.loss_names)),
                "Epoch",
                "GPU_mem",
                *self.loss_names,
                # "Instances",
                "Size",
            )
            tloss = None
            loss: torch.Tensor | None = None
            for i, batch in pbar:
                # Warmup
                num_iterations = i + num_batches * epoch
                if num_iterations <= num_warmup_iterations:
                    warmup_iteration_range = [0, num_warmup_iterations]
                    accumulate = max(
                        1,
                        int(
                            np.interp(
                                num_iterations,
                                warmup_iteration_range,
                                [1, args.nbs / batch_size],
                            ).round()
                        ),
                    )
                    for j, x in enumerate(optimizer.param_groups):
                        x["lr"] = np.interp(
                            num_iterations,
                            warmup_iteration_range,
                            [
                                args.warmup_bias_lr if j == 0 else 0.0,
                                x["initial_lr"] * lf(epoch),
                            ],
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(
                                num_iterations,
                                warmup_iteration_range,
                                [args.warmup_momentum, args.momentum],
                            )

                # Forward
                with torch.autocast(amp_device_type):
                    # batch = self.preprocess_batch(batch)
                    batch["img"] = batch["img"].to(self.device, non_blocking=True)
                    loss_items: torch.Tensor
                    loss, loss_items = cast("tuple[torch.Tensor, torch.Tensor]", model(batch))

                    tloss = (tloss * i + loss_items) / (i + 1) if tloss is not None else loss_items

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

                    if args.time:
                        raise NotImplementedError

                if args.plots:
                    raise NotImplementedError

                # Log
                loss_length = tloss.shape[0] if len(tloss.shape) else 1
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * (1 + loss_length))
                    % (
                        f"{epoch + 1}/{epochs}",
                        f"{self._get_memory():.3g}G",
                        *(tloss if loss_length > 1 else torch.unsqueeze(tloss, 0)),  # losses
                        # batch["cls"].shape[0],  # batch size
                        batch["img"].shape[-1],  # imgsz
                    )
                )

            if loss is None:
                raise RuntimeError

            final_epoch = epoch + 1 >= epochs
            ema.update_attr(
                model, include=["yaml", "nc", "args", "names", "stride", "class_weights"]
            )

            # if self.args.val or final_epoch or stop:
            #     metrics, fitness, best_fitness = self.validate(best_fitness=best_fitness, loss=loss)
            if best_fitness is None:
                best_fitness = loss.item()
            best_fitness = min(loss.item(), best_fitness)

            stop |= final_epoch
            if args.time:
                raise NotImplementedError

            if args.save or final_epoch:
                self.save_model(
                    epoch=epoch,
                    best_fitness=best_fitness,
                    ema=ema,
                    optimizer=optimizer,
                    train_args=args,
                    metrics={},
                    fitness=0,
                )

            t = time.time()
            epoch_time = t - epoch_time_start
            epoch_time_start = t
            if args.time:
                raise NotImplementedError

            self._clear_memory()

            if stop:
                break

            epoch += 1

        seconds = time.time() - train_time_start
        msg = f"\n{epoch + 1} epochs completed in {seconds / 3600:.3f} hours."
        LOGGER.info(msg)

        if args.plots:
            raise NotImplementedError

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
    def get_dataset_configs(
        dataset_config_path: Path,
    ) -> tuple[DatasetConfig, Path | list[Path], Path | list[Path] | None]:
        """Reduction of get_dataset(), check_det_dataset(), check_file() and validation included in Pydantic model."""
        with Path(dataset_config_path).open() as f:
            yaml_config = yaml.safe_load(f)
        dataset_config = DatasetConfig.model_validate(yaml_config)
        return dataset_config, dataset_config.train, dataset_config.val or dataset_config.test

    def get_dataloader(
        self,
        dataset_path: Path | list[Path],
        model: torch.nn.Module,
        batch_size: int = 16,
        rank: int = 0,
        mode: Literal["train", "val"] = "train",
        device: torch.device | str = "cpu",
    ) -> InfiniteDataLoader:
        assert mode in {"train", "val"}
        dataset = self.build_dataset(
            img_path=dataset_path, mode=mode, model=model, batch=batch_size
        )
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning(
                "'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False"
            )
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)

    def build_dataset(
        self,
        img_path: Path | list[Path],
        batch: int,
        mode: Literal["train", "val"] = "train",
        model: torch.nn.Module | None = None,
    ) -> BaseDataset:
        grid_size = max(
            int(model.stride.max() if model and isinstance(model.stride, torch.Tensor) else 0), 32
        )
        # if not isinstance(self.args.imgsz, int):
        #     raise TypeError
        # return MIPDataset(
        return NIfTIDataset(
            src_path=img_path,
            imgsz=tuple(self.args.imgsz),
            # augment=mode == "train",
            augment=False,
            batch_size=batch,
            stride=grid_size,
            pad=0.0 if mode == "train" else 0.5,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )
        return DummyDataset(256, self.data.nc, (3, 640, 640))
        # return YOLODataset(
        #     img_path=img_path,
        #     imgsz=self.args.imgsz,
        #     batch_size=batch,
        #     augment=mode == "train",  # sugmentation
        #     hyp=self.args,  # TODO: probably add a get_hyps_from_cfg function
        #     rect=self.args.rect or mode == "val",  # rectangular batches
        #     cache=self.args.cache or None,
        #     single_cls=self.args.single_cls or False,
        #     stride=grid_size,
        #     pad=0.0 if mode == "train" else 0.5,
        #     task=self.args.task,
        #     classes=self.args.classes,
        #     data=self.data,
        #     fraction=self.args.fraction if mode == "train" else 1.0,
        # )

    def save_model(
        self,
        epoch: int,
        best_fitness: float,
        ema: ModelEMA,
        optimizer: torch.optim.Optimizer,
        train_args: DefaultConfig,
        metrics: dict[str, float],
        fitness: float,
        # results: dict | None = None,
        # model: torch.nn.Module | None = None,
    ):
        import io

        buffer = io.BytesIO()
        checkpoint = ModelCheckpoint(
            epoch=epoch,
            best_fitness=best_fitness,
            model=None,
            ema=deepcopy(ema.ema).half(),
            updates=ema.updates,
            optimizer=convert_optimizer_state_dict_to_fp16(deepcopy(optimizer.state_dict())),
            train_args=train_args,
            train_metrics={**metrics, "fitness": fitness},
            train_results={},
            date=datetime.datetime.now(tz=datetime.UTC),
        )
        torch.save(dict(checkpoint), buffer)
        serialized_checkpoint = buffer.getvalue()

        self.last_model_path.write_bytes(serialized_checkpoint)
        if self.best_fitness == self.fitness:
            self.best_model_path.write_bytes(serialized_checkpoint)
        if (self.save_period > 0) and (epoch % self.save_period == 0):
            (self.wdir / f"epoch{epoch}.pt").write_bytes(serialized_checkpoint)

    # def get_validator(self):
    #     self.loss_names = "box_cls", "cls_loss", "dfl_loss"
    #     return yolo.detect.DetectionValidator(
    #         self.test_loader,
    #         save_dir=self.save_dir,
    #         args=copy(self.args),
    #         _callbacks=self.callbacks,
    #     )

    # def validate(
    #     self, best_fitness: float | None, loss: torch.Tensor
    # ):  # -> tuple[Any, Any | float, Any | float]:
    #     metrics = self.validator(self)
    #     fitness: float = metrics.pop("fitness", -loss.detach().cpu().numpy())
    #     if not best_fitness or best_fitness < fitness:
    #         best_fitness = fitness
    #     return metrics, fitness, best_fitness

    def resume_training(
        self,
        ckpt: ModelCheckpoint | None,
        optimizer: torch.optim.Optimizer,
        ema: ModelEMA,
        best_fitness: float | None,
    ) -> float | None:
        if ckpt is None:  # or not self.resume:
            return best_fitness
        # best_fitness = 0.0
        start_epoch = ckpt["epoch"] + 1
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            best_fitness = ckpt["best_fitness"]
        if self.ema and ckpt["ema"]:
            ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
            ema.updates = ckpt["updates"]
        assert start_epoch > 0
        LOGGER.info(
            "Resuming training %s from epoch %d to %d total epochs.",
            self.args.model,
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
        # self.best_fitness = best_fitness
        return best_fitness

    # def auto_batch(self):
    #     train_dataset = self.build_dataset(trainset, mode="train", batch=16)
    #     max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4
    #     return check_train_batch_size(
    #         model=model, imgsz=imgsz, amp=amp, batch_size=batch_size, max_num_obj=max_num_obj
    #     )

    def build_optimizer(
        self,
        model: torch.nn.Module,
        name: str = "auto",
        lr: float = 0.0001,
        momentum: float = 0.9,
        decay: float = 1e-5,
        iterations: float = 1e5,
    ) -> torch.optim.Optimizer:
        # g = [], [], []
        weights_with_decay, weights_without_decay, bias_weights = [], [], []
        available_normalization_layers = tuple(
            v for k, v in torch.nn.__dict__.items() if "Norm" in k
        )
        if name == "auto":
            raise NotImplementedError
            LOGGER.info(
                "'optimizer=auto' found, "
                "ignoring 'lr0=%f' and 'momentum=%f' and "
                "determining best 'optimizer', 'lr0' and 'momentum' automatically... ",
                self.args.lr0,
                self.args.momentum,
            )
            lr_fit = round(0.002 * 5 / (4 + self.nc), 6)
            name_, lr, momentum = (
                ("SGD", 0.01, 0.9) if iterations > 10000 else ("Adam", lr_fit, 0.9)
            )
            self.args.warmup_bias_lr = 0.0
        else:
            name_ = name

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:
                    bias_weights.append(param)
                    # g[2].append(param)
                elif isinstance(module, available_normalization_layers):
                    weights_without_decay.append(param)
                    # g[1].append(param)
                else:
                    weights_with_decay.append(param)
                    # g[0].append(param)

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
