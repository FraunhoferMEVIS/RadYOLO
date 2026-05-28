from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import yaml
from torch.utils.data import SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from radyolo.models.layers import ModelOutput
from radyolo.models.model import build_model_from_config
from radyolo.training.train import (
    LOG_FILE_ENV,
    LOG_LEVEL_ENV,
    RADYOLO_LOGGER_NAME,
    Trainer,
    _configure_process_logging,
    _distributed_train_worker,
    _resolve_process_log_level,
    _select_distributed_backend,
)
from radyolo.utils.logging_utils import EpochWarningStats
from radyolo.utils.util import ModelEMA, select_device, wrap_model_for_training


class RecordingLoss:
    def __init__(self, loss_fn, *, num_loss_terms: int = 5) -> None:
        self.loss_fn = loss_fn
        self.num_loss_terms = num_loss_terms

    def __call__(self, predictions, batch, current_epoch):
        return self.loss_fn(predictions, batch, current_epoch)

    def reset_epoch_warning_stats(self) -> None:
        return None

    def pop_epoch_warning_stats(self) -> EpochWarningStats:
        return EpochWarningStats()


def test_select_device_accepts_multiple_cuda_devices(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(name="Fake GPU", total_memory=8 * (1 << 30)),
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    device = select_device("0,1", verbose=False)

    assert device == torch.device("cuda:0")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_select_distributed_backend_prefers_nccl_when_available(monkeypatch) -> None:
    monkeypatch.setattr("radyolo.training.train.dist.is_nccl_available", lambda: True)

    assert _select_distributed_backend() == "nccl"


def test_select_distributed_backend_falls_back_to_gloo(monkeypatch) -> None:
    monkeypatch.setattr("radyolo.training.train.dist.is_nccl_available", lambda: False)

    assert _select_distributed_backend() == "gloo"


def test_distributed_train_worker_logs_start(caplog, monkeypatch) -> None:
    trainer_calls: list[str] = []
    env_vars = (
        "RADYOLO_DDP_LOCAL_RANK",
        "RADYOLO_DDP_RANK",
        "RADYOLO_DDP_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    )
    previous_env = {env_var: os.environ.get(env_var) for env_var in env_vars}

    class FakeTrainer:
        def __init__(self, cfg: str) -> None:
            trainer_calls.append(cfg)

        def train(self) -> None:
            trainer_calls.append("train")

    monkeypatch.setattr("radyolo.training.train.get_cfg", lambda *_args, **_kwargs: SimpleNamespace(verbose=True))
    monkeypatch.setattr("radyolo.training.train.Trainer", FakeTrainer)
    for env_var in env_vars:
        monkeypatch.delenv(env_var, raising=False)

    try:
        with caplog.at_level("INFO"):
            _distributed_train_worker(
                local_rank=1,
                world_size=2,
                config_path="fake-config.yaml",
                master_addr="127.0.0.1",
                master_port=23456,
            )

        assert trainer_calls == ["fake-config.yaml", "train"]
        assert (
            "Started distributed train worker rank=1/2 with master=127.0.0.1:23456 "
            "for config=fake-config.yaml."
        ) in caplog.text
        assert os.environ["RADYOLO_DDP_LOCAL_RANK"] == "1"
        assert os.environ["RADYOLO_DDP_RANK"] == "1"
        assert os.environ["RADYOLO_DDP_WORLD_SIZE"] == "2"
    finally:
        for env_var, previous_value in previous_env.items():
            if previous_value is None:
                os.environ.pop(env_var, None)
            else:
                os.environ[env_var] = previous_value


def test_resolve_process_log_level_prefers_env_override(monkeypatch) -> None:
    monkeypatch.setenv(LOG_LEVEL_ENV, str(logging.DEBUG))

    assert _resolve_process_log_level(verbose=False) == logging.DEBUG


def test_configure_process_logging_refreshes_managed_handler_levels(monkeypatch) -> None:
    logger = logging.getLogger(RADYOLO_LOGGER_NAME)
    tqdm_logger = logging.getLogger("tqdm_loggable.tqdm_logging")
    original_handlers = list(logger.handlers)
    original_tqdm_handlers = list(tqdm_logger.handlers)
    monkeypatch.setenv(LOG_LEVEL_ENV, str(logging.DEBUG))

    try:
        _configure_process_logging(verbose=False)

        managed_handlers = [
            handler
            for handler in logger.handlers
            if getattr(handler, "_radyolo_managed_handler", False)
        ]
        assert managed_handlers
        assert any(
            isinstance(handler, logging.StreamHandler) and handler.level == logging.DEBUG
            for handler in managed_handlers
        )
        assert tqdm_logger.handlers
        assert all(handler.level == logging.INFO for handler in tqdm_logger.handlers)
    finally:
        for handler in logger.handlers:
            if handler not in original_handlers:
                handler.close()
        logger.handlers = original_handlers
        for handler in tqdm_logger.handlers:
            if handler not in original_tqdm_handlers:
                handler.close()
        tqdm_logger.handlers = original_tqdm_handlers


def test_distributed_rank_zero_writes_non_empty_log_file(
    minimal_train_config,
    monkeypatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "distributed_logging.yaml"
    with minimal_train_config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["device"] = "0,1"
    config["output_folder_base"] = str(tmp_path / "runs")
    config["experiment_name"] = "distributed-log-test"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    logger = logging.getLogger(RADYOLO_LOGGER_NAME)
    original_handlers = list(logger.handlers)

    monkeypatch.setattr("radyolo.training.train.select_device", lambda _device: torch.device("cuda:0"))
    monkeypatch.setenv("RADYOLO_DDP_LOCAL_RANK", "0")
    monkeypatch.setenv("RADYOLO_DDP_RANK", "0")
    monkeypatch.setenv("RADYOLO_DDP_WORLD_SIZE", "2")
    monkeypatch.setenv(LOG_LEVEL_ENV, str(logging.INFO))
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr("radyolo.training.train.dist.init_process_group", lambda **_kwargs: None)
    monkeypatch.setattr("radyolo.training.train.dist.is_initialized", lambda: False)

    try:
        trainer = Trainer(cfg=config_path)
        log_files = sorted(trainer.save_dir.glob("*.log"))
        managed_file_handlers = [
            handler
            for handler in logger.handlers
            if getattr(handler, "_radyolo_managed_handler", False)
            and isinstance(handler, logging.FileHandler)
        ]

        assert len(log_files) == 1
        assert managed_file_handlers
        assert all(handler.level == logging.INFO for handler in managed_file_handlers)
        assert os.environ[LOG_FILE_ENV] == str(log_files[0])
        assert "Experiment folder:" in log_files[0].read_text(encoding="utf-8")
    finally:
        for handler in logger.handlers:
            if handler not in original_handlers:
                handler.close()
        logger.handlers = original_handlers


def test_launch_distributed_training_preserves_configured_log_level(
    minimal_train_config,
    monkeypatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "distributed_launch.yaml"
    with minimal_train_config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["device"] = "0,1"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    monkeypatch.setattr("radyolo.training.train.select_device", lambda _device: torch.device("cuda:0"))
    monkeypatch.setenv(LOG_LEVEL_ENV, str(logging.INFO))
    monkeypatch.setattr("radyolo.training.train._find_free_port", lambda: 12345)

    captured: dict[str, object] = {}

    def fake_spawn(fn, args, nprocs, join):
        captured["fn"] = fn
        captured["args"] = args
        captured["nprocs"] = nprocs
        captured["join"] = join

    monkeypatch.setattr("radyolo.training.train.mp.spawn", fake_spawn)

    trainer = Trainer(cfg=config_path)
    trainer._launch_distributed_training()

    assert captured["nprocs"] == 2
    assert os.environ[LOG_LEVEL_ENV] == str(logging.INFO)


def test_distributed_nonzero_rank_writes_rank_specific_log_file(
    minimal_train_config,
    monkeypatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "distributed_logging_rank1.yaml"
    with minimal_train_config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["device"] = "0,1"
    config["output_folder_base"] = str(tmp_path / "runs")
    config["experiment_name"] = "distributed-log-rank1-test"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    logger = logging.getLogger(RADYOLO_LOGGER_NAME)
    original_handlers = list(logger.handlers)

    monkeypatch.setattr("radyolo.training.train.select_device", lambda _device: torch.device("cuda:0"))
    monkeypatch.setenv("RADYOLO_DDP_LOCAL_RANK", "1")
    monkeypatch.setenv("RADYOLO_DDP_RANK", "1")
    monkeypatch.setenv("RADYOLO_DDP_WORLD_SIZE", "2")
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr("radyolo.training.train.dist.init_process_group", lambda **_kwargs: None)
    monkeypatch.setattr("radyolo.training.train.dist.is_initialized", lambda: False)

    try:
        trainer = Trainer(cfg=config_path)
        log_files = sorted(trainer.save_dir.glob("*.log"))

        assert len(log_files) == 1
        assert log_files[0].name.endswith("-rank1.log")
        assert os.environ[LOG_FILE_ENV] == str(log_files[0])
    finally:
        for handler in logger.handlers:
            if handler not in original_handlers:
                handler.close()
        logger.handlers = original_handlers


def test_trainer_wraps_training_model_for_multiple_gpus_with_ddp(
    minimal_train_config,
    monkeypatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "multi_gpu_train.yaml"
    with minimal_train_config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["device"] = "0,1"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    monkeypatch.setattr("radyolo.training.train.select_device", lambda _device: torch.device("cuda:0"))
    monkeypatch.setenv("RADYOLO_DDP_LOCAL_RANK", "1")
    monkeypatch.setenv("RADYOLO_DDP_RANK", "1")
    monkeypatch.setenv("RADYOLO_DDP_WORLD_SIZE", "2")
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr("radyolo.training.train.dist.init_process_group", lambda **_kwargs: None)
    monkeypatch.setattr("radyolo.training.train.dist.is_initialized", lambda: False)

    class FakeDistributedDataParallel(torch.nn.Module):
        def __init__(self, module: torch.nn.Module, device_ids: list[int], output_device: int) -> None:
            super().__init__()
            self.module = module
            self.device_ids = device_ids
            self.output_device = output_device

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    monkeypatch.setattr(
        "radyolo.utils.util.DistributedDataParallel",
        FakeDistributedDataParallel,
    )

    trainer = Trainer(cfg=config_path)
    model = build_model_from_config(trainer.config, num_classes=trainer.data.nc)

    wrapped = wrap_model_for_training(
        model,
        distributed=trainer.distributed,
        rank=trainer.rank,
        local_rank=trainer.local_rank,
    )

    assert isinstance(wrapped, FakeDistributedDataParallel)
    assert wrapped.module is model
    assert wrapped.device_ids == [1]
    assert wrapped.output_device == 1


def test_trainer_uses_distributed_sampler_for_training_loader(
    minimal_train_config,
    monkeypatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "distributed_loader.yaml"
    with minimal_train_config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["device"] = "0,1"
    config["workers"] = 4
    config["batch_size"] = 2
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    monkeypatch.setattr("radyolo.training.train.select_device", lambda _device: torch.device("cuda:0"))
    monkeypatch.setenv("RADYOLO_DDP_LOCAL_RANK", "1")
    monkeypatch.setenv("RADYOLO_DDP_RANK", "1")
    monkeypatch.setenv("RADYOLO_DDP_WORLD_SIZE", "2")
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr("radyolo.training.train.dist.init_process_group", lambda **_kwargs: None)
    monkeypatch.setattr("radyolo.training.train.dist.is_initialized", lambda: False)

    trainer = Trainer(cfg=config_path)
    model = build_model_from_config(trainer.config, num_classes=trainer.data.nc)

    train_loader = trainer.get_dataloader(trainer.config.data, model, fold=0, mode="train")
    val_loader = trainer.get_dataloader(trainer.config.data, model, fold=0, mode="val")

    assert isinstance(train_loader.sampler, DistributedSampler)
    assert train_loader.batch_size == 2
    assert train_loader.num_workers == 4
    assert not isinstance(val_loader.sampler, DistributedSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)


def test_distributed_any_uses_collective_max(
    minimal_train_config,
    monkeypatch,
) -> None:
    trainer = Trainer(cfg=minimal_train_config)
    trainer.distributed = True
    trainer.rank = 1
    trainer.world_size = 2
    trainer.device = torch.device("cpu")

    def _all_reduce(tensor: torch.Tensor, op=None) -> None:
        del op
        tensor.fill_(1)

    monkeypatch.setattr("radyolo.training.train.dist.is_initialized", lambda: True)
    monkeypatch.setattr("radyolo.training.train.dist.get_backend", lambda *_args, **_kwargs: "gloo")
    monkeypatch.setattr("radyolo.training.train.dist.all_reduce", _all_reduce)

    assert trainer._distributed_any(False) is True


def test_distributed_zero_loss_surrogate_keeps_model_output_graph_connected(
    minimal_train_config,
) -> None:
    trainer = Trainer(cfg=minimal_train_config)
    trainer.distributed = True
    trainer.device = torch.device("cpu")

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            self.mask_weight = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
            self.label_weight = torch.nn.Parameter(torch.tensor(3.0, dtype=torch.float32))

        def forward(self) -> ModelOutput:
            raw_prediction = self.weight.unsqueeze(0)
            mask_coefficients = self.mask_weight.view(1, 1, 1)
            prototype_masks = self.mask_weight.view(1, 1, 1, 1, 1)
            return ModelOutput(
                raw_predictions=[raw_prediction],
                mask_coefficients=mask_coefficients,
                prototype_masks=prototype_masks,
                image_label_outputs={"label": self.label_weight.view(1, 1)},
            )

    model = FakeModel()
    zero_loss = trainer._build_distributed_safe_zero_loss(model.forward())
    zero_loss.backward()

    assert torch.equal(model.weight.grad, torch.zeros_like(model.weight))
    assert torch.equal(model.mask_weight.grad, torch.zeros_like(model.mask_weight))
    assert torch.equal(model.label_weight.grad, torch.zeros_like(model.label_weight))


def test_trainer_accepts_non_divisible_distributed_batch_size_per_gpu(
    minimal_train_config,
    monkeypatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "bad_distributed_batch.yaml"
    with minimal_train_config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["device"] = "0,1"
    config["batch_size"] = 3
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    monkeypatch.setattr("radyolo.training.train.select_device", lambda _device: torch.device("cuda:0"))

    trainer = Trainer(cfg=config_path)

    trainer._validate_distributed_setup()


def test_trainer_uses_per_gpu_batch_size_without_loss_rescaling(
    minimal_train_config,
    monkeypatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "distributed_loss_scale.yaml"
    with minimal_train_config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["device"] = "0,1"
    config["batch_size"] = 2
    config["workers"] = 3
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    monkeypatch.setattr("radyolo.training.train.select_device", lambda _device: torch.device("cuda:0"))
    monkeypatch.setenv("RADYOLO_DDP_LOCAL_RANK", "1")
    monkeypatch.setenv("RADYOLO_DDP_RANK", "1")
    monkeypatch.setenv("RADYOLO_DDP_WORLD_SIZE", "2")
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr("radyolo.training.train.dist.init_process_group", lambda **_kwargs: None)
    monkeypatch.setattr("radyolo.training.train.dist.is_initialized", lambda: False)

    distributed_trainer = Trainer(cfg=config_path)
    single_process_loss = torch.tensor(3.5)

    assert distributed_trainer._get_per_process_batch_size() == 2
    assert distributed_trainer._get_per_process_workers() == 3
    assert distributed_trainer._get_effective_batch_size() == 4
    assert distributed_trainer._scale_loss_for_backward(single_process_loss) is single_process_loss

    monkeypatch.delenv("RADYOLO_DDP_LOCAL_RANK", raising=False)
    monkeypatch.delenv("RADYOLO_DDP_RANK", raising=False)
    monkeypatch.delenv("RADYOLO_DDP_WORLD_SIZE", raising=False)

    single_gpu_trainer = Trainer(cfg=config_path)

    assert single_gpu_trainer._get_effective_batch_size() == 2
    assert single_gpu_trainer._scale_loss_for_backward(single_process_loss) is single_process_loss


def test_trainer_broadcast_stop_uses_cuda_tensor_for_nccl(
    minimal_train_config,
    monkeypatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "distributed_stop_broadcast.yaml"
    with minimal_train_config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["device"] = "0,1"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    monkeypatch.setattr("radyolo.training.train.select_device", lambda _device: torch.device("cuda:0"))
    monkeypatch.setenv("RADYOLO_DDP_LOCAL_RANK", "1")
    monkeypatch.setenv("RADYOLO_DDP_RANK", "1")
    monkeypatch.setenv("RADYOLO_DDP_WORLD_SIZE", "2")
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr("radyolo.training.train.dist.is_initialized", lambda: True)
    monkeypatch.setattr("radyolo.training.train.dist.get_backend", lambda: "nccl")

    captured: dict[str, object] = {}

    class FakeTensor:
        def __init__(self, value: int, device: torch.device | str) -> None:
            self.value = value
            self.device = torch.device(device)

        def item(self) -> int:
            return self.value

    def fake_tensor(value: int, device: torch.device | str) -> FakeTensor:
        captured["tensor_device"] = torch.device(device)
        return FakeTensor(value, device)

    def fake_broadcast(tensor: FakeTensor, src: int) -> None:
        captured["broadcast_device"] = tensor.device
        captured["src"] = src

    monkeypatch.setattr("radyolo.training.train.torch.tensor", fake_tensor)
    monkeypatch.setattr("radyolo.training.train.dist.broadcast", fake_broadcast)

    trainer = Trainer(cfg=config_path)

    assert trainer._broadcast_stop(stop=True) is True
    assert captured["tensor_device"] == torch.device("cuda:1")
    assert captured["broadcast_device"] == torch.device("cuda:1")
    assert captured["src"] == 0


def test_trainer_tracks_peak_epoch_memory_from_sampled_readings(
    minimal_train_config,
    monkeypatch,
) -> None:
    trainer = Trainer(cfg=minimal_train_config)

    memory_readings = iter((0.25, 0.5, 0.75, 0.6))
    monkeypatch.setattr(trainer, "_get_memory", lambda: next(memory_readings))

    trainer._reset_epoch_peak_memory()
    trainer._update_epoch_peak_memory()
    trainer._update_epoch_peak_memory()
    trainer._update_epoch_peak_memory()

    assert trainer.epoch_peak_memory_gb == pytest.approx(0.75)


def test_trainer_reset_epoch_peak_memory_uses_current_memory_reading(
    minimal_train_config,
    monkeypatch,
) -> None:
    trainer = Trainer(cfg=minimal_train_config)
    trainer.epoch_peak_memory_gb = 1.25
    monkeypatch.setattr(trainer, "_get_memory", lambda: 0.4)

    trainer._reset_epoch_peak_memory()

    assert trainer.epoch_peak_memory_gb == pytest.approx(0.4)


def test_model_ema_updates_from_wrapped_model() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    ema = ModelEMA(model, decay=0.5, tau=1.0)
    with torch.no_grad():
        model.weight.fill_(2.0)

    class WrappedModule(torch.nn.Module):
        def __init__(self, module: torch.nn.Module) -> None:
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    ema.update(WrappedModule(model))
    ema_weight = cast(torch.nn.Linear, ema.ema).weight.detach()

    assert torch.all(ema_weight > 0.0)
    assert torch.all(ema_weight < 2.0)


def test_trainer_updates_ema_once_per_epoch(minimal_train_config, monkeypatch) -> None:
    trainer = Trainer(cfg=minimal_train_config)
    trainer.config = trainer.config.model_copy(
        update={
            "epochs": 2,
            "warmup_epochs": 0.0,
            "val": False,
            "save": False,
            "plots": False,
            "verbose": False,
        }
    )
    trainer.epochs = trainer.config.epochs
    trainer.loss_names = (
        "box_loss",
        "cls_obj_loss",
        "dfl_loss",
        "mask_loss",
        "cls_sep_loss",
    )
    trainer.loss = RecordingLoss(
        lambda _predictions, _batch, current_epoch: (
            torch.tensor(float(current_epoch), dtype=torch.float32, requires_grad=True),
            torch.zeros(5, dtype=torch.float32),
        )
    )

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        def forward(self, _img: torch.Tensor) -> tuple[torch.Tensor]:
            return (self.weight.unsqueeze(0),)

    class RecordingEMA:
        def __init__(self) -> None:
            self.calls: list[float] = []
            self.epoch_advances = 0

        def update(self, model: torch.nn.Module) -> None:
            self.calls.append(float(cast(FakeModel, model).weight.detach().item()))

        def advance_updates(self) -> None:
            self.epoch_advances += 1

    class DummyTrainLoader:
        def __init__(self, batches: list[dict[str, torch.Tensor]]) -> None:
            self._batches = batches
            self.sampler = object()

        def __len__(self) -> int:
            return len(self._batches)

        def __iter__(self):
            return iter(self._batches)

    model = FakeModel()
    ema = RecordingEMA()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.GradScaler("cpu", enabled=False)
    train_loader = DummyTrainLoader(
        [{"img": torch.ones(1, dtype=torch.float32)} for _ in range(2)]
    )

    monkeypatch.setattr("radyolo.training.train.torch.autocast", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(trainer, "_prepare_training_batch", lambda batch, _run_gpu_augmentations: batch)
    monkeypatch.setattr(trainer, "_barrier", lambda: None)
    monkeypatch.setattr(trainer, "_broadcast_stop", lambda stop: stop)
    monkeypatch.setattr(trainer, "_clear_memory", lambda: None)
    monkeypatch.setattr(trainer, "_get_memory", lambda: 0.0)
    monkeypatch.setattr(trainer, "save_metrics", lambda metrics, train_time_start: None)
    monkeypatch.setattr(trainer, "save_model", lambda **_kwargs: None)

    trainer._train_loop(
        model=model,
        train_loader=train_loader,
        validator=None,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0),
        scaler=scaler,
        lf=lambda _epoch: 1.0,
        ema=cast(ModelEMA, ema),
        accumulate=1,
        stopper=lambda _epoch, _fitness: False,
        stop=False,
        best_fitness=None,
        start_epoch=0,
    )

    expected_step_updates = 0
    last_opt_step = len(train_loader) * 0
    for epoch in range(trainer.config.epochs):
        for batch_index in range(len(train_loader)):
            num_iterations = batch_index + len(train_loader) * epoch
            if num_iterations - last_opt_step >= 1:
                expected_step_updates += 1
                last_opt_step = num_iterations

    assert len(ema.calls) == expected_step_updates
    assert ema.epoch_advances == trainer.config.epochs


def test_trainer_skips_non_finite_loss_batch(minimal_train_config, monkeypatch) -> None:
    trainer = Trainer(cfg=minimal_train_config)
    trainer.config = trainer.config.model_copy(
        update={
            "epochs": 1,
            "warmup_epochs": 0.0,
            "val": False,
            "save": False,
            "plots": False,
            "verbose": False,
        }
    )
    trainer.epochs = trainer.config.epochs
    trainer.loss_names = (
        "box_loss",
        "cls_obj_loss",
        "dfl_loss",
        "mask_loss",
        "cls_sep_loss",
    )

    loss_calls = {"count": 0}

    def _loss(_predictions, _batch, current_epoch):
        del current_epoch
        loss_calls["count"] += 1
        if loss_calls["count"] == 1:
            return (
                torch.tensor(float("nan"), dtype=torch.float32, requires_grad=True),
                torch.full((5,), float("nan"), dtype=torch.float32),
            )
        return (
            torch.tensor(1.0, dtype=torch.float32, requires_grad=True),
            torch.ones(5, dtype=torch.float32),
        )

    trainer.loss = RecordingLoss(_loss)

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        def forward(self, _img: torch.Tensor) -> tuple[torch.Tensor]:
            return (self.weight.unsqueeze(0),)

    class RecordingEMA:
        def __init__(self) -> None:
            self.calls = 0
            self.epoch_advances = 0

        def update(self, _model: torch.nn.Module) -> None:
            self.calls += 1

        def advance_updates(self) -> None:
            self.epoch_advances += 1

    class DummyTrainLoader:
        def __init__(self, batches: list[dict[str, torch.Tensor | list[str]]]) -> None:
            self._batches = batches
            self.sampler = object()

        def __len__(self) -> int:
            return len(self._batches)

        def __iter__(self):
            return iter(self._batches)

    model = FakeModel()
    ema = RecordingEMA()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.GradScaler("cpu", enabled=False)
    train_loader = DummyTrainLoader(
        [
            {"img": torch.ones(1, dtype=torch.float32), "case_names": ["bad-case"]},
            {"img": torch.ones(1, dtype=torch.float32), "case_names": ["good-case"]},
        ]
    )

    optimizer_step_calls = {"count": 0}

    def _optimizer_step(*args, **kwargs):
        del args, kwargs
        optimizer_step_calls["count"] += 1
        return True

    monkeypatch.setattr("radyolo.training.train.torch.autocast", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(trainer, "_prepare_training_batch", lambda batch, _run_gpu_augmentations: batch)
    monkeypatch.setattr(trainer, "optimizer_step", _optimizer_step)
    monkeypatch.setattr(trainer, "_barrier", lambda: None)
    monkeypatch.setattr(trainer, "_broadcast_stop", lambda stop: stop)
    monkeypatch.setattr(trainer, "_clear_memory", lambda: None)
    monkeypatch.setattr(trainer, "_get_memory", lambda: 0.0)
    monkeypatch.setattr(trainer, "save_metrics", lambda metrics, train_time_start: None)
    monkeypatch.setattr(trainer, "save_model", lambda **_kwargs: None)

    trainer._train_loop(
        model=model,
        train_loader=train_loader,
        validator=None,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0),
        scaler=scaler,
        lf=lambda _epoch: 1.0,
        ema=cast(ModelEMA, ema),
        accumulate=1,
        stopper=lambda _epoch, _fitness: False,
        stop=False,
        best_fitness=None,
        start_epoch=0,
    )

    assert loss_calls["count"] == 2
    assert optimizer_step_calls["count"] == 1
    assert ema.epoch_advances == 1


def test_trainer_skips_non_finite_input_batch(
    minimal_train_config,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    trainer = Trainer(cfg=minimal_train_config)
    trainer.config = trainer.config.model_copy(
        update={
            "epochs": 1,
            "warmup_epochs": 0.0,
            "val": False,
            "save": False,
            "plots": False,
            "verbose": False,
        }
    )
    trainer.epochs = trainer.config.epochs
    trainer.loss_names = (
        "box_loss",
        "cls_obj_loss",
        "dfl_loss",
        "mask_loss",
        "cls_sep_loss",
    )

    loss_calls = {"count": 0}

    def _loss(_predictions, _batch, current_epoch):
        del current_epoch
        loss_calls["count"] += 1
        return (
            torch.tensor(1.0, dtype=torch.float32, requires_grad=True),
            torch.ones(5, dtype=torch.float32),
        )

    trainer.loss = RecordingLoss(_loss)

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            self.forward_calls = 0

        def forward(self, _img: torch.Tensor) -> tuple[torch.Tensor]:
            self.forward_calls += 1
            return (self.weight.unsqueeze(0),)

    class RecordingEMA:
        def __init__(self) -> None:
            self.calls = 0
            self.epoch_advances = 0

        def update(self, _model: torch.nn.Module) -> None:
            self.calls += 1

        def advance_updates(self) -> None:
            self.epoch_advances += 1

    class DummyTrainLoader:
        def __init__(self, batches: list[dict[str, torch.Tensor | list[str]]]) -> None:
            self._batches = batches
            self.sampler = object()

        def __len__(self) -> int:
            return len(self._batches)

        def __iter__(self):
            return iter(self._batches)

    model = FakeModel()
    ema = RecordingEMA()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.GradScaler("cpu", enabled=False)
    train_loader = DummyTrainLoader(
        [
            {"img": torch.tensor([float("nan")], dtype=torch.float32), "case_names": ["bad-case"]},
            {"img": torch.ones(1, dtype=torch.float32), "case_names": ["good-case"]},
        ]
    )

    optimizer_step_calls = {"count": 0}

    def _optimizer_step(*args, **kwargs):
        del args, kwargs
        optimizer_step_calls["count"] += 1
        return True

    monkeypatch.setattr("radyolo.training.train.torch.autocast", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(trainer, "_prepare_training_batch", lambda batch, _run_gpu_augmentations: batch)
    monkeypatch.setattr(trainer, "optimizer_step", _optimizer_step)
    monkeypatch.setattr(trainer, "_barrier", lambda: None)
    monkeypatch.setattr(trainer, "_broadcast_stop", lambda stop: stop)
    monkeypatch.setattr(trainer, "_clear_memory", lambda: None)
    monkeypatch.setattr(trainer, "_get_memory", lambda: 0.0)
    monkeypatch.setattr(trainer, "save_metrics", lambda metrics, train_time_start: None)
    monkeypatch.setattr(trainer, "save_model", lambda **_kwargs: None)

    with caplog.at_level(logging.WARNING):
        trainer._train_loop(
            model=model,
            train_loader=train_loader,
            validator=None,
            optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda _epoch: 1.0
            ),
            scaler=scaler,
            lf=lambda _epoch: 1.0,
            ema=cast(ModelEMA, ema),
            accumulate=1,
            stopper=lambda _epoch, _fitness: False,
            stop=False,
            best_fitness=None,
            start_epoch=0,
        )

    assert loss_calls["count"] == 1
    assert model.forward_calls == 1
    assert optimizer_step_calls["count"] == 1
    assert ema.epoch_advances == 1
    assert "input image tensor became non-finite" in caplog.text
    assert "bad-case" in caplog.text


def test_distributed_non_finite_loss_uses_zero_backward_surrogate(
    minimal_train_config,
    monkeypatch,
) -> None:
    trainer = Trainer(cfg=minimal_train_config)
    trainer.config = trainer.config.model_copy(
        update={
            "epochs": 1,
            "warmup_epochs": 0.0,
            "val": False,
            "save": False,
            "plots": False,
            "verbose": False,
        }
    )
    trainer.epochs = trainer.config.epochs
    trainer.distributed = True
    trainer.rank = 1
    trainer.world_size = 2
    trainer.is_main_process = False
    trainer.device = torch.device("cpu")
    trainer.loss_names = (
        "box_loss",
        "cls_obj_loss",
        "dfl_loss",
        "mask_loss",
        "cls_sep_loss",
    )

    loss_calls = {"count": 0}

    def _loss(_predictions, _batch, current_epoch):
        del current_epoch
        loss_calls["count"] += 1
        return (
            torch.tensor(float("nan"), dtype=torch.float32, requires_grad=True),
            torch.full((5,), float("nan"), dtype=torch.float32),
        )

    trainer.loss = RecordingLoss(_loss)

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            self.mask_weight = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float32))

        def forward(self, _img: torch.Tensor) -> ModelOutput:
            raw_prediction = self.weight.unsqueeze(0)
            mask_coefficients = self.mask_weight.view(1, 1, 1)
            prototype_masks = self.mask_weight.view(1, 1, 1, 1, 1)
            return ModelOutput(
                raw_predictions=[raw_prediction],
                mask_coefficients=mask_coefficients,
                prototype_masks=prototype_masks,
            )

    class RecordingEMA:
        def __init__(self) -> None:
            self.calls = 0
            self.epoch_advances = 0

        def update(self, _model: torch.nn.Module) -> None:
            self.calls += 1

        def advance_updates(self) -> None:
            self.epoch_advances += 1

    class DummyTrainLoader:
        def __init__(self, batches: list[dict[str, torch.Tensor | list[str]]]) -> None:
            self._batches = batches
            self.sampler = object()

        def __len__(self) -> int:
            return len(self._batches)

        def __iter__(self):
            return iter(self._batches)

    model = FakeModel()
    ema = RecordingEMA()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.GradScaler("cpu", enabled=False)
    train_loader = DummyTrainLoader(
        [{"img": torch.ones(1, dtype=torch.float32), "case_names": ["bad-case"]}]
    )

    optimizer_step_calls = {"count": 0}
    all_reduce_calls = {"count": 0}

    def _optimizer_step(*args, **kwargs):
        del args, kwargs
        optimizer_step_calls["count"] += 1
        return True

    def _all_reduce(tensor: torch.Tensor, op=None) -> None:
        del op
        all_reduce_calls["count"] += 1
        tensor.fill_(1 if all_reduce_calls["count"] > 1 else 0)

    monkeypatch.setattr("radyolo.training.train.torch.autocast", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(trainer, "_prepare_training_batch", lambda batch, _run_gpu_augmentations: batch)
    monkeypatch.setattr(trainer, "optimizer_step", _optimizer_step)
    monkeypatch.setattr(trainer, "_barrier", lambda: None)
    monkeypatch.setattr(
        trainer,
        "_broadcast_stop",
        lambda stop: stop or trainer.epoch + 1 >= trainer.epochs,
    )
    monkeypatch.setattr(trainer, "_clear_memory", lambda: None)
    monkeypatch.setattr(trainer, "_get_memory", lambda: 0.0)
    monkeypatch.setattr(trainer, "_log_epoch_warning_summary", lambda: None)
    monkeypatch.setattr(trainer, "save_metrics", lambda metrics, train_time_start: None)
    monkeypatch.setattr(trainer, "save_model", lambda **_kwargs: None)
    monkeypatch.setattr("radyolo.training.train.dist.is_initialized", lambda: True)
    monkeypatch.setattr("radyolo.training.train.dist.get_backend", lambda *_args, **_kwargs: "gloo")
    monkeypatch.setattr("radyolo.training.train.dist.all_reduce", _all_reduce)

    trainer._train_loop(
        model=model,
        train_loader=train_loader,
        validator=None,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0),
        scaler=scaler,
        lf=lambda _epoch: 1.0,
        ema=cast(ModelEMA, ema),
        accumulate=1,
        stopper=lambda _epoch, _fitness: False,
        stop=False,
        best_fitness=None,
        start_epoch=0,
    )

    assert loss_calls["count"] == 1
    assert optimizer_step_calls["count"] == 0
    assert ema.calls == 0
    assert ema.epoch_advances == 1
    assert all_reduce_calls["count"] >= 2


def test_model_ema_advance_updates_controls_epoch_counter() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    ema = ModelEMA(model, decay=0.5, tau=1.0)

    with torch.no_grad():
        model.weight.fill_(2.0)

    ema.update(model)

    assert ema.updates == 0

    ema.advance_updates()

    assert ema.updates == 1


def test_model_ema_reverts_non_finite_buffers(caplog: pytest.LogCaptureFixture) -> None:
    model = torch.nn.BatchNorm1d(4)
    ema = ModelEMA(model, decay=0.5, tau=1.0)
    previous_running_mean = ema.ema.running_mean.detach().clone()
    previous_running_var = ema.ema.running_var.detach().clone()

    with torch.no_grad():
        model.running_mean.fill_(float("nan"))
        model.running_var.fill_(float("inf"))

    with caplog.at_level(logging.WARNING):
        ema.update(model)

    assert torch.equal(ema.ema.running_mean, previous_running_mean)
    assert torch.equal(ema.ema.running_var, previous_running_var)
    assert "EMA state became non-finite" in caplog.text