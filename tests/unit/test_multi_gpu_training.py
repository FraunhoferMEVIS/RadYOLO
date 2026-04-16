from __future__ import annotations

import os
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import yaml
from torch.utils.data import SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from radyolo.models.model import build_model_from_config
from radyolo.training.train import (
    Trainer,
    _distributed_train_worker,
    _select_distributed_backend,
)
from radyolo.utils.util import ModelEMA, wrap_model_for_training, select_device


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
    assert train_loader.batch_size == 1
    assert train_loader.num_workers == 2
    assert isinstance(val_loader.sampler, SequentialSampler)


def test_trainer_rejects_non_divisible_distributed_batch_size(
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

    with pytest.raises(ValueError, match="batch_size must be divisible"):
        trainer._validate_distributed_setup()


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