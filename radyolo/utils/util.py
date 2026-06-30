from __future__ import annotations

import logging
import math
import os
import platform
import random
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

PYTHON_VERSION = platform.python_version()
NUM_THREADS = min(8, max(1, (os.cpu_count() or 1) - 1))

LOGGER = logging.getLogger(__name__)


def make_divisible(x: float, divisor: int | torch.Tensor) -> int:
    """Returns the nearest number that is divisible by the given divisor.

    Args:
        x (int): The number to make divisible.
        divisor (int | torch.Tensor): The divisor.

    Returns:
        (int): The nearest number divisible by the divisor.
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor


def intersect_dicts(
    a: dict[str, Any], b: dict[str, Any], exclude: tuple[str, ...] = ()
) -> dict[str, Any]:
    return {
        key: value
        for key, value in a.items()
        if key in b and all(x not in key for x in exclude) and value.shape == b[key].shape
    }


def _normalize_device_string(device: str) -> str:
    normalized = str(device).lower()
    for remove in ("cuda:", "none", "(", ")", "[", "]", "'", '"', " "):
        normalized = normalized.replace(remove, "")
    return normalized


def parse_cuda_device_ids(device: str | torch.device = "") -> list[int]:
    if isinstance(device, torch.device):
        if device.type != "cuda":
            return []
        return [0 if device.index is None else int(device.index)]

    normalized = _normalize_device_string(device)
    if normalized in {"", "-1", "cpu", "mps", "mps:0"}:
        return []
    if normalized == "cuda":
        return [0]
    return [int(index) for index in normalized.split(",") if index]


def wrap_model_for_training(
    model: nn.Module,
    *,
    distributed: bool,
    rank: int,
    local_rank: int,
) -> nn.Module:
    """Wrap a model for distributed training when DDP is enabled."""
    if not distributed:
        return model

    LOGGER.info(
        "Rank %d enabling DistributedDataParallel on cuda:%d.",
        rank,
        local_rank,
    )
    return DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module from a wrapped training model."""
    module = getattr(model, "module", None)
    if isinstance(module, nn.Module):
        return module
    return model


def select_device(
    device: str | torch.device = "",
) -> torch.device:
    if isinstance(device, torch.device):
        return device

    version_message = f"Version info - Python: {PYTHON_VERSION} torch: {torch.__version__}"
    LOGGER.info(version_message)
    device = _normalize_device_string(str(device))

    cpu = device == "cpu"
    mps = device in {"mps", "mps:0"}
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device:
        if device == "cuda":
            device = "0"
        if "," in device:
            device = ",".join([x for x in device.split(",") if x])
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.split(","))):
            msg = (
                f"Invalid CUDA 'device={device}' requested"
                f" Use 'device=cpu' or pass valid CUDA device(s) if available,"
                f" i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n"
                f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}"
                f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n"
            )
            raise ValueError(msg)

    device_message = "Device(s): "
    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(",") if device else ["0"]
        for i, d in enumerate(devices):
            device_message += f"CUDA:{d} ({get_gpu_info(i)}), "  # bytes to MB
        arg = "cuda:0"
    elif mps and torch.backends.mps.is_available():
        # Prefer MPS if available
        device_message += "MPS"
        arg = "mps"
    else:  # revert to CPU
        device_message += "CPU"
        arg = "cpu"

    if arg in {"cpu", "mps"}:
        torch.set_num_threads(NUM_THREADS)
    LOGGER.info(device_message)
    return torch.device(arg)


def get_gpu_info(index: torch.device | int | str) -> str:
    properties = torch.cuda.get_device_properties(index)
    return f"{properties.name}, {properties.total_memory / (1 << 20):.0f}MiB"


def init_seeds(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU, exception safe
    torch.backends.cudnn.deterministic = False


class ModelEMA:
    """Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models.

    Keeps a moving average of everything in the model state_dict (parameters and buffers).
    During training, call `update()` on each optimizer step and advance `updates`
    once per epoch with `advance_updates()`.

    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    To disable EMA set the `enabled` attribute to `False`.
    """

    def __init__(
        self, model: nn.Module, decay: float = 0.9999, tau: float = 400, updates: int = 0
    ) -> None:
        """Initialize EMA for 'model' with given arguments."""
        self.ema = deepcopy(unwrap_model(model)).eval()  # FP32 EMA
        self.updates = updates  # number of epoch-level EMA updates
        self.decay = lambda x: decay * (
            1 - math.exp(-x / tau)
        )  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model: nn.Module) -> None:
        """Update EMA parameters from the current model weights."""
        if self.enabled:
            next_updates = self.updates + 1
            decay = self.decay(next_updates)

            model_state_dict = unwrap_model(model).state_dict()  # model state_dict
            ema_state_dict = self.ema.state_dict()
            previous_floating_state = {
                key: value.clone()
                for key, value in ema_state_dict.items()
                if value.dtype.is_floating_point
            }
            for key, value in ema_state_dict.items():
                if value.dtype.is_floating_point:  # true for FP16 and FP32
                    value *= decay  # noqa: PLW2901
                    value += (1 - decay) * model_state_dict[key].detach()  # noqa: PLW2901

            if any(
                not torch.isfinite(value).all().item()
                for value in ema_state_dict.values()
                if value.dtype.is_floating_point
            ):
                for key, value in previous_floating_state.items():
                    ema_state_dict[key].copy_(value)
                LOGGER.warning(
                    "Skipping EMA update because EMA state became non-finite after update."
                )
                return

    def advance_updates(self) -> None:
        """Advance the epoch-level EMA update counter."""
        if self.enabled:
            self.updates += 1
