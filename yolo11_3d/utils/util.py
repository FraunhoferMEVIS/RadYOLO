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


def select_device(
    device: str | torch.device = "", _: int = 0, newline: bool = False, verbose: bool = True
) -> torch.device:
    if isinstance(device, torch.device):
        return device

    s = f"yolo11-3d Python-{PYTHON_VERSION} torch-{torch.__version__}"
    device = str(device).lower()
    for remove in "cuda:", "none", "(", ")", "[", "]", "'", " ":
        device.replace(remove, "")  # to string, 'cuda:0' -> '0' and '(0, 1)' -> '0,1'

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
            LOGGER.info(s)
            msg = (
                f"Invalid CUDA 'device={device}' requested"
                f" Use 'device=cpu' or pass valid CUDA device(s) if available,"
                f" i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n"
                f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}"
                f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n"
            )
            raise ValueError(msg)

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(",") if device else "0"
        gpus = len(devices)  # device count
        if gpus > 1:
            raise NotImplementedError
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            s += f"{'' if i == 0 else space}CUDA:{d} ({get_gpu_info(i)})\n"  # bytes to MB
        arg = "cuda:0"
    elif mps and torch.backends.mps.is_available():
        # Prefer MPS if available
        s += "MPS\n"
        arg = "mps"
    else:  # revert to CPU
        s += "CPU\n"
        arg = "cpu"

    if arg in {"cpu", "mps"}:
        torch.set_num_threads(NUM_THREADS)
    if verbose:
        LOGGER.info(s if newline else s.rstrip())
    return torch.device(arg)


def get_gpu_info(index: torch.device | int | str) -> str:
    properties = torch.cuda.get_device_properties(index)
    return f"{properties.name}, {properties.total_memory / (1 << 20):.0f}MiB"


def init_seeds(seed: int = 0, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU, exception safe
    torch.use_deterministic_algorithms(False)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False


class ModelEMA:
    """Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models.

    Keeps a moving average of everything in the model state_dict (parameters and buffers).

    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    To disable EMA set the `enabled` attribute to `False`.
    """

    def __init__(
        self, model: nn.Module, decay: float = 0.9999, tau: float = 2000, updates: int = 0
    ) -> None:
        """Initialize EMA for 'model' with given arguments."""
        self.ema = deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (
            1 - math.exp(-x / tau)
        )  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model: nn.Module) -> None:
        """Update EMA parameters."""
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:  # true for FP16 and FP32
                    v *= d                          # noqa: PLW2901
                    v += (1 - d) * msd[k].detach()  # noqa: PLW2901
