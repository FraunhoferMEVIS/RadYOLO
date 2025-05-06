from __future__ import annotations

import contextlib
import logging
import math
import os
import platform
import random
from collections.abc import Collection, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from yolo11_3d.nn.layers import (
    C2PSA,
    C3,
    SPPF,
    Bottleneck,
    C2f,
    C2f_3D,
    C3k2,
    Concat,
    Conv,
    Detect,
    Detect_3D,
    DWConv,
)

PYTHON_VERSION = platform.python_version()
NUM_THREADS = min(8, max(1, (os.cpu_count() or 1) - 1))

LOGGER = logging.getLogger(__name__)


def parse_model(
    d: dict, input_channels: int, verbose: bool = True
) -> tuple[nn.Sequential, list[int]]:  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (
        d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape")
    )
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = next(iter(scales.keys()))
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_activation = eval(
            act
        )  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{'activation:'} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    channels: list[int] = [input_channels]
    layers: list[nn.Module] = []
    save: list[int] = []
    c2: int = channels[-1]  # layers, savelist, ch out
    for i, (from_layer_index, n, module_name, args) in enumerate(
        d["backbone"] + d["head"]
    ):  # from, number, module, args
        from_layer_index: int | Sequence[int]
        n: int
        args: list
        module_name = (
            getattr(torch.nn, module_name[3:]) if "nn." in module_name else globals()[module_name]
        )  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if module_name in {
            Conv,
            Bottleneck,
            SPPF,
            C2PSA,
            DWConv,
            C2f,
            C3k2,
            C3,
            nn.ConvTranspose2d,
        }:
            if not isinstance(from_layer_index, int):
                raise ValueError  # TODO add error message
            c1, c2 = channels[from_layer_index], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            if module_name in {
                C2f,
                C3k2,
                C3,
                C2PSA,
            }:
                args.insert(2, n)  # number of repeats
                n = 1
            if module_name is C3k2:  # for M/L/X sizes
                if scale in "mlx":
                    args[3] = True
        elif module_name is nn.BatchNorm2d:
            if not isinstance(from_layer_index, int):
                raise ValueError  # TODO add error message
            args = [channels[from_layer_index]]
        elif module_name is Concat:
            if not isinstance(from_layer_index, Sequence):
                raise ValueError  # TODO add error message
            c2 = sum(channels[x] for x in from_layer_index)
        elif module_name is Detect:
            if not isinstance(from_layer_index, Sequence):
                raise ValueError  # TODO add error message
            args.append([channels[x] for x in from_layer_index])
        else:
            if not isinstance(from_layer_index, int):
                raise ValueError  # TODO add error message
            c2 = channels[from_layer_index]

        m_ = (
            nn.Sequential(*(module_name(*args) for _ in range(n))) if n > 1 else module_name(*args)
        )  # module
        t = str(module_name)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, from_layer_index, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(
                f"{i:>3}{str(from_layer_index):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}"
            )  # print
        save.extend(
            x % i
            for x in ([from_layer_index] if isinstance(from_layer_index, int) else from_layer_index)
            if x != -1 and isinstance(x, int)
        )  # append to savelist
        layers.append(m_)
        if i == 0:
            channels = []
        channels.append(c2)
    return nn.Sequential(*layers), sorted(save)


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


def initialize_weights(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            pass
        elif isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03
        elif isinstance(m, nn.Hardswish | nn.LeakyReLU | nn.ReLU | nn.ReLU6 | nn.SiLU):
            m.inplace = True


def scale_img(
    img: torch.Tensor, ratio: float = 1.0, same_shape: bool = False, gs=32
) -> torch.Tensor:
    if ratio == 1.0:
        return img

    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)
    if not same_shape:
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)


def model_to_dict(model: nn.Module) -> dict:
    return {
        "type": type(model).__name__,
        "parameters": {name: param.shape for name, param in model.named_parameters()},
        "children": [model_to_dict(child) for child in model.children()],
    }


def export_model(
    model: nn.Module, input_shape: tuple[int, ...], dest: Path | str, copy: bool = False
) -> Path:
    inputs = torch.randn((1, *input_shape)).to("cpu")
    if copy:
        model = deepcopy(model)
    model.to("cpu")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()

    for _ in range(2):
        model(inputs)

    for m in model.modules():
        if isinstance(m, Detect | Detect_3D):
            m.export = True
        if isinstance(m, C2f | C2f_3D):
            m.forward = m.forward_split

    onnx_path = Path(dest).with_suffix(".onnx")

    torch.onnx.export(
        model,
        inputs,
        onnx_path,
        verbose=False,
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes=None,
    )
    return onnx_path


def select_device(
    device: str | torch.device = "", _: int = 0, newline: bool = False, verbose: bool = True
) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if str(device).startswith("tpu"):
        raise NotImplementedError

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
        n = len(devices)  # device count
        if n > 1:
            raise NotImplementedError
            # if batch < 1:
            #     raise ValueError
            # if batch >= 0 and batch % n != 0:
            #     raise ValueError
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
    """Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models. Keeps a moving
    average of everything in the model state_dict (parameters and buffers).

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
                    v *= d
                    v += (1 - d) * msd[k].detach()
                    # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype},  model {msd[k].dtype}'

    def update_attr(
        self,
        model: nn.Module,
        include: Collection[str] = (),
        exclude: Collection[str] = ("process_group", "reducer"),
    ) -> None:
        """Updates attributes and saves stripped model with optimizer removed."""
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)


def copy_attr(
    a: object, b: object, include: Collection[str] = (), exclude: Collection[str] = ()
) -> None:
    """Copies attributes from object 'b' to object 'a', with options to include/exclude certain attributes."""
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        setattr(a, k, v)


def check_train_batch_size(model, imgsz=640, amp=True, batch=-1, max_num_obj=1):
    """Compute optimal YOLO training batch size using the autobatch() function.

    Args:
        model (torch.nn.Module): YOLO model to check batch size for.
        imgsz (int, optional): Image size used for training.
        amp (bool, optional): Use automatic mixed precision if True.
        batch (float, optional): Fraction of GPU memory to use. If -1, use default.
        max_num_obj (int, optional): The maximum number of objects from dataset.

    Returns:
        (int): Optimal batch size computed using the autobatch() function.

    Note:
        If 0.0 < batch < 1.0, it's used as the fraction of GPU memory to use.
        Otherwise, a default fraction of 0.6 is used.
    """
    with torch.autocast("cuda", enabled=amp):
        return autobatch(
            deepcopy(model).train(),
            imgsz,
            fraction=batch if 0.0 < batch < 1.0 else 0.6,
            max_num_obj=max_num_obj,
        )
