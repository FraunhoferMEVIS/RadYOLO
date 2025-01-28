from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from nn.layers import C2PSA, C3, SPPF, Bottleneck, C2f, C3k2, Concat, Conv, Detect, DWConv

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


def export_model(model: nn.Module, name: str) -> Path:
    inputs = torch.randn((1, 3, 320, 320)).to("cpu")
    model.to("cpu")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()

    model(inputs)
    model(inputs)

    for m in model.modules():
        if isinstance(m, Detect):
            m.export = True
        if isinstance(m, C2f):
            m.forward = m.forward_split

    onnx_path = Path(name).with_suffix(".onnx")

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
