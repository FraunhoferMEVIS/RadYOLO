from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from yolo11_3d.models.layers import C2f, Detect

if TYPE_CHECKING:
    from yolo11_3d.models.model import HardcodedModel


def export_model(
    model: HardcodedModel,
    input_shape: tuple[int, ...],
    dest: Path | str,
    copy: bool = False,
) -> Path:
    inputs = torch.randn((1, *input_shape)).to("cpu")
    if copy:
        model = deepcopy(model)
    model.to("cpu")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()

    # TODO(kgeissler): Make export to float16 with input and output layers as float32

    for _ in range(2):
        model(inputs)

    for m in model.modules():
        if isinstance(m, Detect):
            m.export = True
        if isinstance(m, C2f):
            m.forward = m.forward_split

    onnx_path = Path(dest).with_suffix(".onnx")

    if model.enable_output_masks and not model.separate_objectness_head:
        output_names = ["boxes", "classScores", "maskCoefficients", "prototypeMasks"]
    elif not model.enable_output_masks and model.separate_objectness_head:
        output_names = ["boxes", "classScores", "objectnessScores"]
    elif model.enable_output_masks and model.separate_objectness_head:
        output_names = [
            "boxes",
            "classScores",
            "objectnessScores",
            "maskCoefficients",
            "prototypeMasks",
        ]
    else:
        output_names = ["boxes", "classScores"]

    torch.onnx.export(
        model,
        inputs,
        onnx_path,
        verbose=False,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=None,
    )
    return onnx_path
