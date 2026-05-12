from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import onnx
import torch
from onnxconverter_common import float16
from torch import nn

from radyolo.models.layers import C2f, Detect

if TYPE_CHECKING:
    from radyolo.models.model import HardcodedModel


ExportInternalDType = Literal["float16", "float32"]


class ExportModelWrapper(nn.Module):
    def __init__(self, model: HardcodedModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.model(x).to_export_tensors()


def export_model(
    model: HardcodedModel,
    input_shape: tuple[int, ...],
    dest: Path | str,
    internal_dtype: ExportInternalDType = "float16",
) -> Path:
    if internal_dtype not in {"float16", "float32"}:
        msg = f"Unsupported internal ONNX dtype: {internal_dtype}"
        raise ValueError(msg)

    inputs = torch.zeros((1, *input_shape)).to("cpu")
    model.to("cpu")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()

    for _ in range(2):
        model(inputs)

    for m in model.modules():
        if isinstance(m, Detect):
            m.export = True
        if isinstance(m, C2f):
            m.forward = m.forward_split

    onnx_path = Path(dest).with_suffix(".onnx")
    export_wrapper = ExportModelWrapper(model)
    export_wrapper.eval()

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
        export_wrapper,
        inputs,
        onnx_path,
        verbose=False,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=None,
    )

    if internal_dtype == "float16":
        onnx_model = onnx.load(onnx_path)
        onnx_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
        onnx.save(onnx_model, onnx_path)

    return onnx_path
