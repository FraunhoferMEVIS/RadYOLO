import torch
from copy import deepcopy
from pathlib import Path

from yolo11_3d.nn.model import HardcodedModel
from yolo11_3d.nn.layers import (
    C2f,
    Detect
)

def export_model(
    model: HardcodedModel, input_shape: tuple[int, ...], dest: Path | str, copy: bool = False
) -> Path:
    inputs = torch.randn((1, *input_shape)).to("cpu")
    if copy:
        model = deepcopy(model)
    model.to("cpu")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()

    # TODO: Make export to float16 with input and output layers as float32

    for _ in range(2):
        model(inputs)

    for m in model.modules():
        if isinstance(m, Detect):
            m.export = True
        if isinstance(m, C2f):
            m.forward = m.forward_split

    onnx_path = Path(dest).with_suffix(".onnx")

    if model.enable_output_masks and not model.separate_objectness_head:
        output_names = ["boxes", "class_scores", "maskCoefficients", "prototypeMasks"]
    elif not model.enable_output_masks and model.separate_objectness_head:
        output_names = ["boxes", "class_scores", "objectness_scores"]
    elif model.enable_output_masks and model.separate_objectness_head:
        output_names = ["boxes", "class_scores", "objectness_scores", "maskCoefficients", "prototypeMasks"]
    else:
        output_names = ["boxes", "class_scores"]

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