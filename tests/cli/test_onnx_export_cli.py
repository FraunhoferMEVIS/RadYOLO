from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx
import onnxruntime as ort
import torch
from click.testing import CliRunner

from radyolo import main
from radyolo.config.config import load_model_checkpoint
from radyolo.models.model import build_model_from_config

if TYPE_CHECKING:
    from pathlib import Path


def _build_deterministic_input(input_shape: tuple[int, ...]) -> torch.Tensor:
    total_values = int(np.prod((1, *input_shape)))
    return torch.linspace(0.0, 1.0, steps=total_values, dtype=torch.float32).reshape(
        1, *input_shape
    )


def _run_torch_and_onnx_inference(
    model_path: Path,
    onnx_path: Path,
    input_tensor: torch.Tensor,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    checkpoint = load_model_checkpoint(model_path)
    train_config = checkpoint["train_config"]
    model = build_model_from_config(
        train_config,
        state_dict=checkpoint["model_state_dict"],
    )
    model.eval()

    with torch.inference_mode():
        torch_outputs = [
            tensor.detach().cpu().numpy() for tensor in model(input_tensor).to_export_tensors()
        ]

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, {"images": input_tensor.cpu().numpy()})
    return torch_outputs, onnx_outputs


def _load_onnx_model(onnx_path: Path) -> onnx.ModelProto:
    return onnx.load(onnx_path)


def test_onnx_export_cli_defaults_to_float16_internal_ops(trained_run: Path) -> None:
    model_path = trained_run / "weights" / "last.pt"
    runner = CliRunner()

    result = runner.invoke(main, ["onnx-export", str(model_path)])

    assert result.exit_code == 0
    onnx_path = model_path.with_suffix(".onnx")
    assert onnx_path.exists(), "ONNX export not created"

    onnx_model = _load_onnx_model(onnx_path)

    assert onnx_model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert all(
        output.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        for output in onnx_model.graph.output
    )
    assert any(
        initializer.data_type == onnx.TensorProto.FLOAT16
        for initializer in onnx_model.graph.initializer
    )


def test_onnx_export_cli_allows_float32_internal_ops(trained_run: Path) -> None:
    model_path = trained_run / "weights" / "last.pt"
    runner = CliRunner()

    result = runner.invoke(main, ["onnx-export", str(model_path), "--internal-dtype", "float32"])

    assert result.exit_code == 0
    onnx_path = model_path.with_suffix(".onnx")
    assert onnx_path.exists(), "ONNX export not created"

    onnx_model = _load_onnx_model(onnx_path)
    assert all(
        initializer.data_type != onnx.TensorProto.FLOAT16
        for initializer in onnx_model.graph.initializer
    )

    checkpoint = load_model_checkpoint(model_path)
    train_config = checkpoint["train_config"]
    input_tensor = _build_deterministic_input(
        (train_config.input_channels, *train_config.patch_size.get_zyx())
    )

    torch_outputs, onnx_outputs = _run_torch_and_onnx_inference(model_path, onnx_path, input_tensor)

    assert len(torch_outputs) == len(onnx_outputs)
    for torch_output, onnx_output in zip(torch_outputs, onnx_outputs, strict=True):
        assert torch_output.shape == onnx_output.shape
        np.testing.assert_allclose(onnx_output, torch_output, rtol=1e-3, atol=1e-4)
