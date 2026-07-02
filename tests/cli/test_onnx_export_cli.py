from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import numpy as np
import onnx
import onnxruntime as ort
import torch
from click.testing import CliRunner

from radyolo import main
from radyolo.config.config import load_model_checkpoint
from radyolo.models.model import load_checkpoint_model

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
    loaded_model = load_checkpoint_model(model_path)
    model = loaded_model.model
    model.eval()

    has_tabular_inputs = (
        model.enable_tabular_image_label_fusion and len(model.tabular_feature_names) > 0
    )
    if has_tabular_inputs:
        num_features = len(model.tabular_feature_names)
        tabular_values = torch.zeros((1, num_features), dtype=torch.float32)
        tabular_mask = torch.zeros((1, num_features), dtype=torch.bool)
        with torch.inference_mode():
            torch_outputs = [
                tensor.detach().cpu().numpy()
                for tensor in model(
                    input_tensor,
                    tabular_values=tabular_values,
                    tabular_mask=tabular_mask,
                ).to_export_tensors()
            ]
    else:
        with torch.inference_mode():
            torch_outputs = [
                tensor.detach().cpu().numpy() for tensor in model(input_tensor).to_export_tensors()
            ]

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_inputs = {"images": input_tensor.cpu().numpy()}
    if has_tabular_inputs:
        onnx_inputs["tabularValues"] = tabular_values.cpu().numpy()
        onnx_inputs["tabularMask"] = tabular_mask.cpu().numpy()
    onnx_outputs = session.run(None, onnx_inputs)
    return torch_outputs, onnx_outputs


def _load_onnx_model(onnx_path: Path) -> onnx.ModelProto:
    return onnx.load(onnx_path)


def _metadata_dict(model: onnx.ModelProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in model.metadata_props}


def test_onnx_export_cli_defaults_to_float16_internal_ops(trained_run: Path) -> None:
    model_path = trained_run / "weights_last.pt"
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

    metadata = _metadata_dict(onnx_model)
    checkpoint = load_model_checkpoint(model_path)
    train_config = checkpoint["train_config"].model_dump(mode="json")
    assert metadata["training_mode"] == str(train_config["training_mode"])
    assert json.loads(metadata["patch_size"]) == train_config["patch_size"]
    assert metadata["voxel_size"] == "null"
    assert json.loads(metadata["normalization"]) == train_config["normalization"]
    assert metadata["data"] == str(train_config["data"])
    assert metadata["fold"] == str(train_config["fold"])
    assert metadata["split_file_name"] == str(train_config["split_file_name"])
    assert metadata["best_nms_iou_threshold"] == "0.1"


def test_onnx_export_cli_allows_float32_internal_ops(trained_run: Path) -> None:
    model_path = trained_run / "weights_last.pt"
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


def test_onnx_export_cli_uses_suggested_best_nms_threshold_when_available(
    trained_run: Path,
) -> None:
    model_path = trained_run / "weights_best.pt"
    csv_path = trained_run / "nms_optimization.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "checkpoint",
                "checkpoint_path",
                "nms_iou_threshold",
                "suggested",
                "fitness",
                "metrics/precision",
                "metrics/recall",
                "metrics/mAP10",
                "metrics/mAP10Agn",
                "metrics/mAP10-95",
                "metrics/mAP10-95Agn",
                "metrics/dice",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "checkpoint": "last",
                "checkpoint_path": "weights_last.pt",
                "nms_iou_threshold": 0.2,
                "suggested": True,
                "fitness": 0.4,
                "metrics/precision": 0.0,
                "metrics/recall": 0.0,
                "metrics/mAP10": 0.0,
                "metrics/mAP10Agn": 0.0,
                "metrics/mAP10-95": 0.0,
                "metrics/mAP10-95Agn": 0.0,
                "metrics/dice": 0.0,
            }
        )
        writer.writerow(
            {
                "checkpoint": "best",
                "checkpoint_path": "weights_best.pt",
                "nms_iou_threshold": 0.6,
                "suggested": True,
                "fitness": 0.9,
                "metrics/precision": 0.0,
                "metrics/recall": 0.0,
                "metrics/mAP10": 0.0,
                "metrics/mAP10Agn": 0.0,
                "metrics/mAP10-95": 0.0,
                "metrics/mAP10-95Agn": 0.0,
                "metrics/dice": 0.0,
            }
        )

    runner = CliRunner()
    result = runner.invoke(main, ["onnx-export", str(model_path), "--internal-dtype", "float32"])

    assert result.exit_code == 0
    onnx_path = model_path.with_suffix(".onnx")
    onnx_model = _load_onnx_model(onnx_path)
    metadata = _metadata_dict(onnx_model)
    assert metadata["best_nms_iou_threshold"] == "0.6"


def test_onnx_export_cli_allows_metadata_overwrite_options(trained_run: Path) -> None:
    model_path = trained_run / "weights_last.pt"
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "onnx-export",
            str(model_path),
            "--internal-dtype",
            "float32",
            "--metadata-overwrite",
            "normalization={type: ct, mean: 0, std: 1, clip_min: -1024, clip_max: 3071}",
            "--metadata-overwrite",
            "data=/tmp/preprocessed_dataset",
            "--metadata-overwrite",
            "best_nms_iou_threshold=0.33",
        ],
    )

    assert result.exit_code == 0
    onnx_model = _load_onnx_model(model_path.with_suffix(".onnx"))
    metadata = _metadata_dict(onnx_model)
    assert json.loads(metadata["normalization"]) == {
        "type": "ct",
        "mean": 0,
        "std": 1,
        "clip_min": -1024,
        "clip_max": 3071,
    }
    assert metadata["data"] == "/tmp/preprocessed_dataset"
    assert metadata["best_nms_iou_threshold"] == "0.33"
