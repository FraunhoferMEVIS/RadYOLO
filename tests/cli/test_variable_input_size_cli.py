from __future__ import annotations

import json
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import yaml
from click.testing import CliRunner

from radyolo import main
from radyolo.data.data import (
    WholeImageDataset,
    pad_batch_to_common_shape,
    preprocess_batch_images,
)
from radyolo.inference.predict import PredictionConfig, run_prediction
from radyolo.models.model import load_checkpoint_model

if TYPE_CHECKING:
    from pathlib import Path

# (x, y, z) shapes of the synthetic volumes; deliberately not multiples of 32 and all different
CASE_SHAPES_XYZ = {
    "case1": (32, 32, 64),
    "case2": (40, 24, 50),
    "case3": (64, 48, 36),
    "case4": (30, 45, 33),
    "case5": (48, 40, 70),
}


@pytest.fixture(scope="module")
def variable_size_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a dataset whose volumes all have different shapes."""
    root = tmp_path_factory.mktemp("variable_size_dataset")
    for sub in ("images", "labels", "masks"):
        (root / sub).mkdir()

    for case, shape in CASE_SHAPES_XYZ.items():
        volume = np.zeros(shape, dtype=np.float32)
        lower = [dim // 4 for dim in shape]
        upper = [dim * 3 // 4 for dim in shape]
        volume[lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]] = 1.0
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(volume, affine), str(root / "images" / f"{case}.nii.gz"))
        nib.save(
            nib.Nifti1Image((volume > 0).astype(np.uint8), affine),
            str(root / "masks" / f"{case}.nii.gz"),
        )
        # normalized xyzwhd box around the bright cuboid
        center = [(lo + hi) / 2 / dim for lo, hi, dim in zip(lower, upper, shape, strict=True)]
        extent = [(hi - lo) / dim for lo, hi, dim in zip(lower, upper, shape, strict=True)]
        (root / "labels" / f"{case}.txt").write_text(
            "0 " + " ".join(f"{value:.6f}" for value in (*center, *extent)) + "\n",
            encoding="utf-8",
        )

    cases = list(CASE_SHAPES_XYZ)
    (root / "dataset.yaml").write_text("names: {0: lesion}\n", encoding="utf-8")
    splits = [{"train": cases[:3], "val": cases[3:4], "test": cases[4:]}]
    (root / "data_splits.json").write_text(json.dumps(splits), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def variable_size_run(
    variable_size_dataset: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Train a tiny model with variable input sizes and return the run directory."""
    root = tmp_path_factory.mktemp("variable_size_run")
    config = {
        "input_channels": 1,
        "model_scale": "n",
        "data": variable_size_dataset.as_posix(),
        "fold": 0,
        "epochs": 1,
        "nominal_batch_size": 2,
        "warmup_epochs": 0.0,
        "patience": 1,
        "batch_size": 2,
        "training_mode": "wholeimage",
        "variable_input_size": True,
        "patch_size": {"x": 64, "y": 64, "z": 64},
        "normalization": {"type": "none"},
        "device": "cpu",
        "workers": 0,
        "output_folder_base": (root / "runs").as_posix(),
        "experiment_name": "variable",
        "exist_ok": True,
        "optimizer": "Adam",
        "seed": 0,
        "resume": False,
        "amp": False,
        "use_cudnn_benchmark": False,
        "val": True,
        "plots": False,
        "nms_iou_threshold_candidates": [0.2],
    }
    config_path = root / "train.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(main, ["train", str(config_path)])

    assert result.exit_code == 0, result.output
    run_dir = root / "runs" / "variable"
    assert (run_dir / "weights_last.pt").exists()
    return run_dir


def test_training_with_variable_input_sizes_completes(variable_size_run: Path) -> None:
    loaded = load_checkpoint_model(variable_size_run / "weights_last.pt")

    assert loaded.train_config.variable_input_size is True
    assert (variable_size_run / "results.csv").exists()


def test_batches_with_different_shapes_are_padded(
    variable_size_dataset: Path,
    variable_size_run: Path,
) -> None:
    config = load_checkpoint_model(variable_size_run / "weights_last.pt").train_config
    dataset = WholeImageDataset(
        src_path=variable_size_dataset,
        patch_size=config.patch_size,
        group="train",
        normalization=config.normalization,
        config=config,
    )

    batch = WholeImageDataset.collate_fn([dataset[0], dataset[1], dataset[2]])
    assert isinstance(batch["img"], tuple)
    assert [tuple(image.shape[1:]) for image in batch["img"]] == [
        (64, 32, 32),
        (50, 24, 40),
        (36, 48, 64),
    ]
    # masks follow the image shape of their own sample
    assert [tuple(masks.shape[1:]) for masks in batch["masks"]] == [
        (64, 32, 32),
        (50, 24, 40),
        (36, 48, 64),
    ]

    preprocess_batch_images(
        batch,
        device=torch.device("cpu"),
        normalization=config.normalization,
        target_shape=config.patch_size,
        apply_preprocessing=config.apply_preprocessing,
        preprocessing_device="cpu",
        variable_input_size=True,
    )
    pad_batch_to_common_shape(batch)

    assert batch["img"].shape == (3, 1, 64, 64, 64)
    assert all(masks.shape[1:] == (64, 64, 64) for masks in batch["masks"])
    # labels stay valid in voxels: the box of case2 is centred in its own (z=50, y=24, x=40) volume
    sample_boxes = batch["bboxes"][batch["batch_idx"] == 1][0]
    scale_xyz = torch.tensor([64.0, 64.0, 64.0])
    center_xyz = sample_boxes[:3] * scale_xyz
    assert torch.allclose(center_xyz, torch.tensor([20.0, 12.0, 25.0]), atol=1.0)


def test_onnx_export_is_dynamic_by_default(variable_size_run: Path) -> None:
    model_path = variable_size_run / "weights_last.pt"

    result = CliRunner().invoke(
        main, ["onnx-export", str(model_path), "--internal-dtype", "float32"]
    )

    assert result.exit_code == 0, result.output
    onnx_path = model_path.with_suffix(".onnx")
    onnx_model = onnx.load(onnx_path)
    metadata = {entry.key: entry.value for entry in onnx_model.metadata_props}
    assert metadata["dynamic_input_size"] == "True"
    assert metadata["size_divisor"] == "32"
    input_dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    assert [dim.HasField("dim_value") for dim in input_dims] == [False, True, False, False, False]

    model = load_checkpoint_model(model_path).model
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    generator = torch.Generator().manual_seed(0)
    for shape in [(1, 1, 32, 32, 32), (2, 1, 64, 32, 96), (1, 1, 96, 64, 32)]:
        image = torch.randn(shape, generator=generator)
        with torch.no_grad():
            expected = [t.numpy() for t in model(image).to_export_tensors()]
        actual = session.run(None, {"images": image.numpy()})
        assert len(actual) == len(expected)
        for expected_output, actual_output in zip(expected, actual, strict=True):
            assert actual_output.shape == expected_output.shape
            np.testing.assert_allclose(actual_output, expected_output, rtol=1e-3, atol=1e-3)


def test_float16_onnx_export_runs_with_dynamic_input_sizes(
    variable_size_run: Path, tmp_path: Path
) -> None:
    model_path = tmp_path / "weights_float16.pt"
    model_path.write_bytes((variable_size_run / "weights_last.pt").read_bytes())

    result = CliRunner().invoke(main, ["onnx-export", str(model_path)])

    assert result.exit_code == 0, result.output
    onnx_path = model_path.with_suffix(".onnx")
    assert any(
        initializer.data_type == onnx.TensorProto.FLOAT16
        for initializer in onnx.load(onnx_path).graph.initializer
    )

    model = load_checkpoint_model(model_path).model
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    generator = torch.Generator().manual_seed(1)
    for shape in [(1, 1, 32, 32, 64), (2, 1, 64, 96, 32)]:
        image = torch.randn(shape, generator=generator)
        with torch.no_grad():
            expected = [t.numpy() for t in model(image).to_export_tensors()]
        actual = session.run(None, {"images": image.numpy()})
        for expected_output, actual_output in zip(expected, actual, strict=True):
            assert actual_output.shape == expected_output.shape
        # boxes are in voxel units, so allow for float16 rounding of the coordinates
        np.testing.assert_allclose(actual[0], expected[0], atol=1.0)
        np.testing.assert_allclose(actual[1], expected[1], atol=5e-2)


def test_onnx_export_fixed_input_size_option(variable_size_run: Path, tmp_path: Path) -> None:
    fixed_model_path = tmp_path / "weights_fixed.pt"
    fixed_model_path.write_bytes((variable_size_run / "weights_last.pt").read_bytes())

    result = CliRunner().invoke(
        main,
        ["onnx-export", str(fixed_model_path), "--internal-dtype", "float32", "--fixed-input-size"],
    )

    assert result.exit_code == 0, result.output
    onnx_model = onnx.load(fixed_model_path.with_suffix(".onnx"))
    metadata = {entry.key: entry.value for entry in onnx_model.metadata_props}
    assert metadata["dynamic_input_size"] == "False"
    input_dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    assert [dim.dim_value for dim in input_dims] == [1, 1, 64, 64, 64]


def test_whole_image_prediction_handles_arbitrary_image_shapes(
    variable_size_dataset: Path,
    variable_size_run: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "predictions"
    output_dir.mkdir()

    result = run_prediction(
        PredictionConfig(
            model_path=variable_size_run / "weights_last.pt",
            input_dir=variable_size_dataset / "images",
            output_dir=output_dir,
            confidence_threshold=0.0,
            nms_iou_threshold=0.2,
            device="cpu",
            clip_boxes_to_image=True,
            num_load_workers=1,
            num_write_workers=1,
            split_file=variable_size_dataset / "data_splits.json",
            split_group="train",
        )
    )

    assert result == 0
    for case in ("case1", "case2", "case3"):
        shape_xyz = CASE_SHAPES_XYZ[case]
        predictions = json.loads((output_dir / f"{case}.json").read_text(encoding="utf-8"))
        assert predictions, f"expected detections for {case} at confidence threshold 0"
        for prediction in predictions:
            # boxes are clipped to the original (unpadded) image
            x1, y1, z1, x2, y2, z2 = prediction["box_image_xyzxyz"]
            assert 0.0 <= x1 <= x2 <= shape_xyz[0]
            assert 0.0 <= y1 <= y2 <= shape_xyz[1]
            assert 0.0 <= z1 <= z2 <= shape_xyz[2]


@pytest.mark.parametrize("variable_input_size", [False, True])
def test_training_with_downsize_augmentation(
    variable_size_dataset: Path,
    tmp_path: Path,
    variable_input_size: bool,
) -> None:
    config = {
        "input_channels": 1,
        "model_scale": "n",
        "data": variable_size_dataset.as_posix(),
        "fold": 0,
        "epochs": 2,
        "nominal_batch_size": 2,
        "warmup_epochs": 0.0,
        "patience": 5,
        "batch_size": 2,
        "training_mode": "wholeimage",
        "variable_input_size": variable_input_size,
        "patch_size": {"x": 96, "y": 96, "z": 96},
        "prob_downsize": 1.0,
        "downsize_min_patch_size": {"x": 32, "y": 32, "z": 64},
        "normalization": {"type": "none"},
        "device": "cpu",
        "workers": 0,
        "output_folder_base": (tmp_path / "runs").as_posix(),
        "experiment_name": "downsize",
        "optimizer": "Adam",
        "resume": False,
        "amp": False,
        "use_cudnn_benchmark": False,
        "plots": False,
        "nms_iou_threshold_candidates": [0.2],
    }
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(main, ["train", str(config_path)])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "runs" / "downsize" / "weights_last.pt").exists()
