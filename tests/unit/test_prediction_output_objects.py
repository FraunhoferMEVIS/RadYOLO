from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass

import nibabel as nib
import numpy as np
import pytest
import torch

from radyolo.inference import predict as predict_module
from radyolo.inference.predict import (
    PredictionConfig,
    LoadedImageTask,
    _apply_patch_border_weight,
    _get_patch_starts,
    _merge_patch_detections,
    _predict_loaded_image,
    _predict_patchwise_image,
    reconstruct_and_save_mask,
    _relative_box,
    _write_prediction_result,
)
from radyolo.config.spatial import PatchSize, VoxelSize
from radyolo.data.image_labels import ImageLabelTaskSpec
from radyolo.models.layers import ModelOutput
from radyolo.utils.ops import (
    box_image_to_world,
    non_max_suppression,
    prepare_model_output_for_nms,
    scale_boxes,
)


def test_model_output_converts_to_prediction_components() -> None:
    output = ModelOutput(
        decoded_boxes=torch.ones((1, 6, 2), dtype=torch.float32),
        class_probabilities=torch.full((1, 2, 2), 0.5, dtype=torch.float32),
        objectness_probabilities=torch.full((1, 1, 2), 0.25, dtype=torch.float32),
    )

    components = output.to_prediction_components()

    assert len(components) == 3
    assert torch.equal(components[0], output.decoded_boxes)
    assert torch.equal(components[1], output.class_probabilities)
    assert torch.equal(components[2], output.objectness_probabilities)


def test_prepare_prediction_for_nms_accepts_model_output() -> None:
    model_output = ModelOutput(
        decoded_boxes=torch.arange(12, dtype=torch.float32).reshape(1, 6, 2),
        class_probabilities=torch.tensor([[[0.4, 0.8], [0.5, 0.25]]], dtype=torch.float32),
        objectness_probabilities=torch.tensor([[[0.5, 0.25]]], dtype=torch.float32),
        mask_coefficients=torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
    )

    prepared = prepare_model_output_for_nms(model_output)

    if model_output.class_probabilities is None or model_output.objectness_probabilities is None:
        raise AssertionError
    expected_scores = model_output.class_probabilities * model_output.objectness_probabilities
    if model_output.decoded_boxes is None or model_output.mask_coefficients is None:
        raise AssertionError
    expected = torch.cat(
        (model_output.decoded_boxes, expected_scores, model_output.mask_coefficients), dim=1
    )
    assert torch.equal(prepared, expected)


def test_model_output_export_tensors_include_masks_when_available() -> None:
    model_output = ModelOutput(
        decoded_boxes=torch.ones((1, 6, 2), dtype=torch.float32),
        class_probabilities=torch.ones((1, 2, 2), dtype=torch.float32),
        mask_coefficients=torch.ones((1, 4, 2), dtype=torch.float32),
        prototype_masks=torch.ones((1, 4, 3, 3, 3), dtype=torch.float32),
    )

    export_tensors = model_output.to_export_tensors()

    assert len(export_tensors) == 4
    assert torch.equal(export_tensors[2], model_output.mask_coefficients)
    assert torch.equal(export_tensors[3], model_output.prototype_masks)


def test_non_max_suppression_accepts_model_output(monkeypatch) -> None:
    model_output = ModelOutput(
        decoded_boxes=torch.arange(12, dtype=torch.float32).reshape(1, 6, 2),
        class_probabilities=torch.tensor([[[0.6, 0.2], [0.3, 0.9]]], dtype=torch.float32),
        mask_coefficients=torch.tensor([[[0.1, 0.2]]], dtype=torch.float32),
    )
    captured: dict[str, object] = {}

    def fake_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float,
        max_detections: int,
    ) -> torch.Tensor:
        captured["boxes"] = boxes
        captured["scores"] = scores
        captured["iou_threshold"] = iou_threshold
        captured["max_detections"] = max_detections
        return torch.tensor([0], dtype=torch.int64)

    monkeypatch.setattr("radyolo.utils.ops.nms", fake_nms)

    result = non_max_suppression(
        model_output,
        conf_thres=0.1,
        iou_thres=0.2,
        nc=2,
    )

    assert len(result) == 1
    assert result[0].shape[1] == 9
    assert captured["iou_threshold"] == 0.2
    assert captured["max_detections"] == 300


def test_non_max_suppression_keeps_high_class_indices_separate() -> None:
    model_output = ModelOutput(
        decoded_boxes=torch.tensor(
            [
                [
                    [5.0, 5.0],
                    [5.0, 5.0],
                    [5.0, 5.0],
                    [2.0, 2.0],
                    [2.0, 2.0],
                    [2.0, 2.0],
                ]
            ],
            dtype=torch.float32,
        ),
        class_probabilities=torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.95, 0.0],
                    [0.0, 0.96],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        ),
    )

    result = non_max_suppression(
        model_output,
        conf_thres=0.1,
        iou_thres=0.2,
        nc=15,
    )

    assert len(result) == 1
    assert result[0].shape[0] == 2
    assert set(result[0][:, 7].tolist()) == {9.0, 10.0}


def test_scale_boxes_can_skip_clipping() -> None:
    boxes = torch.tensor([[-2.0, -3.0, -4.0, 12.0, 13.0, 14.0]], dtype=torch.float32)

    unclipped = scale_boxes((10, 10, 10), boxes.clone(), (10, 10, 10), clip=False)
    clipped = scale_boxes((10, 10, 10), boxes.clone(), (10, 10, 10), clip=True)

    assert torch.equal(unclipped, boxes)
    assert torch.equal(clipped, torch.tensor([[0.0, 0.0, 0.0, 10.0, 10.0, 10.0]]))


def test_scale_boxes_defaults_to_no_clipping() -> None:
    boxes = torch.tensor([[-2.0, -3.0, -4.0, 12.0, 13.0, 14.0]], dtype=torch.float32)

    scaled = scale_boxes((10, 10, 10), boxes.clone(), (10, 10, 10))

    assert torch.equal(scaled, boxes)


def test_relative_box_normalizes_against_original_shape() -> None:
    box = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=torch.float32)

    relative = _relative_box(box, (20, 40, 80))

    assert relative == [0.025, 0.1, 0.3, 0.1, 0.25, 0.6]


def test_relative_box_uses_xyx_denominators_for_anisotropic_shape() -> None:
    box = torch.tensor([200.0, 30.0, 100.0, 250.0, 50.0, 200.0], dtype=torch.float32)

    relative = _relative_box(box, (320, 60, 320))

    assert relative == [0.625, 0.5, 0.3125, 0.78125, 0.833333, 0.625]


def test_box_image_to_world_applies_affine_to_all_box_corners() -> None:
    box = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float32)
    affine = torch.tensor(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, 20.0],
            [0.0, 0.0, 4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    world_box = box_image_to_world(box, affine)

    expected = torch.tensor([12.0, 26.0, 42.0, 18.0, 35.0, 54.0], dtype=torch.float32)
    assert torch.equal(world_box, expected)


def test_reconstruct_and_save_mask_writes_thresholded_nifti_mask(tmp_path, monkeypatch) -> None:
    det = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.9, 0.0, 0.25, 0.75], dtype=torch.float32)
    proto = torch.zeros((2, 2, 2), dtype=torch.float32)
    nifti_img = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), affine=np.eye(4))
    expected_mask = torch.tensor(
        [
            [[0.2, 0.7], [0.8, 0.1]],
            [[0.6, 0.4], [0.3, 0.9]],
        ],
        dtype=torch.float32,
    )

    monkeypatch.setattr(
        predict_module.ops,
        "reconstruct_mask_probabilities",
        lambda coeff, proto_masks: expected_mask,
    )

    mask_filename = reconstruct_and_save_mask(
        det,
        proto,
        original_shape=(2, 2, 2),
        mask_affine=nifti_img.affine,
        mask_dir=tmp_path / "masks",
        det_index=3,
        case_name="caseA",
    )

    assert mask_filename == "caseA_det3.nii.gz"
    saved_mask_path = tmp_path / "masks" / mask_filename
    assert saved_mask_path.exists()

    saved_mask = nib.load(str(saved_mask_path))
    saved_data = np.asanyarray(saved_mask.dataobj)
    assert np.array_equal(saved_mask.affine, np.eye(4))
    assert set(np.unique(saved_data)).issubset({0, 1})
    assert int(saved_data.sum()) == 4


@dataclass
class _DummyTrainConfig:
    normalization: dict[str, str] = None  # type: ignore[assignment]
    patch_size: PatchSize = None  # type: ignore[assignment]
    voxel_size: VoxelSize | None = None
    num_classes: int = 1
    agnostic_nms: bool = False
    max_det: int = 300
    training_mode: str = "wholeimage"

    def __post_init__(self) -> None:
        if self.normalization is None:
            self.normalization = {"type": "none"}
        if self.patch_size is None:
            self.patch_size = PatchSize(x=10, y=11, z=12)


class _DummyModel:
    device = torch.device("cpu")

    def __call__(self, image: torch.Tensor) -> ModelOutput:
        return ModelOutput(
            decoded_boxes=torch.zeros((1, 6, 0), dtype=torch.float32),
            class_probabilities=torch.zeros((1, 1, 0), dtype=torch.float32),
        )


def test_predict_loaded_image_uses_prediction_overrides(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    image_np = np.zeros((4, 5, 6), dtype=np.float32)
    loaded_image = LoadedImageTask(
        case_name="case1",
        affine=np.eye(4, dtype=np.float32),
        image=image_np,
    )

    def fake_preprocess_image(
        image: torch.Tensor,
        normalization,
        *,
        target_shape=None,
        input_voxel_size=None,
        target_voxel_size=None,
        apply_resampling: bool,
        apply_normalization: bool,
        output_dtype: torch.dtype,
    ):
        captured["normalization"] = normalization
        captured["target_shape"] = target_shape
        captured["input_voxel_size"] = input_voxel_size
        captured["target_voxel_size"] = target_voxel_size
        captured["apply_resampling"] = apply_resampling
        captured["apply_normalization"] = apply_normalization
        captured["output_dtype"] = output_dtype
        return image.to(dtype=output_dtype), (6, 5, 4), (6, 5, 4)

    monkeypatch.setattr(predict_module, "preprocess_image", fake_preprocess_image)
    monkeypatch.setattr(
        predict_module,
        "_postprocess_single",
        lambda model_output, train_config, prediction_config, num_classes: (
            torch.zeros((0, 8), dtype=torch.float32),
            None,
        ),
    )

    write_task = _predict_loaded_image(
        loaded_image,
        model=_DummyModel(),
        train_config=_DummyTrainConfig(),
        names={0: "class_0"},
        prediction_config=PredictionConfig(
            model_path=tmp_path / "model.pt",
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            confidence_threshold=0.1,
            nms_iou_threshold=0.2,
            device="cpu",
            clip_boxes_to_image=False,
            overwrite_patch_size=PatchSize(x=20, y=21, z=22),
            overwrite_normalization={"type": "zscore"},
        ),
    )

    assert captured["normalization"] == {"type": "zscore"}
    assert captured["target_shape"] == PatchSize(x=20, y=21, z=22)
    assert captured["input_voxel_size"] is None
    assert captured["target_voxel_size"] is None
    assert captured["apply_resampling"] is True
    assert captured["apply_normalization"] is True
    assert captured["output_dtype"] == torch.float32
    assert write_task.image_results == []
    assert write_task.case_name == "case1"
    assert write_task.prototype_masks is None
    assert write_task.mask_detections == []


def test_write_prediction_result_saves_json_and_masks(tmp_path, monkeypatch) -> None:
    proto = torch.ones((2, 2, 2), dtype=torch.float32)
    write_task = predict_module.WriteTask(
        case_name="case2",
        image_results=[{"class": 0, "mask_path": "case2_det0.nii.gz"}],
        affine=np.eye(4, dtype=np.float32),
        original_shape=(2, 2, 2),
        prototype_masks=proto,
        mask_detections=[
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.9, 0.0, 0.25, 0.75], dtype=torch.float32)
        ],
    )
    captured: dict[str, object] = {}

    def fake_reconstruct_and_save_mask(
        det: torch.Tensor,
        proto_tensor: torch.Tensor,
        original_shape: tuple[int, int, int],
        mask_affine: np.ndarray,
        mask_dir,
        det_index: int,
        case_name: str,
    ) -> str | None:
        captured["det"] = det
        captured["proto_shape"] = tuple(proto_tensor.shape)
        captured["original_shape"] = original_shape
        captured["mask_affine"] = mask_affine
        captured["mask_dir"] = mask_dir
        captured["det_index"] = det_index
        captured["case_name"] = case_name
        mask_dir.mkdir(exist_ok=True)
        output_path = mask_dir / f"{case_name}_det{det_index}.nii.gz"
        output_path.write_bytes(b"mask")
        return output_path.name

    monkeypatch.setattr(
        predict_module,
        "reconstruct_and_save_mask",
        fake_reconstruct_and_save_mask,
    )

    _write_prediction_result(tmp_path, write_task)

    assert (tmp_path / "case2.json").exists()
    assert (tmp_path / "masks" / "case2_det0.nii.gz").exists()
    assert tuple(captured["proto_shape"]) == (2, 2, 2)
    assert captured["original_shape"] == (2, 2, 2)


def test_write_prediction_result_saves_global_image_outputs(tmp_path) -> None:
    write_task = predict_module.WriteTask(
        case_name="case_global",
        image_results=[],
        image_label_results={
            "tumor_grade": {
                "predicted_class": 1,
                "predicted_class_name": "high",
                "probabilities": [0.2, 0.8],
                "logits": [0.1, 2.1],
            }
        },
        affine=np.eye(4, dtype=np.float32),
        original_shape=(2, 2, 2),
        prototype_masks=None,
        mask_detections=[],
    )

    _write_prediction_result(tmp_path, write_task)

    labels_image_path = tmp_path / "labels_image" / "case_global.json"
    assert labels_image_path.exists()
    payload = json.loads(labels_image_path.read_text(encoding="utf-8"))
    assert payload["tumor_grade"]["predicted_class"] == 1


def test_load_case_tabular_inputs_reads_optional_tabular_json(tmp_path) -> None:
    tabular_dir = tmp_path / "inputs_tabular"
    tabular_dir.mkdir()
    (tabular_dir / "case_a.json").write_text(
        json.dumps({"age": 61, "weight": None}),
        encoding="utf-8",
    )

    values, observed_mask = predict_module._load_case_tabular_inputs(
        tabular_dir,
        "case_a",
        ("age", "weight"),
    )

    assert values is not None and observed_mask is not None
    assert values.shape == (1, 2)
    assert observed_mask.shape == (1, 2)
    assert values[0, 0].item() == pytest.approx(61.0)
    assert observed_mask[0, 0].item() is True
    assert observed_mask[0, 1].item() is False


def test_serialize_image_label_outputs_adds_risk_and_scalar_bin_width_for_rfs() -> None:
    serialized = predict_module._serialize_image_label_outputs(
        {
            "rfs": torch.tensor([[0.0, 1.0, -0.5]], dtype=torch.float32),
        },
        {
            "rfs": ImageLabelTaskSpec(
                type="survival",
                survival_bin_thresholds=(0.0, 2.0, 5.0),
            )
        },
    )

    assert serialized is not None
    rfs_payload = serialized["rfs"]
    assert isinstance(rfs_payload, dict)
    assert "risk" in rfs_payload
    assert "bin_width" in rfs_payload
    assert rfs_payload["bin_width"] == 2.0


def test_predict_loaded_image_skips_masks_when_disabled(monkeypatch, tmp_path) -> None:
    loaded_image = LoadedImageTask(
        case_name="case_no_masks",
        affine=np.eye(4, dtype=np.float32),
        image=np.zeros((4, 5, 6), dtype=np.float32),
    )

    class _DummySegmentationModel:
        device = torch.device("cpu")

        def __call__(self, image: torch.Tensor) -> ModelOutput:
            del image
            return ModelOutput(
                decoded_boxes=torch.zeros((1, 6, 1), dtype=torch.float32),
                class_probabilities=torch.ones((1, 1, 1), dtype=torch.float32),
                mask_coefficients=torch.ones((1, 2, 1), dtype=torch.float32),
                prototype_masks=torch.ones((1, 2, 2, 2, 2), dtype=torch.float32),
            )

    monkeypatch.setattr(
        predict_module,
        "_postprocess_single",
        lambda model_output, train_config, prediction_config, num_classes: (
            torch.tensor(
                [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.95, 0.0, 0.5, 0.25]],
                dtype=torch.float32,
            ),
            torch.ones((1, 2, 2, 2, 2), dtype=torch.float32),
        ),
    )

    write_task = _predict_loaded_image(
        loaded_image,
        model=_DummySegmentationModel(),
        train_config=_DummyTrainConfig(),
        names={0: "class_0"},
        prediction_config=PredictionConfig(
            model_path=tmp_path / "model.pt",
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            confidence_threshold=0.1,
            nms_iou_threshold=0.2,
            device="cpu",
            clip_boxes_to_image=False,
            save_masks=False,
        ),
    )

    assert write_task.image_results
    assert "mask_path" not in write_task.image_results[0]
    assert write_task.prototype_masks is None
    assert write_task.mask_detections == []


def test_predict_patchwise_image_skips_mask_reconstruction_when_disabled(
    monkeypatch,
    tmp_path,
) -> None:
    loaded_image = LoadedImageTask(
        case_name="case_patch_no_masks",
        affine=np.eye(4, dtype=np.float32),
        image=np.zeros((4, 5, 6), dtype=np.float32),
    )

    class _DummySegmentationModel:
        device = torch.device("cpu")

        def __call__(self, image: torch.Tensor) -> ModelOutput:
            del image
            return ModelOutput(
                decoded_boxes=torch.zeros((1, 6, 1), dtype=torch.float32),
                class_probabilities=torch.ones((1, 1, 1), dtype=torch.float32),
                mask_coefficients=torch.ones((1, 2, 1), dtype=torch.float32),
                prototype_masks=torch.ones((1, 2, 2, 2, 2), dtype=torch.float32),
            )

    monkeypatch.setattr(
        predict_module, "_iter_patch_origins", lambda *_args, **_kwargs: [(0, 0, 0)]
    )
    monkeypatch.setattr(
        predict_module,
        "_postprocess_single",
        lambda model_output, train_config, prediction_config, num_classes: (
            torch.tensor(
                [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.9, 0.0, 0.2, 0.3]],
                dtype=torch.float32,
            ),
            torch.ones((1, 2, 2, 2, 2), dtype=torch.float32),
        ),
    )

    def fail_reconstruct(*args, **kwargs):
        del args, kwargs
        raise AssertionError("mask reconstruction should not run when save_masks is disabled")

    monkeypatch.setattr(predict_module, "_reconstruct_patch_mask_to_image", fail_reconstruct)

    write_task = _predict_patchwise_image(
        loaded_image,
        model=_DummySegmentationModel(),
        train_config=_DummyTrainConfig(training_mode="patchwise"),
        names={0: "class_0"},
        prediction_config=PredictionConfig(
            model_path=tmp_path / "model.pt",
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            confidence_threshold=0.1,
            nms_iou_threshold=0.2,
            device="cpu",
            clip_boxes_to_image=False,
            save_masks=False,
        ),
    )

    assert write_task.image_results
    assert "mask_path" not in write_task.image_results[0]
    assert write_task.merged_mask_probabilities == []


def test_predict_patchwise_image_reconstructs_masks_only_for_kept_detections(
    monkeypatch,
    tmp_path,
) -> None:
    loaded_image = LoadedImageTask(
        case_name="case_patch_kept_masks",
        affine=np.eye(4, dtype=np.float32),
        image=np.zeros((4, 5, 6), dtype=np.float32),
    )

    class _DummySegmentationModel:
        device = torch.device("cpu")

        def __call__(self, image: torch.Tensor) -> ModelOutput:
            del image
            return ModelOutput(
                decoded_boxes=torch.zeros((1, 6, 2), dtype=torch.float32),
                class_probabilities=torch.ones((1, 1, 2), dtype=torch.float32),
                mask_coefficients=torch.ones((1, 2, 2), dtype=torch.float32),
                prototype_masks=torch.ones((1, 2, 2, 2, 2), dtype=torch.float32),
            )

    monkeypatch.setattr(
        predict_module, "_iter_patch_origins", lambda *_args, **_kwargs: [(0, 0, 0)]
    )
    monkeypatch.setattr(
        predict_module,
        "_postprocess_single",
        lambda model_output, train_config, prediction_config, num_classes: (
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.8, 0.0, 0.2, 0.3],
                    [0.0, 0.0, 0.0, 1.2, 1.2, 1.2, 0.7, 0.0, 0.4, 0.5],
                ],
                dtype=torch.float32,
            ),
            torch.ones((1, 2, 2, 2, 2), dtype=torch.float32),
        ),
    )

    def fake_merge_patch_detections(
        detections,
        *,
        iou_threshold: float,
        agnostic_nms: bool,
        max_det: int,
    ):
        del iou_threshold, agnostic_nms, max_det
        return torch.stack([detections[1]], dim=0), [1]

    reconstructed: list[float] = []

    def fake_reconstruct_patch_mask_to_image(
        detection: torch.Tensor,
        prototype_masks: torch.Tensor,
        patch_shape_zyx: tuple[int, int, int],
        image_shape_zyx: tuple[int, int, int],
        patch_origin_xyz: tuple[int, int, int],
    ) -> torch.Tensor:
        del prototype_masks, patch_shape_zyx, image_shape_zyx, patch_origin_xyz
        reconstructed.append(float(detection[6].item()))
        return torch.ones((6, 5, 4), dtype=torch.float32)

    monkeypatch.setattr(predict_module, "_merge_patch_detections", fake_merge_patch_detections)
    monkeypatch.setattr(
        predict_module,
        "_reconstruct_patch_mask_to_image",
        fake_reconstruct_patch_mask_to_image,
    )

    write_task = _predict_patchwise_image(
        loaded_image,
        model=_DummySegmentationModel(),
        train_config=_DummyTrainConfig(training_mode="patchwise"),
        names={0: "class_0"},
        prediction_config=PredictionConfig(
            model_path=tmp_path / "model.pt",
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            confidence_threshold=0.1,
            nms_iou_threshold=0.2,
            device="cpu",
            clip_boxes_to_image=False,
            save_masks=True,
        ),
    )

    assert reconstructed == pytest.approx([0.7])
    assert len(write_task.merged_mask_probabilities) == 1
    assert write_task.image_results
    assert "mask_path" in write_task.image_results[0]


def test_get_patch_starts_covers_image_with_requested_overlap() -> None:
    assert _get_patch_starts(10, 4, 0.5) == [0, 2, 4, 6]
    assert _get_patch_starts(3, 4, 0.5) == [0]


def test_get_patch_starts_evenly_spreads_uneven_tail() -> None:
    assert _get_patch_starts(11, 4, 0.5) == [0, 2, 4, 5, 7]


def test_get_patch_starts_allows_larger_overlap_to_keep_even_spacing() -> None:
    assert _get_patch_starts(11, 4, 0.25) == [0, 2, 5, 7]


def test_patch_border_weight_downweights_border_predictions() -> None:
    detection = torch.tensor([0.0, 2.0, 2.0, 2.0, 4.0, 4.0, 1.0, 0.0], dtype=torch.float32)

    weighted = _apply_patch_border_weight(detection, (4, 4, 4), 0.8)

    assert weighted[6].item() == pytest.approx(0.8)


def test_merge_patch_detections_applies_image_level_nms() -> None:
    detections = [
        torch.tensor([0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 0.9, 0.0], dtype=torch.float32),
        torch.tensor([0.1, 0.1, 0.1, 4.1, 4.1, 4.1, 0.7, 0.0], dtype=torch.float32),
    ]

    merged, keep_indices = _merge_patch_detections(
        detections,
        iou_threshold=0.2,
        agnostic_nms=False,
        max_det=10,
    )

    assert merged.shape == (1, 8)
    assert keep_indices == [0]


def test_predict_loaded_image_uses_patchwise_voxel_resampling(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    loaded_image = LoadedImageTask(
        case_name="case_patch",
        affine=np.eye(4, dtype=np.float32),
        image=np.zeros((4, 5, 6), dtype=np.float32),
        voxel_size_xyz=(1.0, 2.0, 3.0),
    )

    def fake_patchwise_predict(*args, **kwargs):
        captured["loaded_image"] = kwargs["loaded_image"] if "loaded_image" in kwargs else args[0]
        captured["prediction_config"] = kwargs["prediction_config"]
        return predict_module.WriteTask(
            case_name="case_patch",
            image_results=[],
            affine=np.eye(4, dtype=np.float32),
            original_shape=(6, 5, 4),
            prototype_masks=None,
            mask_detections=[],
        )

    monkeypatch.setattr(predict_module, "_predict_patchwise_image", fake_patchwise_predict)

    write_task = _predict_loaded_image(
        loaded_image,
        model=_DummyModel(),
        train_config=_DummyTrainConfig(training_mode="patchwise"),
        names={0: "class_0"},
        prediction_config=PredictionConfig(
            model_path=tmp_path / "model.pt",
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            confidence_threshold=0.1,
            nms_iou_threshold=0.2,
            device="cpu",
            clip_boxes_to_image=False,
            overwrite_voxel_size=VoxelSize(x=1.5, y=1.5, z=1.5),
        ),
    )

    assert captured["loaded_image"].voxel_size_xyz == (1.0, 2.0, 3.0)
    assert captured["prediction_config"].overwrite_voxel_size == VoxelSize(x=1.5, y=1.5, z=1.5)
    assert write_task.case_name == "case_patch"


def test_predict_loaded_image_uses_checkpoint_voxel_size_by_default(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    loaded_image = LoadedImageTask(
        case_name="case_voxel_default",
        affine=np.eye(4, dtype=np.float32),
        image=np.zeros((4, 5, 6), dtype=np.float32),
        voxel_size_xyz=(2.0, 2.0, 2.0),
    )

    def fake_preprocess_image(
        image: torch.Tensor,
        normalization,
        *,
        target_shape=None,
        input_voxel_size=None,
        target_voxel_size=None,
        apply_resampling: bool,
        apply_normalization: bool,
        output_dtype: torch.dtype,
    ):
        captured["target_shape"] = target_shape
        captured["input_voxel_size"] = input_voxel_size
        captured["target_voxel_size"] = target_voxel_size
        return image.to(dtype=output_dtype), (6, 5, 4), (6, 10, 8)

    monkeypatch.setattr(predict_module, "preprocess_image", fake_preprocess_image)
    monkeypatch.setattr(
        predict_module,
        "_postprocess_single",
        lambda model_output, train_config, prediction_config, num_classes: (
            torch.zeros((0, 8), dtype=torch.float32),
            None,
        ),
    )

    _predict_loaded_image(
        loaded_image,
        model=_DummyModel(),
        train_config=_DummyTrainConfig(voxel_size=VoxelSize(x=1.0, y=1.0, z=1.0)),
        names={0: "class_0"},
        prediction_config=PredictionConfig(
            model_path=tmp_path / "model.pt",
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            confidence_threshold=0.1,
            nms_iou_threshold=0.2,
            device="cpu",
            clip_boxes_to_image=False,
        ),
    )

    assert captured["target_shape"] is None
    assert captured["input_voxel_size"] == VoxelSize(x=2.0, y=2.0, z=2.0)
    assert captured["target_voxel_size"] == VoxelSize(x=1.0, y=1.0, z=1.0)


def test_predict_patchwise_image_moves_image_before_preprocessing_when_enabled(
    monkeypatch,
    tmp_path,
) -> None:
    events: list[str] = []
    loaded_image = LoadedImageTask(
        case_name="case_gpu_preprocess",
        affine=np.eye(4, dtype=np.float32),
        image=np.zeros((4, 5, 6), dtype=np.float32),
    )
    model = _DummyModel()
    model.device = torch.device("cuda")
    original_tensor_to = torch.Tensor.to

    def fake_tensor_to(self, *args, **kwargs):
        target_device = kwargs.get("device")
        remaining_args = args
        remaining_kwargs = dict(kwargs)
        if args and isinstance(args[0], torch.device | str):
            target_device = args[0]
            remaining_args = args[1:]
        if target_device is not None and torch.device(target_device).type == "cuda":
            events.append("to_cuda")
            if remaining_kwargs.pop("device", None) is not None:
                return original_tensor_to(self, *remaining_args, **remaining_kwargs)
            if remaining_args or remaining_kwargs:
                return original_tensor_to(self, *remaining_args, **remaining_kwargs)
            return self
        return original_tensor_to(self, *args, **kwargs)

    def fake_preprocess_image(
        image: torch.Tensor,
        normalization,
        *,
        target_shape=None,
        input_voxel_size=None,
        target_voxel_size=None,
        apply_resampling: bool,
        apply_normalization: bool,
        output_dtype: torch.dtype,
    ):
        del normalization, target_shape, input_voxel_size, target_voxel_size
        del apply_resampling, apply_normalization, output_dtype
        events.append("preprocess")
        return image, (6, 5, 4), (6, 5, 4)

    def fake_extract_tensor_patch(image, patch_origin_xyz, patch_shape_xyz, fill_value):
        del image, patch_origin_xyz, patch_shape_xyz, fill_value
        events.append("extract_patch")
        return torch.zeros((1, 12, 11, 10), dtype=torch.float32)

    monkeypatch.setattr(torch.Tensor, "to", fake_tensor_to)
    monkeypatch.setattr(
        predict_module.torch, "autocast", lambda *_args, **_kwargs: contextlib.nullcontext()
    )
    monkeypatch.setattr(predict_module, "preprocess_image", fake_preprocess_image)
    monkeypatch.setattr(predict_module, "_extract_tensor_patch", fake_extract_tensor_patch)
    monkeypatch.setattr(
        predict_module, "_iter_patch_origins", lambda *_args, **_kwargs: [(0, 0, 0)]
    )
    monkeypatch.setattr(
        predict_module,
        "_postprocess_single",
        lambda model_output, train_config, prediction_config, num_classes: (
            torch.zeros((0, 8), dtype=torch.float32),
            None,
        ),
    )

    _predict_patchwise_image(
        loaded_image,
        model=model,
        train_config=_DummyTrainConfig(training_mode="patchwise"),
        names={0: "class_0"},
        prediction_config=PredictionConfig(
            model_path=tmp_path / "model.pt",
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            confidence_threshold=0.1,
            nms_iou_threshold=0.2,
            device="0",
            clip_boxes_to_image=False,
            preprocessing_on_gpu=True,
        ),
    )

    assert events[:3] == ["to_cuda", "preprocess", "extract_patch"]


def test_predict_patchwise_image_moves_only_patches_when_disabled(monkeypatch, tmp_path) -> None:
    events: list[str] = []
    loaded_image = LoadedImageTask(
        case_name="case_gpu_model",
        affine=np.eye(4, dtype=np.float32),
        image=np.zeros((4, 5, 6), dtype=np.float32),
    )
    model = _DummyModel()
    model.device = torch.device("cuda")
    original_tensor_to = torch.Tensor.to

    def fake_tensor_to(self, *args, **kwargs):
        target_device = kwargs.get("device")
        remaining_args = args
        remaining_kwargs = dict(kwargs)
        if args and isinstance(args[0], torch.device | str):
            target_device = args[0]
            remaining_args = args[1:]
        if target_device is not None and torch.device(target_device).type == "cuda":
            events.append("to_cuda")
            if remaining_kwargs.pop("device", None) is not None:
                return original_tensor_to(self, *remaining_args, **remaining_kwargs)
            if remaining_args or remaining_kwargs:
                return original_tensor_to(self, *remaining_args, **remaining_kwargs)
            return self
        return original_tensor_to(self, *args, **kwargs)

    def fake_preprocess_image(
        image: torch.Tensor,
        normalization,
        *,
        target_shape=None,
        input_voxel_size=None,
        target_voxel_size=None,
        apply_resampling: bool,
        apply_normalization: bool,
        output_dtype: torch.dtype,
    ):
        del image, normalization, target_shape, input_voxel_size, target_voxel_size
        del apply_resampling, apply_normalization, output_dtype
        events.append("preprocess")
        return torch.zeros((1, 6, 5, 4), dtype=torch.float32), (6, 5, 4), (6, 5, 4)

    def fake_extract_tensor_patch(image, patch_origin_xyz, patch_shape_xyz, fill_value):
        del image, patch_origin_xyz, patch_shape_xyz, fill_value
        events.append("extract_patch")
        return torch.zeros((1, 12, 11, 10), dtype=torch.float32)

    monkeypatch.setattr(torch.Tensor, "to", fake_tensor_to)
    monkeypatch.setattr(
        predict_module.torch, "autocast", lambda *_args, **_kwargs: contextlib.nullcontext()
    )
    monkeypatch.setattr(predict_module, "preprocess_image", fake_preprocess_image)
    monkeypatch.setattr(predict_module, "_extract_tensor_patch", fake_extract_tensor_patch)
    monkeypatch.setattr(
        predict_module, "_iter_patch_origins", lambda *_args, **_kwargs: [(0, 0, 0)]
    )
    monkeypatch.setattr(
        predict_module,
        "_postprocess_single",
        lambda model_output, train_config, prediction_config, num_classes: (
            torch.zeros((0, 8), dtype=torch.float32),
            None,
        ),
    )

    _predict_patchwise_image(
        loaded_image,
        model=model,
        train_config=_DummyTrainConfig(training_mode="patchwise"),
        names={0: "class_0"},
        prediction_config=PredictionConfig(
            model_path=tmp_path / "model.pt",
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            confidence_threshold=0.1,
            nms_iou_threshold=0.2,
            device="0",
            clip_boxes_to_image=False,
            preprocessing_on_gpu=False,
        ),
    )

    assert events[:3] == ["preprocess", "extract_patch", "to_cuda"]
