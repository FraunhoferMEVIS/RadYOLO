from __future__ import annotations

import nibabel as nib
import numpy as np
import torch

from radyolo.inference import predict as predict_module
from radyolo.inference.predict import _box_image_to_world, _reconstruct_and_save_mask, _relative_box
from radyolo.models.layers import ModelOutput
from radyolo.utils.ops import non_max_suppression, prepare_model_output_for_nms, scale_boxes


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

    def fake_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        captured["boxes"] = boxes
        captured["scores"] = scores
        captured["iou_threshold"] = iou_threshold
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

    assert relative == [0.05, 0.05, 0.3, 0.2, 0.125, 0.6]


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

    world_box = _box_image_to_world(box, affine)

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

    mask_filename = _reconstruct_and_save_mask(
        det,
        proto,
        original_shape=(2, 2, 2),
        nifti_img=nifti_img,
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
