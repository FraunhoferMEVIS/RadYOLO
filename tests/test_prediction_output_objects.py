from __future__ import annotations

import torch

from radyolo.models.layers import ModelOutput
from radyolo.utils.ops import non_max_suppression, prepare_model_output_for_nms


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
