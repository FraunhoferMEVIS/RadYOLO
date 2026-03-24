from __future__ import annotations

import torch

from radyolo.models.layers import Detect, DFL
from radyolo.training.loss import BboxLoss, DFLoss
from radyolo.training.tal import (
    CENTER_OFFSET_LIMIT,
    bbox2center_extents,
    dist2bbox,
    bbox2dist,
    center_extents2bbox,
)


def test_bbox2dist_keeps_one_negative_bin() -> None:
    anchor_points = torch.tensor([[5.0, 5.0, 5.0]])
    bbox = torch.tensor([[5.5, 4.0, 4.0, 6.0, 6.0, 6.0]])

    dist = bbox2dist(anchor_points, bbox, reg_max=15.0, reg_min=-1.0)

    assert torch.isclose(dist[0, 0], torch.tensor(-0.5))
    assert dist.min().item() >= -1.0


def test_dfl_projection_supports_negative_offsets() -> None:
    dfl = DFL(in_channels=17, min_value=-1.0)
    logits = torch.full((1, 17 * 6, 1), -20.0)
    logits[0, 0, 0] = 20.0
    logits[0, 17 + 1, 0] = 20.0

    projected = dfl(logits)

    assert torch.isclose(projected[0, 0, 0], torch.tensor(-1.0), atol=1e-3)
    assert torch.isclose(projected[0, 1, 0], torch.tensor(0.0), atol=1e-3)


def test_dfl_loss_accepts_negative_targets() -> None:
    loss_fn = DFLoss(reg_bins=17, reg_min=-1.0)
    pred_dist = torch.full((1, 17), -20.0)
    pred_dist[0, 0] = 20.0
    loss = loss_fn(pred_dist, torch.tensor([[-0.75]]))

    assert torch.isfinite(loss).all()


def test_direct_regression_uses_raw_box_offsets() -> None:
    detect = Detect(num_classes=2, ch=(16,), dfl_reg_max=1)

    assert not detect.use_dfl
    assert detect.reg_channels == 6


def test_bbox_loss_direct_path_uses_l1_offsets() -> None:
    loss_fn = BboxLoss(reg_bins=1)
    anchor_points = torch.tensor([[5.0, 5.0, 5.0]])
    target_bboxes = torch.tensor([[[4.0, 4.0, 4.0, 6.0, 6.0, 6.0]]])
    pred_bboxes = target_bboxes.clone()
    pred_dist = bbox2dist(anchor_points, target_bboxes)
    target_scores = torch.ones((1, 1, 1))
    target_scores_sum = torch.tensor(1.0)
    fg_mask = torch.tensor([[True]])
    imgsz = torch.tensor([40.0, 40.0, 40.0])
    stride_tensor = torch.tensor([[4.0]])

    loss_iou, loss_l1 = loss_fn(
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        imgsz,
        stride_tensor,
    )

    assert torch.isclose(loss_iou, torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(loss_l1, torch.tensor(0.0), atol=1e-6)


def test_center_extents_decode_applies_constraints() -> None:
    anchor_points = torch.tensor([[5.0, 5.0, 5.0]])
    raw_prediction = torch.tensor([[10.0, -10.0, 0.0, -20.0, 2.0, 4.0]])

    decoded = center_extents2bbox(raw_prediction, anchor_points, xyzwhd=True)

    assert decoded[0, 0].item() <= 5.0 + CENTER_OFFSET_LIMIT + 1e-6
    assert decoded[0, 1].item() >= 5.0 - CENTER_OFFSET_LIMIT - 1e-6
    assert decoded[0, 3].item() > 0.0


def test_center_extents_decode_uses_configurable_offset_limit() -> None:
    anchor_points = torch.tensor([[5.0, 5.0, 5.0]])
    raw_prediction = torch.tensor([[10.0, 0.0, 0.0, 1.0, 1.0, 1.0]])

    decoded = center_extents2bbox(
        raw_prediction,
        anchor_points,
        xyzwhd=True,
        center_offset_limit=0.4,
    )

    assert decoded[0, 0].item() <= 5.4 + 1e-6
    assert decoded[0, 0].item() >= 5.39


def test_bbox2center_extents_preserves_center_offsets() -> None:
    anchor_points = torch.tensor([[5.0, 5.0, 5.0]])
    bbox = torch.tensor([[0.0, 0.0, 0.0, 20.0, 20.0, 20.0]])

    encoded = bbox2center_extents(anchor_points, bbox)

    assert torch.equal(encoded[0, :3], torch.tensor([5.0, 5.0, 5.0]))


def test_detect_center_extents_requires_direct_regression() -> None:
    try:
        Detect(num_classes=2, ch=(16,), dfl_reg_max=16, bbox_parametrization="center_extents")
    except ValueError as exc:
        assert "DFL" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected center_extents parametrization to reject DFL")


def test_detect_center_extents_bias_init_zeros_center_offsets() -> None:
    detect = Detect(num_classes=2, ch=(16,), dfl_reg_max=1, bbox_parametrization="center_extents")
    detect.stride = torch.tensor([8.0])

    detect.bias_init()

    box_last = detect.cv_boxes[0][-1]
    assert torch.allclose(box_last.bias[:3], torch.zeros(3))
    assert torch.all(box_last.bias[3:6] > 0.0)


def test_detect_center_extents_decode_uses_custom_offset_limit() -> None:
    detect = Detect(
        num_classes=2,
        ch=(16,),
        dfl_reg_max=1,
        bbox_parametrization="center_extents",
        center_offset_limit=0.25,
    )
    anchors = torch.tensor([[[5.0], [5.0], [5.0]]])
    raw_prediction = torch.tensor([[[10.0], [0.0], [0.0], [1.0], [1.0], [1.0]]])

    decoded = detect.decode_bboxes(raw_prediction, anchors, xyzwhd=True)

    assert decoded[0, 0, 0].item() <= 5.25 + 1e-6
    assert decoded[0, 0, 0].item() >= 5.24


def test_bbox_loss_center_extents_direct_path_uses_constrained_parameters() -> None:
    loss_fn = BboxLoss(reg_bins=1, bbox_parametrization="center_extents")
    anchor_points = torch.tensor([[5.0, 5.0, 5.0]])
    target_bboxes = torch.tensor([[[4.0, 4.0, 4.0, 6.0, 6.0, 6.0]]])
    target_params = bbox2center_extents(anchor_points, target_bboxes)
    raw_prediction = target_params.clone()
    raw_prediction[..., 3:] = torch.log(torch.expm1(target_params[..., 3:]))
    pred_bboxes = center_extents2bbox(raw_prediction, anchor_points, xyzwhd=False)
    target_scores = torch.ones((1, 1, 1))
    target_scores_sum = torch.tensor(1.0)
    fg_mask = torch.tensor([[True]])
    imgsz = torch.tensor([40.0, 40.0, 40.0])
    stride_tensor = torch.tensor([[4.0]])

    loss_iou, loss_l1 = loss_fn(
        raw_prediction,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        imgsz,
        stride_tensor,
    )

    assert torch.isclose(loss_iou, torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(loss_l1, torch.tensor(0.0), atol=1e-6)


def test_bbox_loss_center_extents_direct_path_uses_custom_offset_limit() -> None:
    loss_fn = BboxLoss(
        reg_bins=1,
        bbox_parametrization="center_extents",
        bbox_center_offset_limit=0.25,
    )
    anchor_points = torch.tensor([[5.0, 5.0, 5.0]])
    target_scores = torch.ones((1, 1, 1))
    target_scores_sum = torch.tensor(1.0)
    fg_mask = torch.tensor([[True]])
    imgsz = torch.tensor([40.0, 40.0, 40.0])
    stride_tensor = torch.tensor([[4.0]])

    raw_prediction = torch.tensor([[[10.0, 0.0, 0.0, 0.0, 1.0, 1.0]]])
    pred_bboxes = center_extents2bbox(
        raw_prediction,
        anchor_points,
        xyzwhd=False,
        center_offset_limit=0.25,
    )
    target_bboxes = pred_bboxes.clone()

    loss_iou, loss_l1 = loss_fn(
        raw_prediction,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        imgsz,
        stride_tensor,
    )

    assert torch.isclose(loss_iou, torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(loss_l1, torch.tensor(0.0), atol=1e-6)


def test_dist2bbox_reorders_invalid_extents() -> None:
    anchor_points = torch.tensor([[5.0, 5.0, 5.0]])
    pred_offsets = torch.tensor([[0.0, 0.0, 0.0, -1.0, -2.0, -3.0]])

    decoded = dist2bbox(pred_offsets, anchor_points, xyzwhd=False)

    assert torch.equal(decoded, torch.tensor([[4.0, 3.0, 2.0, 5.0, 5.0, 5.0]]))
