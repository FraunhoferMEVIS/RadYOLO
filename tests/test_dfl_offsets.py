from __future__ import annotations

import torch

from radyolo.models.layers import Detect
from radyolo.models.layers import DFL
from radyolo.training.loss import BboxLoss
from radyolo.training.loss import DFLoss
from radyolo.training.tal import dist2bbox
from radyolo.training.tal import bbox2dist


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


def test_dist2bbox_reorders_invalid_extents() -> None:
    anchor_points = torch.tensor([[5.0, 5.0, 5.0]])
    pred_offsets = torch.tensor([[0.0, 0.0, 0.0, -1.0, -2.0, -3.0]])

    decoded = dist2bbox(pred_offsets, anchor_points, xyzwhd=False)

    assert torch.equal(decoded, torch.tensor([[4.0, 3.0, 2.0, 5.0, 5.0, 5.0]]))