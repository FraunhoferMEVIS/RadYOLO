# Ultralytics YOLO 🚀, AGPL-3.0 license

from __future__ import annotations

from typing import TYPE_CHECKING, cast
import logging

import torch
from torch import nn
from torch.nn import functional as F

from yolo11_3d.ultralytics.utils.metrics import bbox_iou
from yolo11_3d.ultralytics.utils.ops import xyzwhd2xyzxyz
from yolo11_3d.ultralytics.utils.tal import (
    TaskAlignedAssigner,
    bbox2dist,
    dist2bbox,
    make_anchors,
)

if TYPE_CHECKING:
    from yolo11_3d.nn.data import Batch
    from yolo11_3d.nn.layers import Detect
    from yolo11_3d.nn.model import HardcodedModel

LOGGER = logging.getLogger(__name__)

class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)

class DiceLoss:
    def __call__(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1):
        intersection = (pred * target).sum()
        summed = pred.sum() + target.sum()
        dice = (2. * intersection + eps) / (summed + eps)
        return 1 - dice

class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xyzwhd=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = (
                self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]
                )
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class v8DetectionLoss:  # noqa: N801
    """Criterion class for computing training losses."""

    def __init__(
        self,
        model: HardcodedModel,
        box: float,
        cls: float,
        dfl: float,
        mask_gain: float,
        class_separate_gain: float,
        detect_layer: Detect,
        device: torch.device,
        tal_topk: int = 10,
        tal_alpha: float = 0.5,
        tal_beta: float = 6.0,
        tal_fallback: bool = True,
        enable_mask_loss: bool = False,
        separate_objectness_head: bool = False,
    ) -> None:  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        m = detect_layer
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.box_gain = box
        self.classification_gain = cls
        self.class_separate_gain = class_separate_gain
        self.dfl_gain = dfl
        self.mask_gain = mask_gain
        self.stride: torch.Tensor = m.stride  # model strides
        self.num_classes: int = m.nc  # number of classes
        self.separate_objectness_head: bool = separate_objectness_head
        self.no: int = (m.nc + 1 if self.separate_objectness_head else m.nc) + m.reg_max * 6
        self.reg_max: int = m.reg_max
        self.device: torch.device = device

        self.use_dfl: bool = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.num_classes,
            alpha=tal_alpha,
            beta=tal_beta,
            use_fallback_if_alignment_zero=tal_fallback,
            anchor_inclusion_distance= -2 if model.include_stride4_output else -4
        )
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj: torch.Tensor = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.enable_mask_loss = enable_mask_loss
        self.mask_dice_loss = DiceLoss()

    def preprocess(
        self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out: torch.Tensor = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = cast("tuple[torch.Tensor, torch.Tensor]", i.unique(return_counts=True))
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, int(counts.max()), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:7] = xyzwhd2xyzxyz(out[..., 1:7].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = (
                pred_dist.view(b, a, 6, c // 6).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(
        self,
        preds: tuple[list[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]],
                     torch.Tensor | None,
                     torch.Tensor | None],
        batch: Batch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls / objectness, dfl, mask, class separate multiplied by batch size."""
        loss = torch.zeros(5, device=self.device)  # box, cls_obj, dfl, mask, cls_sep
        feats = preds[0]
        feats = feats[1] if isinstance(feats, tuple) else feats # Handling of inference pass
        flat = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        )
        if self.separate_objectness_head:
            pred_distri, pred_cls_logits, pred_obj_logit = flat.split((self.reg_max * 6, self.num_classes, 1), 1)
            pred_class_logits = pred_cls_logits.permute(0, 2, 1).contiguous()  # (B, A, C) logits
            pred_objectness_logit = pred_obj_logit.permute(0, 2, 1).contiguous()  # (B, A, 1) logit
        else:
            pred_distri, pred_class_logits = flat.split((self.reg_max * 6, self.num_classes), 1)
            pred_class_logits = pred_class_logits.permute(0, 2, 1).contiguous()  # (B, A, C) logits for BCE-with-logits
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_class_logits.dtype
        batch_size = pred_class_logits.shape[0]
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        )  # image size (d,h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[2, 1, 0, 2, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 6), 2)  # cls, xyzxyz
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyzxyz, (b, h*w*d, 6)
        pred_bboxes_image_coordinates = (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype)

        # For assignment: use objectness when available so a prediction score exists even if no class selected
        if self.separate_objectness_head:
            assign_scores = pred_objectness_logit.detach().squeeze(-1).sigmoid()  # (B, A)
            # Expand to (B, A, C) so assigner API stays the same
            assign_scores = assign_scores.unsqueeze(-1).expand(-1, -1, self.num_classes)
        else:
            assign_scores = pred_class_logits.detach().sigmoid()
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            assign_scores,
            pred_bboxes_image_coordinates,
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        # Ensure indices used for gathering are int64
        target_gt_idx = target_gt_idx.to(torch.int64)

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls/Objectness loss
        if self.separate_objectness_head:
            # For classification loss, build targets and ignore unknown class (-1)
            # target_scores here is one-hot (0/1) per class for fg anchors
            target_labels = target_scores.argmax(-1)  # (B, A)
            # Determine unknown mask from gt_labels and assigned gt indices
            B, A, C = target_scores.shape
            # If there are no GT labels in this batch (e.g., empty images), avoid invalid gather on size-0 dim
            if gt_labels.shape[1] > 0:
                safe_indices = target_gt_idx.unsqueeze(-1).clamp_min(0).clamp_max(gt_labels.shape[1] - 1)
                per_anchor_gt = gt_labels.gather(1, safe_indices)  # (B, A, 1)
                unknown_mask = (per_anchor_gt.squeeze(-1) < 0) & fg_mask.bool()
            else:
                unknown_mask = torch.zeros((B, A), dtype=torch.bool, device=fg_mask.device)
            # Cross entropy expects class indices and can take ignore_index
            ce_loss = self.ce(
                pred_class_logits.view(-1, C),
                target_labels.view(-1)
            )
            # Mask out ignored positions (unknown class)
            ce_loss = ce_loss.view(B, A)
            ce_loss = ce_loss.masked_fill(unknown_mask, 0.0)
            # Only count fg anchors
            denom = fg_mask.sum().clamp_min(1)
            ce_loss = (ce_loss * fg_mask).sum() / denom
            loss[4] = ce_loss
            # Objectness targets: 1 for fg, 0 for bg
            obj_target = fg_mask.float().unsqueeze(-1)
            loss_obj = self.bce(pred_objectness_logit, obj_target).sum() / target_scores_sum
            loss[1] = loss_obj
        else:
            # BCE-with-logits on multi-label scores as in YOLO
            loss[1] = self.bce(pred_class_logits, target_scores.to(dtype)).sum() / target_scores_sum

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

            # Mask loss
            if self.enable_mask_loss:
                mask_loss_count = 0
                mask_coefficients = preds[1]
                mask_coefficients = mask_coefficients.permute(0, 2, 1).contiguous()
                prototype_masks = preds[2]

                # Downsample cell grid by factor between image resolution and prototype mask resolution 
                image_shape = batch['shape'][0]
                downsampling_rate = int(image_shape[-1] / prototype_masks.shape[-1])
                target_bboxes_grid = target_bboxes / downsampling_rate
                pred_bboxes_grid = pred_bboxes_image_coordinates / downsampling_rate

                for batch_index, masks in enumerate(batch["masks"]):
                    target_gt_idx_batch = target_gt_idx[batch_index]
                    fg_mask_batch = fg_mask[batch_index]
                    prototype_masks_batch = prototype_masks[batch_index]
                    if masks is not None and masks.shape[0] >= 1:
                        masks = F.interpolate(masks[None, :], prototype_masks_batch.shape[1:], mode="nearest-exact")[0]
                        for mask_index, gt_mask in enumerate(masks):
                            if not gt_mask.any():
                                LOGGER.warning("Empty mask after downsampling. Skipping this mask.")
                                continue
                            grid_cell_indices = torch.nonzero((target_gt_idx_batch == mask_index) & (fg_mask_batch > 0)).squeeze(1)
                            if not grid_cell_indices.any():
                                LOGGER.warning("No assinged grid cells. Skipping this mask.")
                                continue
                            
                            # Compute box for cropping
                            target_box = target_bboxes_grid[batch_index, mask_index]
                            predicted_boxes = pred_bboxes_grid[batch_index, grid_cell_indices]
                            predicted_box = torch.cat([predicted_boxes[:,:3].amin(dim=0), predicted_boxes[:,3:].amax(dim=0)])
                            cropping_box = torch.cat([torch.amin(torch.cat([target_box[:3, None], predicted_box[:3, None]], dim=1), dim=1),
                                                      torch.amax(torch.cat([target_box[3:, None], predicted_box[3:, None]], dim=1), dim=1)])
                            cropping_box[:3] = cropping_box[:3].floor()
                            cropping_box[3:] = cropping_box[3:].ceil()
                            cropping_box[:3] = cropping_box[:3].clamp(min=0)
                            cropping_box[3] = cropping_box[3].clamp(max=prototype_masks.shape[2])
                            cropping_box[4] = cropping_box[4].clamp(max=prototype_masks.shape[3])
                            cropping_box[5] = cropping_box[5].clamp(max=prototype_masks.shape[4])
                            cropping_box = cropping_box.to(torch.int)

                            prototype_masks_batch_cropped = prototype_masks_batch[:,
                                                                                  cropping_box[0].item():cropping_box[3].item(),
                                                                                  cropping_box[1].item():cropping_box[4].item(),
                                                                                  cropping_box[2].item():cropping_box[5].item()]
                            mask_coefficients_i = mask_coefficients[batch_index, grid_cell_indices]
                            pred_masks = torch.einsum("in,nhwd->ihwd", mask_coefficients_i, prototype_masks_batch_cropped)  # (i, n_proto) @ (n_proto, h, w, d) -> (i, h, w, d)
                            pred_masks = torch.sigmoid(pred_masks)
                            pred_masks_average = pred_masks.mean(dim=0)
                            gt_mask = gt_mask.to(pred_masks_average.dtype)
                            gt_mask_cropped = gt_mask[cropping_box[0].item():cropping_box[3].item(),
                                                      cropping_box[1].item():cropping_box[4].item(),
                                                      cropping_box[2].item():cropping_box[5].item()]
                            mask_loss = self.mask_dice_loss(pred_masks_average, gt_mask_cropped)
                            loss[3] += mask_loss
                            mask_loss_count += 1
                if mask_loss_count > 0:
                    loss[3] = loss[3] / mask_loss_count

        loss[0] *= self.box_gain
        loss[1] *= self.classification_gain
        loss[2] *= self.dfl_gain
        loss[3] *= self.mask_gain
        loss[4] *= self.class_separate_gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl, mask)
