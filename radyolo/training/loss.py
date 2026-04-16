# Ultralytics YOLO 🚀, AGPL-3.0 license

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from radyolo.training.tal import (
    BoxParametrization,
    TaskAlignedAssigner,
    activate_box_prediction,
    decode_bboxes,
    encode_bbox_targets,
    make_anchors,
)
from radyolo.utils.ops import xyzwhd2xyzxyz
from radyolo.utils.util import unwrap_model
from radyolo.validation.metrics import bbox_iou

if TYPE_CHECKING:
    from radyolo.config.config import DefaultConfig
    from radyolo.data.data import Batch
    from radyolo.models.layers import Detect, ModelOutput
    from radyolo.models.model import HardcodedModel

LOGGER = logging.getLogger(__name__)


class QualityFocalLoss(nn.Module):
    """Quality focal loss for classification.

    Implements the Quality Focal Loss (https://arxiv.org/pdf/2006.04388) for addressing class
    imbalance by down-weighting easy examples and focusing on hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on
            hard-to-classify examples.
    """

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
        pred_prob = pred_logits.sigmoid()
        modulating_factor = torch.abs(pred_prob - target) ** self.gamma
        loss *= modulating_factor
        return loss.sum()


class DFLoss(nn.Module):
    """Class for computing DFL losses during training."""

    def __init__(self, reg_bins: int = 16, reg_min: float = 0.0) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_bins = reg_bins
        self.reg_min = reg_min
        self.reg_max = reg_min + reg_bins - 1

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(self.reg_min, self.reg_max - 0.01) - self.reg_min
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class DiceLoss:
    def __call__(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(1, 2, 3))
        summed = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = ((2.0 * intersection + eps) / (summed + eps)).mean()
        return 1 - dice


class BboxLoss(nn.Module):
    """Class for computing training losses during training."""

    def __init__(
        self,
        reg_bins: int = 16,
        reg_min: float = 0.0,
        bbox_parametrization: BoxParametrization = "ltrb",
        bbox_center_offset_limit: float = 1.2,
    ) -> None:
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.bbox_parametrization = bbox_parametrization
        self.bbox_center_offset_limit = bbox_center_offset_limit
        self.dfl_loss = DFLoss(reg_bins, reg_min=reg_min) if reg_bins > 1 else None
        if self.dfl_loss is not None and self.bbox_parametrization != "ltrb":
            msg = "DFL bbox loss requires bbox_parametrization='ltrb'"
            raise ValueError(msg)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        patch_size: torch.Tensor,
        stride_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xyzwhd=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = encode_bbox_targets(
                anchor_points,
                target_bboxes,
                bbox_parametrization=self.bbox_parametrization,
                reg_max=self.dfl_loss.reg_max,
                reg_min=self.dfl_loss.reg_min,
            )
            loss_dfl = (
                self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_bins), target_ltrb[fg_mask]
                )
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = encode_bbox_targets(
                anchor_points,
                target_bboxes,
                bbox_parametrization=self.bbox_parametrization,
            )
            scale = patch_size[[2, 1, 0, 2, 1, 0]].to(pred_dist.device, pred_dist.dtype)
            target_ltrb = target_ltrb * stride_tensor.unsqueeze(0)
            target_ltrb = target_ltrb / scale
            pred_dist = activate_box_prediction(
                pred_dist,
                self.bbox_parametrization,
                center_offset_limit=self.bbox_center_offset_limit,
            )
            pred_dist = pred_dist * stride_tensor.unsqueeze(0)
            pred_dist = pred_dist / scale
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none")
                .mean(-1, keepdim=True)
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class v8DetectionLoss:  # noqa: N801
    """Class for computing training losses."""

    def __init__(
        self,
        box: float,
        cls: float,
        dfl: float,
        dfl_reg_max: int,
        bbox_center_offset_limit: float,
        mask_loss_gain: float,
        class_separate_loss_gain: float,
        detect_layer: Detect,
        device: torch.device,
        include_stride4_output: bool,
        tal_topk: int = 10,
        tal_topk_final: int | None = None,
        tal_topk_final_epoch: int | None = None,
        tal_alpha: float = 0.5,
        tal_beta: float = 1.0,
        gamma: float = 2.0,
        tal_fallback: bool = True,
        enable_mask_loss: bool = False,
        separate_objectness_head: bool = False,
        objectness_loss: Literal["focal", "bce"] = "bce",
        bbox_parametrization: BoxParametrization = "ltrb",
    ) -> None:  # model must be de-paralleled
        m = detect_layer
        self.objectness_loss = (
            nn.BCEWithLogitsLoss(reduction="sum")
            if objectness_loss == "bce"
            else QualityFocalLoss(gamma=gamma)
        )
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.box_gain = box
        self.classification_gain = cls
        self.class_separate_loss_gain = class_separate_loss_gain
        self.dfl_gain = dfl
        self.mask_loss_gain = mask_loss_gain
        self.stride: torch.Tensor = m.stride  # model strides
        self.num_classes: int = m.nc  # number of classes
        self.separate_objectness_head: bool = separate_objectness_head
        self.reg_bins: int = m.dfl_bins
        self.reg_min: float = m.dfl_min
        self.reg_channels: int = m.reg_channels
        self.no: int = (m.nc + 1 if self.separate_objectness_head else m.nc) + self.reg_channels
        self.reg_max: int = dfl_reg_max
        self.device: torch.device = device
        self.bbox_parametrization = bbox_parametrization
        self.bbox_center_offset_limit = bbox_center_offset_limit
        self.tal_topk_initial = tal_topk
        self.tal_topk_final = tal_topk if tal_topk_final is None else tal_topk_final
        self.tal_topk_final_epoch = 1 if tal_topk_final_epoch is None else tal_topk_final_epoch

        self.use_dfl: bool = m.use_dfl
        if self.use_dfl and self.bbox_parametrization != "ltrb":
            msg = "DFL is only supported with bbox_parametrization='ltrb'"
            raise ValueError(msg)

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.num_classes,
            alpha=tal_alpha,
            beta=tal_beta,
            use_fallback_if_alignment_zero=tal_fallback,
            anchor_inclusion_distance=-2 if include_stride4_output else -4,
        )
        self.bbox_loss = BboxLoss(
            self.reg_bins,
            reg_min=self.reg_min,
            bbox_parametrization=self.bbox_parametrization,
            bbox_center_offset_limit=self.bbox_center_offset_limit,
        ).to(device)
        self.proj: torch.Tensor | None = None
        if self.use_dfl:
            self.proj = torch.arange(self.reg_bins, dtype=torch.float, device=device) + self.reg_min
        self.enable_mask_loss = enable_mask_loss
        self.mask_dice_loss = DiceLoss()
        self.mask_bce_loss = nn.BCEWithLogitsLoss()

    @staticmethod
    def interpolate_tal_topk(
        initial_topk: int,
        final_topk: int,
        current_epoch: int,
        final_epoch: int,
    ) -> int:
        if final_epoch <= 1:
            return final_topk
        clamped_epoch = min(max(current_epoch, 1), final_epoch)
        progress = (clamped_epoch - 1) / (final_epoch - 1)
        interpolated_topk = (1 - progress) * initial_topk + progress * final_topk
        return round(interpolated_topk)

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
            if self.proj is None:
                raise RuntimeError
            pred_dist = (
                pred_dist.view(b, a, 6, c // 6).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            )
        return decode_bboxes(
            pred_dist,
            anchor_points,
            bbox_parametrization=self.bbox_parametrization,
            xyzwhd=False,
            center_offset_limit=self.bbox_center_offset_limit,
        )

    def __call__(
        self,
        model_output: ModelOutput,
        batch: Batch,
        current_epoch: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the detection loss multiplied by batch size."""
        loss = torch.zeros(5, device=self.device)  # box, cls_obj, dfl, mask, cls_sep
        feats = model_output.raw_predictions
        flat = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2)
        if self.separate_objectness_head:
            pred_distri, pred_cls_logits, pred_obj_logit = flat.split(
                (self.reg_channels, self.num_classes, 1), 1
            )
            pred_class_logits = pred_cls_logits.permute(0, 2, 1).contiguous()  # (B, A, C) logits
            pred_objectness_logit = pred_obj_logit.permute(0, 2, 1).contiguous()  # (B, A, 1) logit
        else:
            pred_distri, pred_class_logits = flat.split((self.reg_channels, self.num_classes), 1)
            pred_class_logits = pred_class_logits.permute(
                0, 2, 1
            ).contiguous()  # (B, A, C) logits for BCE-with-logits
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_class_logits.dtype
        batch_size = pred_class_logits.shape[0]
        patch_size = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        )  # image size (d,h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=patch_size[[2, 1, 0, 2, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 6), 2)  # cls, xyzxyz
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyzxyz, (b, h*w*d, 6)
        pred_bboxes_image_coordinates = (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype)

        # For assignment: use objectness when available so a prediction score exists
        # even if no class is selected / assigned
        self.assigner.topk = (
            self.tal_topk_initial
            if current_epoch is None
            else self.interpolate_tal_topk(
                initial_topk=self.tal_topk_initial,
                final_topk=self.tal_topk_final,
                current_epoch=current_epoch,
                final_epoch=self.tal_topk_final_epoch,
            )
        )
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
            target_labels = target_scores.argmax(-1)  # (B, A)
            target_objectness_scores = target_scores.amax(dim=-1, keepdim=True)  # (B, A, 1)
            target_objectness_scores_sum = max(target_objectness_scores.sum(), 1)
            # Determine unknown mask from gt_labels and assigned gt indices
            batch_size, anchors, classes = target_scores.shape
            # If there are no GT labels in this batch, avoid invalid gather on size-0 dim
            if gt_labels.shape[1] > 0:
                safe_indices = (
                    target_gt_idx.unsqueeze(-1).clamp_min(0).clamp_max(gt_labels.shape[1] - 1)
                )
                per_anchor_gt = gt_labels.gather(1, safe_indices)  # (B, A, 1)
                unknown_mask = (per_anchor_gt.squeeze(-1) < 0) & fg_mask.bool()
            else:
                unknown_mask = torch.zeros(
                    (batch_size, anchors), dtype=torch.bool, device=fg_mask.device
                )
            # Cross entropy expects class indices and can take ignore_index
            ce_loss = self.ce(pred_class_logits.view(-1, classes), target_labels.view(-1))
            # Mask out ignored positions (unknown class)
            ce_loss = ce_loss.view(batch_size, anchors)
            ce_loss = ce_loss.masked_fill(unknown_mask, 0.0)
            # Only count fg anchors
            denom = fg_mask.sum().clamp_min(1)
            ce_loss = (ce_loss * fg_mask).sum() / denom
            loss[4] = ce_loss
            # BCE-with-logits on objectness score
            loss[1] = (
                self.objectness_loss(pred_objectness_logit, target_objectness_scores)
                / target_objectness_scores_sum
            )
        else:
            # BCE-with-logits on multi-label scores as in YOLO
            loss[1] = (
                self.objectness_loss(pred_class_logits, target_scores.to(dtype)) / target_scores_sum
            )

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
                patch_size,
                stride_tensor,
            )

            # Mask loss
            if self.enable_mask_loss:
                mask_loss_count = 0
                mask_coefficients = model_output.mask_coefficients
                mask_coefficients = mask_coefficients.permute(0, 2, 1).contiguous()
                prototype_masks = model_output.prototype_masks
                for batch_index, masks in enumerate(batch["masks"]):
                    target_gt_idx_batch = target_gt_idx[batch_index]
                    fg_mask_batch = fg_mask[batch_index]
                    prototype_masks_batch = prototype_masks[batch_index]
                    if masks is not None and masks.shape[0] >= 1:
                        masks_resampled = F.interpolate(
                            masks[None, :], prototype_masks_batch.shape[1:], mode="nearest-exact"
                        )[0]
                        for mask_index, gt_mask in enumerate(masks_resampled):
                            if not gt_mask.any():
                                LOGGER.warning("Empty mask after downsampling. Skipping this mask.")
                                continue
                            grid_cell_indices = torch.nonzero(
                                (target_gt_idx_batch == mask_index) & (fg_mask_batch > 0)
                            ).squeeze(1)
                            if not grid_cell_indices.any():
                                LOGGER.warning("No assinged grid cells. Skipping this mask.")
                                continue
                            mask_coefficients_i = mask_coefficients[batch_index, grid_cell_indices]
                            pred_masks_logits = torch.einsum(
                                "in,nhwd->ihwd", mask_coefficients_i, prototype_masks_batch
                            )  # (i, n_proto) @ (n_proto, h, w, d) -> (i, h, w, d)
                            gt_mask_correct_dtype = gt_mask.to(pred_masks_logits.dtype)
                            gt_mask_expanded = gt_mask_correct_dtype.expand(
                                pred_masks_logits.shape[0], -1, -1, -1
                            )
                            mask_dice_loss = self.mask_dice_loss(
                                pred_masks_logits, gt_mask_expanded
                            )
                            mask_bce_loss = self.mask_bce_loss(pred_masks_logits, gt_mask_expanded)
                            loss[3] += mask_dice_loss + mask_bce_loss
                            mask_loss_count += 1
                if mask_loss_count > 0:
                    loss[3] = loss[3] / mask_loss_count

        loss[0] *= self.box_gain
        loss[1] *= self.classification_gain
        loss[2] *= self.dfl_gain
        loss[3] *= self.mask_loss_gain
        loss[4] *= self.class_separate_loss_gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl, mask)


def build_detection_loss_from_config(
    train_config: DefaultConfig,
    model: nn.Module,
    device: torch.device,
) -> v8DetectionLoss:
    unwrapped_model = cast("HardcodedModel", unwrap_model(model))
    return v8DetectionLoss(
        box=train_config.box_loss_gain,
        cls=train_config.class_loss_gain,
        dfl=train_config.dfl_loss_gain,
        dfl_reg_max=train_config.dfl_reg_max,
        bbox_center_offset_limit=train_config.bbox_center_offset_limit,
        mask_loss_gain=train_config.mask_loss_gain,
        class_separate_loss_gain=train_config.class_separate_loss_gain,
        detect_layer=unwrapped_model.l23,
        device=device,
        include_stride4_output=unwrapped_model.include_stride4_output,
        tal_topk=train_config.tal_topk,
        tal_topk_final=train_config.tal_topk_final,
        tal_topk_final_epoch=train_config.tal_topk_final_epoch,
        tal_alpha=train_config.tal_alpha,
        tal_beta=train_config.tal_beta,
        gamma=train_config.gamma,
        tal_fallback=True,
        enable_mask_loss=train_config.segmentation_output,
        separate_objectness_head=train_config.separate_objectness_head,
        objectness_loss=train_config.objectness_loss,
        bbox_parametrization=train_config.bbox_parametrization,
    )
