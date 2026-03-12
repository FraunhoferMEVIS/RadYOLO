# Ultralytics YOLO 🚀, AGPL-3.0 license

from __future__ import annotations

import logging

import torch
from torch import nn

from radyolo.validation.metrics import bbox_iou

LOGGER = logging.getLogger(__name__)


class TaskAlignedAssigner(nn.Module):
    """A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which
    combines both classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned
        metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
        anchor_inclusion_distance (float): Allow to match anchors that are slightly outside
            the gt_box. Should be set to half mininum model stride.
    """

    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-9,
        use_fallback_if_alignment_zero: bool = True,
        anchor_inclusion_distance: float = -2,
    ) -> None:
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.use_fallback_if_alignment_zero = use_fallback_if_alignment_zero
        self.anchor_inclusion_distance = anchor_inclusion_distance

    @torch.no_grad()
    def forward(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, ...]
    ):
        """Compute the task-aligned assignment.

        Reference code is available at
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py.

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                # Indices tensor must be int64 for downstream gather/index ops
                torch.zeros_like(pd_scores[..., 0], dtype=torch.int64),
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except torch.OutOfMemoryError:
            # Move tensors to CPU, compute, then move back to original device
            LOGGER.warning("WARNING: CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
            cpu_tensors = [
                t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
            ]
            result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)

    def _forward(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the task-aligned assignment.

        Reference code is available at
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py.

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
        """
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes
        )

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )
        # Note: At this point, target_scores consists of values from the set (0, 1).

        # Normalize

        # align_metric is filtered using mask_pos
        align_metric *= mask_pos  # b, max_num_obj, num_anchors

        # for each object in max_num_obj the largest align_metric over all anchor points is selected
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj, 1

        # for each object in max_num_obj the largest overlap (filtered using mask_pos) over all
        # anchor points is selected
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj, 1

        # Reminder: align_metric consists of weighted product of pred_score and overlaps.
        #
        # For each object in max_num_obj, the align_metric (b, max_num_obj, num_anchors) of all
        # anchors is scaled by fraction (b, max_num_obj, 1) of largest overlap and largest align
        # metric.
        #
        # This line is somewhat obscure and does several semantic operations.
        #
        # The first operation is `align_metric / pos_align_metrics` (ignoring the self.eps).
        # Since `pos_align_metrics` contains maxima from `align_metric`, this results in a
        # normalization of all values along axis -1 to range (0, 1).
        #
        # As second operation, these normalized values are then scaled by the largest overlap,
        # resulting in an effective rescale to the range of overlaps.
        #
        # For each individual value the calculation can be summarized to
        # $ (pred_score * overlap / max(pred_score * overlap)) * overlap $
        # This results in a score which is weighted based on pred_score and the corresponding
        # weights, but still retains the value range of overlap.
        #
        # For each anchor the largest align_metric is selected and a dimension appended resulting in
        # a shape of (b, num_anchors, 1).
        norm_align_metric = (
            (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        )

        # For each anchor the class-wise target scores are scaled respectively by the largest
        # normalized align_metric.
        # Since target_scores are effectively a boolean mask, this means that that the normalized
        # align metrics are selected according to the target_scores mask
        target_scores = (
            target_scores * norm_align_metric
        )  # (b, num_anchors, num_classes) * (b, num_anchors, 1)

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        ground_truth_labels: torch.Tensor,
        ground_truth_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        ground_truth__mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get anchors_in_gts mask, (b, max_num_obj, num_anchors)."""
        # Create mask which contains for each ground truth bounding box all anchor points that are
        # included in respective bbox area. (b, max_num_obj, num_anchors)
        anchors_in_gts__mask = self.select_candidates_in_gts(
            anchor_points, ground_truth_bboxes, self.anchor_inclusion_distance
        )

        # Get anchor align_metric, (b, max_num_obj, num_anchors)
        align_metric, overlaps = self.get_box_metrics(
            pred_scores,
            pred_bboxes,
            ground_truth_labels,
            ground_truth_bboxes,
            anchors_in_gts__mask * ground_truth__mask,
        )

        # Get topk_metric mask, (b, max_num_obj, num_anchors)
        mask_topk = self.select_topk_candidates(
            align_metric, topk_mask=ground_truth__mask.expand(-1, -1, self.topk).bool()
        )

        # Boolean mask: (B, M) True if all align_metric entries are zero for this GT box
        zero_mask = (align_metric.sum(dim=2) == 0) & (ground_truth__mask.sum(dim=2) > 0)

        if self.use_fallback_if_alignment_zero and zero_mask.any():
            LOGGER.warning(
                f"Using fall back mechanism to assign alignment metrics because alignment for all "
                f"grid cells was zero for {zero_mask.sum().item()} objects."
            )
            batch_size, max_num_objects, num_anchors = align_metric.shape

            # Expand GT and predicted boxes for broadcasting
            gt_boxes_exp = ground_truth_bboxes.unsqueeze(2).expand(
                -1, -1, num_anchors, -1
            )  # (B, M, A, 6)
            pred_boxes_exp = pred_bboxes.unsqueeze(1).expand(
                -1, max_num_objects, -1, -1
            )  # (B, M, A, 6)

            # Compute Euclidean distance in 6D (between corners of GT and anchor boxes)
            distances = torch.norm(gt_boxes_exp - pred_boxes_exp, dim=3)  # (B, M, A)
            max_distance = distances.max()
            distances = torch.where(anchors_in_gts__mask == 0, max_distance, distances)

            # Loop over batch and GT boxes where zero_mask is True
            for b in range(batch_size):
                for m in range(max_num_objects):
                    if zero_mask[b, m]:
                        # Get indices of k closest anchors
                        topk_vals, topk_idx = torch.topk(distances[b, m], k=8, largest=False)

                        # Assign alignment metric based on inverse distance scaling
                        # Closest gets 0.1, futher anchors get proportionally less
                        norm_dist = (topk_vals - topk_vals.min()) / (
                            topk_vals.max() - topk_vals.min()
                        )
                        new_metrics = 0.1 * (1 - norm_dist)

                        # Update align_metric, overlaps, mask_topk, anchors_in_gts__mask for these
                        # anchors. The resulting values are technically not correct, but allow
                        # the model to learn some foreground anchors for the objects
                        align_metric[b, m, topk_idx] = new_metrics
                        overlaps[b, m, topk_idx] = new_metrics
                        mask_topk[b, m, topk_idx] = 1

        # Merge all mask to a final mask, (b, max_num_obj, num_anchors)
        mask_pos = mask_topk * anchors_in_gts__mask * ground_truth__mask

        boxes_without_anchor = ground_truth__mask.sum(2).gt(mask_pos.sum(2)).sum().item()
        if boxes_without_anchor:
            LOGGER.warning(
                f"Did not find anchor points for {boxes_without_anchor} ground-truth bounding "
                f"boxes. Your bounding boxes could be too small for the configured heads and "
                f"input resolution."
            )

        return mask_pos, align_metric, overlaps

    def get_box_metrics(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        ground_truth_labels: torch.Tensor,
        ground_truth_bboxes: torch.Tensor,
        ground_truth__mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute alignment metric given predicted and ground truth bounding boxes."""
        na = pred_bboxes.shape[-2]
        ground_truth__mask = ground_truth__mask.bool()  # b, max_num_obj, h*w*d
        overlaps = torch.zeros(
            [self.bs, self.n_max_boxes, na], dtype=pred_bboxes.dtype, device=pred_bboxes.device
        )
        bbox_scores = torch.zeros(
            [self.bs, self.n_max_boxes, na], dtype=pred_scores.dtype, device=pred_scores.device
        )

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = (
            torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        )  # b, max_num_obj
        # Clamp unknown class (-1) to 0 only to avoid indexing errors; loss handles unknown classes
        ind[1] = ground_truth_labels.squeeze(-1).clamp_min(0)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        bbox_scores[ground_truth__mask] = pred_scores[ind[0], :, ind[1]][
            ground_truth__mask
        ]  # b, max_num_obj, h*w*d

        # (b, max_num_obj, 1, 6), (b, 1, h*w*d, 6)
        pd_boxes = pred_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[ground_truth__mask]
        gt_boxes = ground_truth_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[ground_truth__mask]
        overlaps[ground_truth__mask] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes: torch.Tensor, pd_bboxes: torch.Tensor) -> torch.Tensor:
        """IoU calculation for horizontal bounding boxes."""
        return bbox_iou(gt_bboxes, pd_bboxes, xyzwhd=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(
        self, metrics: torch.Tensor, largest: bool = True, topk_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Select the top-k candidates based on the given metrics.

        Args:
            metrics (Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size,
                              max_num_obj is the maximum number of objects, and h*w represents the
                              total number of anchor points.
            largest (bool): If True, select the largest values; otherwise, select smallest values.
            topk_mask (Tensor): An optional boolean tensor of shape (b, max_num_obj, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on given metrics.

        Returns:
            (Tensor): Tensor of shape (b, max_num_obj, d*h*w) containing selected top-k candidates.
        """
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(
        self,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute target labels, target bounding boxes, and target scores for positive anchor points.

        Args:
            gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 6).
            target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w*d), where h*w*d is the total
                                    number of anchor points.
            fg_mask (Tensor): A boolean tensor of shape (b, h*w*d) indicating the positive
                              (foreground) anchor points.

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                - target_labels (Tensor): Shape (b, h*w*d), containing the target labels for
                                          positive anchor points.
                - target_bboxes (Tensor): Shape (b, h*w*d, 6), containing the target bounding boxes
                                          for positive anchor points.
                - target_scores (Tensor): Shape (b, h*w*d, num_classes), containing the target
                                          scores for positive anchor points, where num_classes
                                          is the number of object classes.
        """
        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w*d)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w*d)

        # Assigned target boxes, (b, max_num_obj, 6) -> (b, h*w*d, 6)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w*d, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w*d, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(
        xy_centers: torch.Tensor, gt_bboxes: torch.Tensor, anchor_inclusion_distance: float
    ) -> torch.Tensor:
        """Selection of positive anchors within ground truth boxes.

        Select positive anchor centers within ground truth bounding boxes.
        The anchor_inclusion_distance ensures that there are also anchor candidates
        for ground truth bounding boxes that are very small and fall between anchor centers.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w*d, 3).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 6).
            anchor_inclusion_distance (float, optional): Value to indicate the per axis minimal
                anchor inclusion distance

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w*d).

        Note:
            b: batch size, n_boxes: number of ground truth boxes, h: height, w: width, d: depth.
            Bounding box format: [x_min, y_min, z_min, x_max, y_max, z_max].
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 6).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(
            bs, n_boxes, n_anchors, -1
        )
        pos_anchors = bbox_deltas.amin(3).gt_(anchor_inclusion_distance)
        return pos_anchors  # noqa: RET504

    @staticmethod
    def select_highest_overlaps(
        mask_pos: torch.Tensor, overlaps: torch.Tensor, n_max_boxes: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select anchor boxes with highest IoU when assigned to multiple ground truths.

        Args:
            mask_pos (torch.Tensor): Positive mask, shape (b, n_max_boxes, h*w*d).
            overlaps (torch.Tensor): IoU overlaps, shape (b, n_max_boxes, h*w*d).
            n_max_boxes (int): Maximum number of ground truth boxes.

        Returns:
            target_gt_idx (torch.Tensor): Indices of assigned ground truths, shape (b, h*w*d).
            fg_mask (torch.Tensor): Foreground mask, shape (b, h*w*d).
            mask_pos (torch.Tensor): Updated positive mask, shape (b, n_max_boxes, h*w*d).

        Note:
            b: batch size, h: height, w: width, d: depth.
        """
        # Keep a copy of the original candidate mask to know which anchors were valid for each GT
        # before resolving conflicts. This helps us ensure every GT with candidates gets at least
        # one anchor in the final assignment.
        orig_mask_pos = mask_pos.clone()

        # Convert (b, n_max_boxes, h*w*d) -> (b, h*w*d)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(
                -1, n_max_boxes, -1
            )  # (b, n_max_boxes, h*w*d)
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w*d)

            is_max_overlaps = torch.zeros(
                mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device
            )
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(
                mask_multi_gts, is_max_overlaps, mask_pos
            ).float()  # (b, n_max_boxes, h*w*d)
            fg_mask = mask_pos.sum(-2)

        # Ensure each GT with at least one candidate in the original mask retains at least one
        # unique anchor assignment after conflict resolution.
        # Shapes
        B, _M, A = mask_pos.shape
        # Count anchors per GT after conflict resolution
        per_gt_counts = mask_pos.sum(-1)  # (B, M)

        # A GT is considered "eligible" if it had candidates originally (avoid padded GTs).
        gt_had_candidates = orig_mask_pos.sum(-1) > 0  # (B, M)

        # For each batch element, rebalance as needed.
        for b in range(B):
            # Indices of GTs that had candidates but currently have 0 assigned anchors
            need_assign = torch.nonzero(
                (per_gt_counts[b] == 0) & gt_had_candidates[b], as_tuple=False
            ).flatten()
            if need_assign.numel() == 0:
                continue

            LOGGER.warning(
                f"Using fall back mechanism to assign targets for {need_assign.numel()} objects "
                f"because they had no anchors left after highest overlap filtering."
            )

            # For fast lookup of current owner per anchor (unique after conflict resolution)
            # owner[j] = gt index that currently owns anchor j, or -1 if unassigned
            # Build owner by argmax over GT dim where there is a 1
            owner = torch.full((A,), -1, device=mask_pos.device, dtype=torch.long)
            # anchors that are currently assigned
            assigned_anchors = torch.nonzero(fg_mask[b] > 0, as_tuple=False).flatten()
            if assigned_anchors.numel() > 0:
                # For these anchors, find gt index with value==1
                gt_idx_for_anchor = mask_pos[b, :, assigned_anchors].argmax(0)
                owner[assigned_anchors] = gt_idx_for_anchor

            # Prepare per-gt sorted candidate anchors by descending overlap
            # We only consider original candidates for that GT
            # overlaps[b, m, :] may be zero for non-candidates; sorting still works
            for m in need_assign.tolist():
                candidates_mask = orig_mask_pos[b, m].bool()
                if not candidates_mask.any():
                    # No valid candidates originally -> skip (likely padded GT)
                    continue

                # Sort candidate anchors by descending IoU overlap
                cand_overlaps = overlaps[b, m]
                cand_indices = torch.nonzero(candidates_mask, as_tuple=False).flatten()
                # If for some reason overlaps are identical, torch.sort is deterministic
                _sorted_vals, order = torch.sort(cand_overlaps[cand_indices], descending=True)
                sorted_candidates = cand_indices[order]

                # Try to assign the best possible anchor to this GT.
                assigned = False
                for j in sorted_candidates.tolist():
                    cur_owner = owner[j].item()
                    if cur_owner == -1:
                        # Anchor is currently free: assign it to GT m
                        mask_pos[b, m, j] = 1.0
                        per_gt_counts[b, m] += 1
                        fg_mask[b, j] += 1
                        owner[j] = m
                        assigned = True
                        break
                    if cur_owner != m and per_gt_counts[b, cur_owner] > 1:
                        # Steal from current owner only if that owner will still have >=1 anchor
                        mask_pos[b, cur_owner, j] = 0.0
                        per_gt_counts[b, cur_owner] -= 1
                        mask_pos[b, m, j] = 1.0
                        per_gt_counts[b, m] += 1
                        # fg_mask stays 1 for that anchor (still assigned to exactly one GT)
                        owner[j] = m
                        assigned = True
                        break
                    # else: try next best candidate

                # If we couldn't assign any candidate (extremely rare), fall back to best global
                # anchor by overlap for this GT, even if it wasn't in original candidates.
                if not assigned:
                    j = int(torch.argmax(overlaps[b, m]).item())
                    cur_owner = owner[j].item()
                    if cur_owner == -1:
                        mask_pos[b, m, j] = 1.0
                        per_gt_counts[b, m] += 1
                        fg_mask[b, j] += 1
                        owner[j] = m
                    elif cur_owner != m and per_gt_counts[b, cur_owner] > 1:
                        mask_pos[b, cur_owner, j] = 0.0
                        per_gt_counts[b, cur_owner] -= 1
                        mask_pos[b, m, j] = 1.0
                        per_gt_counts[b, m] += 1
                        owner[j] = m

        # Recompute fg_mask after potential rebalancing
        fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w*d)
        return target_gt_idx, fg_mask, mask_pos


def make_anchors(
    feats: list[torch.Tensor],
    strides: torch.Tensor,
    grid_cell_offset: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    anchor_points, stride_tensor = [], []
    if feats is None:
        raise ValueError
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        d, h, w = (
            feats[i].shape[2:]
            if isinstance(feats, list)
            else (int(feats[i][0]), int(feats[i][1]), int(feats[i][2]))
        )
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sz = torch.arange(end=d, device=device, dtype=dtype) + grid_cell_offset
        sz, sy, sx = torch.meshgrid(sz, sy, sx, indexing="ij")

        # this creates a tensor of shape (d*h*w, 3) which represents each coordinate triple
        anchor_points.append(torch.stack((sx, sy, sz), -1).view(-1, 3))

        stride_tensor.append(torch.full((d * h * w, 1), stride.item(), dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(
    distance: torch.Tensor, anchor_points: torch.Tensor, xyzwhd: bool = True, dim: int = -1
) -> torch.Tensor:
    """Transform distance(ltrb) to box(xyzwhd or xyzxyz)."""
    lt, rb = distance.chunk(2, dim)
    x1y1z1 = anchor_points - lt
    x2y2z2 = anchor_points + rb
    x_min_y_min_z_min = torch.minimum(x1y1z1, x2y2z2)
    x_max_y_max_z_max = torch.maximum(x1y1z1, x2y2z2)
    if xyzwhd:
        c_xyz = (x_min_y_min_z_min + x_max_y_max_z_max) / 2
        whd = x_max_y_max_z_max - x_min_y_min_z_min
        return torch.cat((c_xyz, whd), dim)  # xyzwhd bbox
    return torch.cat((x_min_y_min_z_min, x_max_y_max_z_max), dim)  # xyzxyz bbox


def bbox2dist(
    anchor_points: torch.Tensor,
    bbox: torch.Tensor,
    reg_max: float | None = None,
    reg_min: float = 0.0,
) -> torch.Tensor:
    """Transform bbox(xyzxyz) to dist(ltrb)."""
    x1y1z1, x2y2z2 = bbox.chunk(2, -1)
    dist = torch.cat((anchor_points - x1y1z1, x2y2z2 - anchor_points), -1)
    if reg_max is None:
        return dist
    return dist.clamp_(reg_min, reg_max - 0.01)  # dist (lt, rb)
