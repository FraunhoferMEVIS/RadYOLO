# Ultralytics YOLO 🚀, AGPL-3.0 license

from __future__ import annotations

import logging

import torch
from torch import nn

from yolo11_3d.ultralytics.utils.metrics import bbox_iou, bbox_iou_3d

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
    """

    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-9,
    ) -> None:
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

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
                torch.zeros_like(pd_scores[..., 0]),
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
        if mask_gt.sum(2).gt(mask_pos.sum(2)).any().item():
            LOGGER.warning(
                "Did not find anchor points for all ground-truth bounding boxes. Your bounding boxes could be too small for the configured heads."
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
        anchors_in_gts__mask = self.select_candidates_in_gts(anchor_points, ground_truth_bboxes)

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

        # Merge all mask to a final mask, (b, max_num_obj, num_anchors)
        mask_pos = mask_topk * anchors_in_gts__mask * ground_truth__mask

        return mask_pos, align_metric, overlaps

    def get_box_metrics(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        ground_truth_labels: torch.Tensor,
        ground_truth_bboxes: torch.Tensor,
        ground_truth__mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute alignment metric given predicted and ground truth bounding boxes.

        The metric is the product of 'predicted score' and 'IoU with ground truth bboxes'.

        The predicted score is weighted using 0 < `self.alpha` < 1. Larger means smaller.

        The IoU is weighted using 1 <= `self.beta`. Larger means smaller.
        """
        num_anchors = pred_bboxes.shape[-2]
        ground_truth__mask = ground_truth__mask.bool()  # b, max_num_obj, num_anchors

        # init results
        overlaps = torch.zeros(
            [self.bs, self.n_max_boxes, num_anchors],
            dtype=pred_bboxes.dtype,
            device=pred_bboxes.device,
        )
        bbox_scores = torch.zeros(
            [self.bs, self.n_max_boxes, num_anchors],
            dtype=pred_scores.dtype,
            device=pred_scores.device,
        )

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = (
            torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        )  # b, max_num_obj
        ind[1] = ground_truth_labels.squeeze(-1)  # b, max_num_obj

        # Get the scores of each grid for each gt cls
        #
        # Indexing here: ind[0] and ind[1] are iterated together, so
        # the value at x[i, j, :] == pred_scores[ind[0, i, j], :, ind[1, i, j]],
        # e.g. x[1, 2, :] == pred_scores[ind[0, 1, 2], :, ind[1, 1, 2]]
        # this is the special case of 'advanced indexing combined with simple indexing'.
        # It results in the arrays broadcasted together (the shapes are already identical)
        # and their shape prepended to the result. Then all simple indexing operations' shapes
        # are appended, if pred_scores (2, 2352, 2), the result would be (2, 2, 2352).
        #
        # So this is essentially a more sophisticated reshape to make it compatible with
        # `ground_truth__mask` and `bbox_scores`.
        #
        # As a result, `pred_scores`` is expanded to associate the class scores for each anchor
        # point with ground truth bounding boxes (which are padded). The `ground_truth__mask` is
        # used to filter for all anchor points that were matched with ground truth bounding boxes.
        #
        # `bbox_scores` finally contains all scores for all anchor points that were matched with
        # ground truth bounding boxes.
        bbox_scores[ground_truth__mask] = pred_scores[ind[0], :, ind[1]][
            ground_truth__mask
        ]  # b, max_num_obj, num_anchors

        # Here we want to calculate the overlaps between predicted bounding boxes and ground truth
        # bounding boxes. To do this we need to align the shapes of predicted bounding boxes which
        # are made for all anchor points with ground truth bounding boxes which are padded to
        # `max_num_obj`. The goal is to expand both tensor to shape (b, max_num_obj, num_anchors, 4)
        # `pred_bboxes` has shape (b, num_anchors, 4). Using `unsqueeze`, we insert an axis
        # resulting in shape (b, 1, num_anchors, 4). Using `expand`, receive shape
        # (b, max_num_obj, num_anchors, 4)
        # `ground_truth_bboxes` based on shape (b, max_num_obj, 4) is expanded analogous.
        #
        # Both tensors are indexed using `ground_truth__mask` to reduce to all anchor points that
        # were matched with ground truth bounding boxes.
        #
        # (b, max_num_obj, 1, 4), (b, 1, num_anchors, 4)
        pred_boxes = pred_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[
            ground_truth__mask
        ]
        ground_truth_boxes = ground_truth_bboxes.unsqueeze(2).expand(-1, -1, num_anchors, -1)[
            ground_truth__mask
        ]

        overlaps[ground_truth__mask] = self.iou_calculation(ground_truth_boxes, pred_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes: torch.Tensor, pd_bboxes: torch.Tensor) -> torch.Tensor:
        """IoU calculation for horizontal bounding boxes."""
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(
        self, metrics: torch.Tensor, largest: bool = True, topk_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Select the top-k candidates based on the given metrics.

        Args:
            metrics (Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size,
                              max_num_obj is the maximum number of objects, and h*w represents the
                              total number of anchor points.
            largest (bool): If True, select the largest values; otherwise, select the smallest values.
            topk_mask (Tensor): An optional boolean tensor of shape (b, max_num_obj, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on the given metrics.

        Returns:
            (Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
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
        # count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8, device=topk_idxs.device))
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
        """Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                - target_labels (Tensor): Shape (b, h*w), containing the target labels for
                                          positive anchor points.
                - target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                                          for positive anchor points.
                - target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                                          for positive anchor points, where num_classes is the number
                                          of object classes.
        """
        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(
        xy_centers: torch.Tensor, gt_bboxes: torch.Tensor, eps: float = 1e-9
    ) -> torch.Tensor:
        """Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
            eps (float, optional): Small value for numerical stability. Defaults to 1e-9.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Note:
            b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            Bounding box format: [x_min, y_min, x_max, y_max].
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(
            bs, n_boxes, n_anchors, -1
        )
        # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
        return bbox_deltas.amin(3).gt_(eps)

    @staticmethod
    def select_highest_overlaps(
        mask_pos: torch.Tensor, overlaps: torch.Tensor, n_max_boxes: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select anchor boxes with highest IoU when assigned to multiple ground truths.

        Args:
            mask_pos (torch.Tensor): Positive mask, shape (b, n_max_boxes, h*w).
            overlaps (torch.Tensor): IoU overlaps, shape (b, n_max_boxes, h*w).
            n_max_boxes (int): Maximum number of ground truth boxes.

        Returns:
            target_gt_idx (torch.Tensor): Indices of assigned ground truths, shape (b, h*w).
            fg_mask (torch.Tensor): Foreground mask, shape (b, h*w).
            mask_pos (torch.Tensor): Updated positive mask, shape (b, n_max_boxes, h*w).

        Note:
            b: batch size, h: height, w: width.
        """
        # Convert (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(
                -1, n_max_boxes, -1
            )  # (b, n_max_boxes, h*w)
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

            is_max_overlaps = torch.zeros(
                mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device
            )
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(
                mask_multi_gts, is_max_overlaps, mask_pos
            ).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos


class TaskAlignedAssigner_3D(TaskAlignedAssigner):
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
    """

    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-9,
    ) -> None:
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

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
        ind[1] = ground_truth_labels.squeeze(-1)  # b, max_num_obj
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
        return bbox_iou_3d(gt_bboxes, pd_bboxes, xyzwhd=False, CIoU=True).squeeze(-1).clamp_(0)

    @staticmethod
    def select_candidates_in_gts(
        xy_centers: torch.Tensor, gt_bboxes: torch.Tensor, eps: float = 1e-9
    ) -> torch.Tensor:
        """Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w*d, 3).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 6).
            eps (float, optional): Small value for numerical stability. Defaults to 1e-9.

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
        # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
        return bbox_deltas.amin(3).gt_(eps)


def make_anchors(
    feats: list[torch.Tensor], strides: torch.Tensor, grid_cell_offset: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = (
            feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        )
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride.item(), dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def make_anchors_3d(
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
    distance: torch.Tensor, anchor_points: torch.Tensor, xywh: bool = True, dim: int = -1
) -> torch.Tensor:
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: float) -> torch.Tensor:
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(
        0, reg_max - 0.01
    )  # dist (lt, rb)
