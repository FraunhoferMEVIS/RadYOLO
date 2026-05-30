from __future__ import annotations

import torch

from radyolo.training.tal import TaskAlignedAssigner
from radyolo.utils.logging_utils import (
    EpochWarningStats,
    MaskEpochWarningStats,
    TaskAlignedEpochWarningStats,
    format_epoch_warning_summary,
)


def test_format_epoch_warning_summary_lists_only_nonzero_warning_categories() -> None:
    summary = format_epoch_warning_summary(
        EpochWarningStats(
            tal=TaskAlignedEpochWarningStats(
                total_gt_objects=12,
                alignment_zero_fallback_objects=3,
                boxes_without_anchor=2,
                target_assignment_fallback_objects=1,
            ),
            mask=MaskEpochWarningStats(total_objects=5, empty_after_downsampling=2),
        )
    )

    assert summary == (
        "TaskAlignedAssinger fallback for 3/12 reference objects because they had zero "
        "alignment with all predictions; "
        "TaskAlignedAssinger fallback for 1/12 reference objects becaue they had no "
        "assigned anchor after assignment conflict resolution; "
        "no anchors found for 2/12 reference objects; "
        "empty references for 2/5 masks which can happen because the segmentation acts "
        "on half the input image resolution or due to data augmentation; "
        "TaskAlignedAssinger fallbacks or empty masks can be caused by small or "
        "overlapping reference objects - the model can still learn, but increasing the "
        "input resolution might help to improve it."
    )


def test_format_epoch_warning_summary_returns_none_when_no_warnings_occur() -> None:
    assert format_epoch_warning_summary(EpochWarningStats()) is None


def test_epoch_warning_stats_roundtrip_via_list_preserves_field_values() -> None:
    stats = EpochWarningStats(
        tal=TaskAlignedEpochWarningStats(
            total_gt_objects=12,
            alignment_zero_fallback_objects=3,
            boxes_without_anchor=2,
            target_assignment_fallback_objects=1,
        ),
        mask=MaskEpochWarningStats(
            total_objects=5,
            empty_after_downsampling=2,
            without_assigned_grid_cells=4,
        ),
    )

    assert EpochWarningStats.from_list(stats.as_list()) == stats


def test_task_aligned_assigner_pop_epoch_warning_stats_resets_counts() -> None:
    assigner = TaskAlignedAssigner()
    assigner._epoch_warning_stats.total_gt_objects = 9  # noqa: SLF001
    assigner._epoch_warning_stats.alignment_zero_fallback_objects = 2  # noqa: SLF001

    assert assigner.pop_epoch_warning_stats() == TaskAlignedEpochWarningStats(
        total_gt_objects=9,
        alignment_zero_fallback_objects=2,
    )
    assert assigner.pop_epoch_warning_stats() == TaskAlignedEpochWarningStats()


def test_select_highest_overlaps_prefers_highest_align_metric_over_iou() -> None:
    assigner = TaskAlignedAssigner()
    mask_pos = torch.tensor([[[1.0], [1.0]]], dtype=torch.float32)
    align_metric = torch.tensor([[[0.2], [0.9]]], dtype=torch.float32)
    overlaps = torch.tensor([[[0.9], [0.3]]], dtype=torch.float32)

    target_gt_idx, fg_mask, resolved_mask_pos = assigner.select_highest_alignments(
        mask_pos,
        align_metric,
        overlaps,
        n_max_boxes=2,
    )

    assert target_gt_idx.tolist() == [[1]]
    assert fg_mask.tolist() == [[1.0]]
    assert resolved_mask_pos.tolist() == [[[0.0], [1.0]]]
