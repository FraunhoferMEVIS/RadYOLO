from __future__ import annotations

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
        "TAL alignment fallback for 3/12 ground-truth objects; "
        "TAL target-assignment fallback for 1/12 ground-truth objects; "
        "no anchor points found for 2/12 ground-truth objects; "
        "empty masks after downsampling for 2/5 masks."
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
