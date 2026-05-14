from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TaskAlignedEpochWarningStats:
    total_gt_objects: int = 0
    alignment_zero_fallback_objects: int = 0
    boxes_without_anchor: int = 0
    target_assignment_fallback_objects: int = 0


@dataclass(slots=True)
class MaskEpochWarningStats:
    total_objects: int = 0
    empty_after_downsampling: int = 0
    without_assigned_grid_cells: int = 0


@dataclass(slots=True)
class EpochWarningStats:
    tal: TaskAlignedEpochWarningStats = field(default_factory=TaskAlignedEpochWarningStats)
    mask: MaskEpochWarningStats = field(default_factory=MaskEpochWarningStats)

    def as_list(self) -> list[int]:
        return [
            self.tal.total_gt_objects,
            self.tal.alignment_zero_fallback_objects,
            self.tal.boxes_without_anchor,
            self.tal.target_assignment_fallback_objects,
            self.mask.total_objects,
            self.mask.empty_after_downsampling,
            self.mask.without_assigned_grid_cells,
        ]

    @classmethod
    def from_list(cls, values: list[int]) -> EpochWarningStats:
        if len(values) != 7:
            msg = f"Expected 7 warning-stat values, got {len(values)}."
            raise ValueError(msg)

        return cls(
            tal=TaskAlignedEpochWarningStats(
                total_gt_objects=values[0],
                alignment_zero_fallback_objects=values[1],
                boxes_without_anchor=values[2],
                target_assignment_fallback_objects=values[3],
            ),
            mask=MaskEpochWarningStats(
                total_objects=values[4],
                empty_after_downsampling=values[5],
                without_assigned_grid_cells=values[6],
            ),
        )


def format_epoch_warning_summary(stats: EpochWarningStats) -> str | None:
    warning_parts: list[str] = []

    if stats.tal.alignment_zero_fallback_objects:
        warning_parts.append(
            "TaskAlignedAssinger fallback for "
            f"{stats.tal.alignment_zero_fallback_objects}/{stats.tal.total_gt_objects} "
            "reference objects because they had 0 alignment with all predictions"
        )
    if stats.tal.target_assignment_fallback_objects:
        warning_parts.append(
            "TaskAlignedAssinger fallback for "
            f"{stats.tal.target_assignment_fallback_objects}/{stats.tal.total_gt_objects} "
            "reference objects becaue they had no assigned anchor "
            "after assignment conflict resolution"
        )
    if stats.tal.boxes_without_anchor:
        warning_parts.append(
            "no anchors found for "
            f"{stats.tal.boxes_without_anchor}/{stats.tal.total_gt_objects} "
            "reference objects"
        )
    if stats.mask.empty_after_downsampling:
        warning_parts.append(
            "empty references for "
            f"{stats.mask.empty_after_downsampling}/{stats.mask.total_objects} masks "
            "which can happen because the segmentation acts on half the input image resolution "
            "or due to data augmentation"
        )
    if stats.mask.without_assigned_grid_cells:
        warning_parts.append(
            f"{stats.mask.without_assigned_grid_cells}/{stats.mask.total_objects} masks "
            "were skipped because they had no assinged grid cell"
        )

    if not warning_parts:
        return None

    warning_parts.append(
        "TaskAlignedAssinger fallbacks or empty masks can be caused by small or overlapping "
        "reference objects - the model can still learn, but increasing the input resolution "
        "might help to improve it"
    )

    return "; ".join(warning_parts) + "."
