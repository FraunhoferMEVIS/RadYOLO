from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from tqdm_loggable.auto import tqdm

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _MaskCandidate:
    """A mask candidate selected from one detection entry."""
    class_id: int
    mask_path: str
    box_volume: float


def _detection_box_volume(prediction: dict[str, Any]) -> float:
    """Return the detection box volume from an absolute or relative box."""
    for key in ("box_image_xyzxyz", "box_relative_xyzxyz"):
        raw_box = prediction.get(key)
        if isinstance(raw_box, list) and len(raw_box) == 6:
            try:
                x1, y1, z1, x2, y2, z2 = (float(value) for value in raw_box)
            except (TypeError, ValueError):
                return 0.0
            dx = max(0.0, x2 - x1)
            dy = max(0.0, y2 - y1)
            dz = max(0.0, z2 - z1)
            return dx * dy * dz
    return 0.0


def _resolve_mask_path(predictions_dir: Path, masks_dir: Path, raw_mask_path: str) -> Path:
    """Resolve a mask path relative to the prediction or masks directory."""
    mask_path = Path(raw_mask_path)
    if mask_path.is_absolute():
        return mask_path

    relative_candidate = predictions_dir / mask_path
    if relative_candidate.exists():
        return relative_candidate

    return masks_dir / mask_path.name


def _collect_candidates(
    prediction_entries: list[Any],
    confidence_threshold: float,
) -> tuple[list[_MaskCandidate], str | None]:
    """Collect eligible mask candidates and remember the first mask path seen."""
    candidates: list[_MaskCandidate] = []
    first_mask_path: str | None = None
    for entry in prediction_entries:
        if not isinstance(entry, dict):
            continue

        raw_mask_path = entry.get("mask_path")
        if not isinstance(raw_mask_path, str) or not raw_mask_path.strip():
            continue

        if first_mask_path is None:
            first_mask_path = raw_mask_path

        try:
            confidence = float(entry["confidence"])
            class_id = int(entry["class"])
        except (KeyError, TypeError, ValueError):
            continue

        if confidence < confidence_threshold:
            continue

        candidates.append(
            _MaskCandidate(
                class_id=class_id,
                mask_path=raw_mask_path,
                box_volume=_detection_box_volume(entry),
            )
        )

    candidates.sort(key=lambda candidate: candidate.box_volume, reverse=True)
    return candidates, first_mask_path


def merge_prediction_masks(
    predictions_dir: Path,
    confidence_threshold: float = 0.1,
) -> Path:
    """Merge per-detection masks into one labeled mask per prediction case."""
    if confidence_threshold < 0.0:
        msg = "Confidence threshold must be >= 0."
        raise ValueError(msg)

    masks_dir = predictions_dir / "masks"
    if not masks_dir.is_dir():
        msg = f"Prediction folder must contain a masks directory: {masks_dir}"
        raise FileNotFoundError(msg)

    output_dir = predictions_dir / "masks_merged"
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_paths = sorted(predictions_dir.glob("*.json"))
    for prediction_path in tqdm(
        prediction_paths,
        desc="Merging prediction masks",
        unit="case",
    ):
        case_name = prediction_path.stem
        parsed = json.loads(prediction_path.read_text(encoding="utf-8"))
        if not isinstance(parsed, list):
            msg = f"Prediction file {prediction_path} must contain a list of detections."
            raise TypeError(msg)

        candidates, fallback_mask_path = _collect_candidates(parsed, confidence_threshold)
        if not candidates and fallback_mask_path is None:
            LOGGER.info(
                "Skipping case %s because it has no mask_path entries in prediction JSON.",
                case_name,
            )
            continue

        reference_mask = candidates[0].mask_path if candidates else fallback_mask_path
        if reference_mask is None:
            continue
        reference_mask_path = _resolve_mask_path(predictions_dir, masks_dir, reference_mask)
        if not reference_mask_path.exists():
            LOGGER.warning(
                "Skipping case %s because reference mask %s does not exist.",
                case_name,
                reference_mask_path,
            )
            continue

        reference_nii = nib.load(str(reference_mask_path))
        merged_mask = np.zeros(reference_nii.shape, dtype=np.uint16)

        for candidate in candidates:
            resolved_mask_path = _resolve_mask_path(predictions_dir, masks_dir, candidate.mask_path)
            if not resolved_mask_path.exists():
                LOGGER.warning("Mask %s does not exist and is skipped.", resolved_mask_path)
                continue

            mask_nii = nib.load(str(resolved_mask_path))
            if mask_nii.shape != merged_mask.shape:
                LOGGER.warning(
                    "Mask %s has shape %s, expected %s; skipping.",
                    resolved_mask_path,
                    mask_nii.shape,
                    merged_mask.shape,
                )
                continue

            mask_array = np.asarray(mask_nii.dataobj)
            merged_mask[mask_array > 0] = np.uint16(candidate.class_id + 1)

        merged_path = output_dir / f"{case_name}.nii.gz"
        nib.save(nib.Nifti1Image(merged_mask, affine=reference_nii.affine), str(merged_path))

    return output_dir
