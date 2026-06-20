from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
from nibabel.filebasedimages import ImageFileError
from tqdm.auto import tqdm

from radyolo.data.convert_to_nndetection import _labels_for_case
from radyolo.data.mask_compositing import (
    compose_labelmap_from_boxes as _compose_integer_encoded_from_boxes,
)
from radyolo.data.mask_compositing import (
    compose_labelmap_from_masks as _compose_integer_encoded_from_masks,
)

if TYPE_CHECKING:
    from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _find_case_ids(images_dir: Path) -> list[str]:
    if not images_dir.exists() or not images_dir.is_dir():
        msg = f"Images directory not found: {images_dir}"
        raise FileNotFoundError(msg)

    case_ids: list[str] = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file():
            continue
        name = image_path.name
        if name.endswith(".nii.gz"):
            case_ids.append(name[:-7])
        elif name.endswith(".nii"):
            case_ids.append(name[:-4])

    if not case_ids:
        msg = f"No .nii or .nii.gz files found in {images_dir}"
        raise RuntimeError(msg)
    return case_ids


def _try_load_masks(masks_dir: Path, case_id: str) -> np.ndarray | None:
    mask_path = masks_dir / f"{case_id}.nii.gz"
    if not mask_path.exists():
        return None
    try:
        nii = nib.load(mask_path)
        return np.asanyarray(nii.dataobj)
    except (FileNotFoundError, ImageFileError, OSError) as exc:
        LOGGER.warning("Failed to load mask %s: %s", mask_path, exc)
        return None


def _write_integer_encoded_mask(
    output_dir: Path,
    case_id: str,
    labelmap: np.ndarray,
    ref_nii: nib.Nifti1Image,
) -> None:
    _ensure_dir(output_dir)
    out_path = output_dir / f"{case_id}.nii.gz"
    out_img = nib.Nifti1Image(labelmap.astype(np.uint16, copy=False), affine=ref_nii.affine)
    out_img.set_data_dtype(np.uint16)
    nib.save(out_img, out_path)


def convert_dataset_to_integer_encoded_masks(dataset_dir: Path) -> Path:
    """Create integer encoded masks under masks_integer_encoded/ for a RadYOLO dataset."""
    dataset_dir = dataset_dir.resolve()
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    masks_dir = dataset_dir / "masks"
    output_dir = dataset_dir / "masks_integer_encoded"
    _ensure_dir(output_dir)

    case_ids = _find_case_ids(images_dir)
    with tqdm(total=len(case_ids), desc="Converting", unit="case", smoothing=0.05) as pbar:
        for case_id in case_ids:
            image_path = None
            for suffix in (".nii.gz", ".nii"):
                candidate = images_dir / f"{case_id}{suffix}"
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is None:
                LOGGER.warning("Skipping case %s because the image file is missing.", case_id)
                continue

            ref_nii = nib.load(image_path)
            labels_tuples = _labels_for_case(labels_dir, case_id)
            ref_shape = tuple(np.asanyarray(ref_nii.dataobj).shape[:3])

            if not labels_tuples:
                _write_integer_encoded_mask(
                    output_dir,
                    case_id,
                    np.zeros(ref_shape, dtype=np.uint16),
                    ref_nii,
                )
                pbar.update(1)
                continue

            mask_arr = _try_load_masks(masks_dir, case_id)

            labelmap: np.ndarray | None = None
            if mask_arr is not None:
                labelmap = _compose_integer_encoded_from_masks(mask_arr, labels_tuples)
                if labelmap is not None and labelmap.shape != ref_shape:
                    LOGGER.warning(
                        "Mask shape %s does not match image shape %s for case %s; "
                        "falling back to boxes.",
                        labelmap.shape,
                        ref_shape,
                        case_id,
                    )
                    labelmap = None

            if labelmap is None:
                labelmap = _compose_integer_encoded_from_boxes(ref_shape, labels_tuples)

            _write_integer_encoded_mask(output_dir, case_id, labelmap, ref_nii)
            pbar.update(1)

    return output_dir

