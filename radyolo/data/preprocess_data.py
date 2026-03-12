from __future__ import annotations

import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)


def _iter_cases(images_dir: Path) -> list[tuple[str, Path]]:
    return [
        (path.name[:-7], path)
        for path in sorted(images_dir.glob("*.nii.gz"))
        if path.name.endswith(".nii.gz")
    ]


def _copy_file_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def _convert_nifti_to_numpy(
    nifti_path: Path,
    out_path: Path,
    dtype: np.dtype | type | None = None,
) -> Path:
    nifti_image = nib.loadsave.load(nifti_path)
    if not isinstance(nifti_image, nib.nifti1.Nifti1Image):
        msg = f"Unsupported NIfTI image type for case at {nifti_path}"
        raise TypeError(msg)

    image = np.asanyarray(nifti_image.dataobj, dtype=dtype)
    contiguous = np.ascontiguousarray(image)
    np.save(out_path, contiguous, allow_pickle=False)
    return out_path


def _convert_case_to_numpy(
    case_id: str,
    image_path: Path,
    masks_dir: Path,
    out_images_dir: Path,
    out_masks_dir: Path | None,
) -> None:
    image_out_path = out_images_dir / f"{case_id}.npy"
    _convert_nifti_to_numpy(image_path, image_out_path)

    if out_masks_dir is None:
        return

    mask_path = masks_dir / f"{case_id}.nii.gz"
    if mask_path.exists():
        nifti_mask = nib.loadsave.load(mask_path)
        if not isinstance(nifti_mask, nib.nifti1.Nifti1Image):
            msg = f"Unsupported NIfTI image type for case at {mask_path}"
            raise TypeError(msg)
        mask_data = np.asanyarray(nifti_mask.dataobj, dtype=np.uint8)
        mask_array = np.ascontiguousarray(mask_data)
        mask_out_path = out_masks_dir / f"{case_id}.npz"
        np.savez_compressed(mask_out_path, mask=mask_array)


def preprocess_dataset(
    dataset_path: Path,
    output_path: Path,
    workers: int = 0,
    overwrite: bool = False,
) -> Path:
    """Convert NIfTI images to NumPy arrays in a mirrored dataset folder structure.

    The output dataset keeps labels unchanged and stores images as `.npy` arrays and
    optional masks as compressed `.npz` arrays.
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    masks_dir = dataset_path / "masks"

    if not images_dir.exists():
        msg = f"Missing images directory: {images_dir}"
        raise FileNotFoundError(msg)
    if not labels_dir.exists():
        msg = f"Missing labels directory: {labels_dir}"
        raise FileNotFoundError(msg)

    cases = _iter_cases(images_dir)
    if not cases:
        msg = f"No .nii.gz images found in {images_dir}"
        raise RuntimeError(msg)

    out_images_dir = output_path / "images"
    out_masks_dir = output_path / "masks" if masks_dir.exists() else None
    if out_images_dir.exists() and any(out_images_dir.glob("*.npy")) and not overwrite:
        msg = (
            f"Output images directory already contains .npy files at {out_images_dir}. "
            "Use overwrite=True to allow replacing files."
        )
        raise FileExistsError(msg)
    if (
        out_masks_dir
        and out_masks_dir.exists()
        and any(out_masks_dir.glob("*.npz"))
        and not overwrite
    ):
        msg = (
            f"Output masks directory already contains .npz files at {out_masks_dir}. "
            "Use overwrite=True to allow replacing files."
        )
        raise FileExistsError(msg)

    output_path.mkdir(parents=True, exist_ok=True)
    out_images_dir.mkdir(parents=True, exist_ok=True)
    if out_masks_dir:
        out_masks_dir.mkdir(parents=True, exist_ok=True)

    shutil.copytree(labels_dir, output_path / "labels", dirs_exist_ok=True)

    _copy_file_if_exists(dataset_path / "dataset.yaml", output_path / "dataset.yaml")
    _copy_file_if_exists(dataset_path / "data_splits.json", output_path / "data_splits.json")

    max_workers = os.cpu_count() or 1
    if workers > 0:
        max_workers = workers

    LOGGER.info(
        "Preprocessing %d cases from %s to %s using %d workers",
        len(cases),
        dataset_path,
        output_path,
        max_workers,
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _convert_case_to_numpy,
                case_id,
                image_path,
                masks_dir,
                out_images_dir,
                out_masks_dir,
            ): case_id
            for case_id, image_path in cases
        }
        progress_bar = tqdm(total=len(futures), desc="Preprocessing", unit="case")
        try:
            for future in as_completed(futures):
                case_id = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    msg = f"Failed to preprocess case '{case_id}'"
                    raise RuntimeError(msg) from exc
                progress_bar.update(1)
        finally:
            progress_bar.close()

    return output_path
