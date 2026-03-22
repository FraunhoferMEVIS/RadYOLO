from __future__ import annotations

import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm.auto import tqdm

from radyolo.data.normalization import NormalizationConfig
from radyolo.data.preprocessing import ImageArrayDType, preprocess_image, resample_mask
from radyolo.data.preprocessing import resolve_numpy_dtype, resolve_torch_dtype

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


def _load_nifti_tensor(nifti_path: Path) -> tuple[torch.Tensor, tuple[float, float, float]]:
    nifti_image = nib.loadsave.load(nifti_path)
    if not isinstance(nifti_image, nib.nifti1.Nifti1Image):
        msg = f"Unsupported NIfTI image type for case at {nifti_path}"
        raise TypeError(msg)

    image = np.asanyarray(nifti_image.dataobj)
    image_tensor = torch.from_numpy(image)
    image_tensor = image_tensor.permute(*reversed(range(image_tensor.ndim)))
    voxel_size = tuple(float(zoom) for zoom in reversed(nifti_image.header.get_zooms()[:3]))
    return image_tensor, voxel_size


def _tensor_to_saved_numpy(image: torch.Tensor, original_ndim: int) -> np.ndarray:
    image_to_store = image.squeeze(0) if original_ndim == 3 and image.shape[0] == 1 else image
    return np.ascontiguousarray(image_to_store.permute(*reversed(range(image_to_store.ndim))).cpu().numpy())


def _convert_case_to_numpy(
    case_id: str,
    image_path: Path,
    masks_dir: Path,
    out_images_dir: Path,
    out_masks_dir: Path | None,
    target_shape: tuple[int, int, int] | None,
    target_voxel_size: tuple[float, float, float] | None,
    normalization: NormalizationConfig | list[NormalizationConfig],
    image_dtype: ImageArrayDType,
) -> None:
    image_out_path = out_images_dir / f"{case_id}.npy"
    image_tensor, voxel_size = _load_nifti_tensor(image_path)
    image_preprocessed, _, resized_shape = preprocess_image(
        image_tensor,
        normalization,
        target_shape=target_shape,
        input_voxel_size=voxel_size,
        target_voxel_size=target_voxel_size,
        output_dtype=resolve_torch_dtype(image_dtype),
    )
    np.save(
        image_out_path,
        _tensor_to_saved_numpy(image_preprocessed, image_tensor.ndim).astype(
            resolve_numpy_dtype(image_dtype),
            copy=False,
        ),
        allow_pickle=False,
    )

    if out_masks_dir is None:
        return

    mask_path = masks_dir / f"{case_id}.nii.gz"
    if mask_path.exists():
        mask_tensor, _ = _load_nifti_tensor(mask_path)
        mask_resampled = resample_mask(mask_tensor.to(dtype=torch.uint8), resized_shape)
        mask_out_path = out_masks_dir / f"{case_id}.npz"
        np.savez_compressed(
            mask_out_path,
            mask=_tensor_to_saved_numpy(mask_resampled, mask_tensor.ndim).astype(np.uint8, copy=False),
        )


def preprocess_dataset(
    dataset_path: Path,
    output_path: Path,
    workers: int = 0,
    overwrite: bool = False,
    target_shape: tuple[int, int, int] | None = None,
    target_voxel_size: tuple[float, float, float] | None = None,
    normalization: NormalizationConfig | list[NormalizationConfig] = {"type": "none"},
    image_dtype: ImageArrayDType = "float32",
) -> Path:
    """Convert NIfTI images to NumPy arrays in a mirrored dataset folder structure.

    The output dataset keeps labels unchanged and stores images as `.npy` arrays and
    optional masks as compressed `.npz` arrays.
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    if target_shape is not None and target_voxel_size is not None:
        msg = "Only one of target_shape or target_voxel_size can be specified."
        raise ValueError(msg)

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
                target_shape,
                target_voxel_size,
                normalization,
                image_dtype,
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
