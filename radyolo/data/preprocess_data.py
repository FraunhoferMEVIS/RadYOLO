from __future__ import annotations

import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import blosc2
import nibabel as nib
import numpy as np
import torch
from tqdm.auto import tqdm

from radyolo.config.spatial import PatchSize, VoxelSize
from radyolo.data.data_splits import DEFAULT_SPLIT_FILE_NAME, get_split_file_path
from radyolo.data.preprocessing import (
    ImageArrayDType,
    preprocess_image,
    resample_mask,
    resolve_numpy_dtype,
    resolve_torch_dtype,
)

if TYPE_CHECKING:
    from radyolo.data.normalization import NormalizationConfig

LOGGER = logging.getLogger(__name__)

DEFAULT_L1_CACHE_SIZE_PER_CORE_IN_BYTES = 32768
DEFAULT_L3_CACHE_SIZE_PER_CORE_IN_BYTES = 1441792
DEFAULT_CACHE_SAFETY_FACTOR = 0.8

type NormalizationSpec = NormalizationConfig | list[NormalizationConfig]


@dataclass(frozen=True)
class PreprocessCaseConfig:
    """Immutable preprocessing settings passed to each case conversion."""

    target_image_size: PatchSize | None
    target_voxel_size: VoxelSize | None
    normalization: NormalizationSpec
    image_dtype: ImageArrayDType


def _iter_cases(images_dir: Path) -> list[tuple[str, Path]]:
    """Return sorted case ids and image paths from an images directory."""
    return [
        (path.name[:-7], path)
        for path in sorted(images_dir.glob("*.nii.gz"))
        if path.name.endswith(".nii.gz")
    ]


def _copy_file_if_exists(src: Path, dst: Path) -> None:
    """Copy a file if it exists."""
    if src.exists():
        shutil.copy2(src, dst)


def _copy_tree_if_exists(src: Path, dst: Path) -> None:
    """Copy a directory tree if it exists."""
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)


def _load_nifti_tensor(
    nifti_path: Path, *, squeeze: bool = False
) -> tuple[torch.Tensor, VoxelSize]:
    """Load a NIfTI image as a tensor in zyx order plus voxel size."""
    nifti_image = nib.loadsave.load(nifti_path)
    if not isinstance(nifti_image, nib.nifti1.Nifti1Image):
        msg = f"Unsupported NIfTI image type for case at {nifti_path}"
        raise TypeError(msg)

    image = np.asanyarray(nifti_image.dataobj)
    if squeeze:
        image = np.squeeze(image)
    image_tensor = torch.from_numpy(image)
    image_tensor = image_tensor.permute(*reversed(range(image_tensor.ndim)))
    voxel_size = VoxelSize.from_xyz(nifti_image.header.get_zooms()[:3])
    return image_tensor, voxel_size


def _tensor_to_saved_numpy(image: torch.Tensor, original_ndim: int) -> np.ndarray:
    """Convert an internal tensor back to contiguous numpy storage layout."""
    image_to_store = image.squeeze(0) if original_ndim == 3 and image.shape[0] == 1 else image
    return np.ascontiguousarray(
        image_to_store.permute(*reversed(range(image_to_store.ndim))).cpu().numpy()
    )


def _remove_path_if_exists(path: Path) -> None:
    """Remove a file or directory path if present."""
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _remove_case_outputs(out_dir: Path, case_id: str, suffixes: tuple[str, ...]) -> None:
    """Delete any existing output artifacts for a case and suffix set."""
    for suffix in suffixes:
        _remove_path_if_exists(out_dir / f"{case_id}{suffix}")


def _estimate_array_nbytes(shape: tuple[int, ...], itemsize: int) -> int:
    """Estimate array memory usage in bytes for a shape and itemsize."""
    return int(np.prod(shape, dtype=np.int64)) * itemsize


def _previous_power_of_two(value: int) -> int:
    """Return the greatest power of two less than or equal to value."""
    if value <= 1:
        return 1
    return 1 << ((value - 1).bit_length() - 1)


def _compute_block_shape(
    image_shape: tuple[int, ...],
    dtype: np.dtype,
    l1_cache_size_per_core_in_bytes: int = DEFAULT_L1_CACHE_SIZE_PER_CORE_IN_BYTES,
    safety_factor: float = DEFAULT_CACHE_SAFETY_FACTOR,
) -> tuple[int, ...]:
    """Choose a block shape sized to fit within the target L1 cache budget."""
    if any(dim <= 0 for dim in image_shape):
        msg = f"Array shape must contain only positive integers, got {image_shape}."
        raise ValueError(msg)

    itemsize = np.dtype(dtype).itemsize
    target_nbytes = max(1, int(l1_cache_size_per_core_in_bytes * safety_factor))
    block_shape = list(image_shape)

    while _estimate_array_nbytes(tuple(block_shape), itemsize) > target_nbytes:
        axis_order = sorted(
            range(len(block_shape)),
            key=lambda axis: (
                block_shape[axis] / image_shape[axis],
                block_shape[axis],
            ),
            reverse=True,
        )
        for axis in axis_order:
            if block_shape[axis] <= 1:
                continue
            block_shape[axis] = _previous_power_of_two(block_shape[axis])
            break
        else:
            break

    return tuple(block_shape)


def _compute_chunk_shape(
    image_shape: tuple[int, ...],
    block_shape: tuple[int, ...],
    dtype: np.dtype,
    l3_cache_size_per_core_in_bytes: int = DEFAULT_L3_CACHE_SIZE_PER_CORE_IN_BYTES,
    safety_factor: float = DEFAULT_CACHE_SAFETY_FACTOR,
) -> tuple[int, ...]:
    """Grow chunk shape from block shape within the target L3 cache budget."""
    itemsize = np.dtype(dtype).itemsize
    target_nbytes = max(1, int(l3_cache_size_per_core_in_bytes * safety_factor))
    chunk_shape = list(block_shape)

    while tuple(chunk_shape) != image_shape:
        axis_order = sorted(
            range(len(chunk_shape)),
            key=lambda axis: (
                chunk_shape[axis] / block_shape[axis],
                chunk_shape[axis],
            ),
        )
        for axis in axis_order:
            if chunk_shape[axis] >= image_shape[axis]:
                continue
            candidate = chunk_shape.copy()
            candidate[axis] = min(candidate[axis] + block_shape[axis], image_shape[axis])
            if _estimate_array_nbytes(tuple(candidate), itemsize) <= target_nbytes:
                chunk_shape = candidate
                break
        else:
            break

    return tuple(chunk_shape)


def _compute_storage_shapes(
    image_shape: tuple[int, ...],
    dtype: np.dtype,
    l1_cache_size_per_core_in_bytes: int = DEFAULT_L1_CACHE_SIZE_PER_CORE_IN_BYTES,
    l3_cache_size_per_core_in_bytes: int = DEFAULT_L3_CACHE_SIZE_PER_CORE_IN_BYTES,
    safety_factor: float = DEFAULT_CACHE_SAFETY_FACTOR,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Compute Blosc2 block and chunk shapes for a given array shape and dtype."""
    block_shape = _compute_block_shape(
        image_shape,
        dtype,
        l1_cache_size_per_core_in_bytes=l1_cache_size_per_core_in_bytes,
        safety_factor=safety_factor,
    )
    chunk_shape = _compute_chunk_shape(
        image_shape,
        block_shape,
        dtype,
        l3_cache_size_per_core_in_bytes=l3_cache_size_per_core_in_bytes,
        safety_factor=safety_factor,
    )
    return block_shape, chunk_shape


def _save_preprocessed_blosc2_array(
    array: np.ndarray,
    output_path: Path,
    *,
    use_full_image_chunk: bool,
) -> None:
    """Persist a numpy array as a compressed Blosc2 ndarray on disk."""
    array_shape = tuple(int(dim) for dim in array.shape)
    array_dtype = np.dtype(array.dtype)

    _remove_path_if_exists(output_path)
    blosc2.set_nthreads(1)
    if use_full_image_chunk:
        normalized_chunk_shape = array_shape
        normalized_block_shape = _compute_block_shape(array_shape, array_dtype)
    else:
        normalized_block_shape, normalized_chunk_shape = _compute_storage_shapes(
            array_shape,
            array_dtype,
        )
    cparams = {"codec": blosc2.Codec.ZSTD, "clevel": 8, "nthreads": 1}
    blosc2.asarray(
        np.ascontiguousarray(array),
        urlpath=str(output_path),
        chunks=normalized_chunk_shape,
        blocks=normalized_block_shape,
        cparams=cparams,
    )


def _save_preprocessed_image(
    image: np.ndarray,
    out_images_dir: Path,
    case_id: str,
    *,
    use_full_image_chunk: bool,
) -> None:
    """Write one preprocessed image case to the output images directory."""
    _remove_case_outputs(out_images_dir, case_id, (".npy", ".b2nd"))

    _save_preprocessed_blosc2_array(
        image,
        out_images_dir / f"{case_id}.b2nd",
        use_full_image_chunk=use_full_image_chunk,
    )


def _save_preprocessed_mask(
    mask: np.ndarray,
    out_masks_dir: Path,
    case_id: str,
    *,
    use_full_image_chunk: bool,
) -> None:
    """Write one preprocessed mask case to the output masks directory."""
    _remove_case_outputs(out_masks_dir, case_id, (".npz", ".b2nd"))

    _save_preprocessed_blosc2_array(
        mask,
        out_masks_dir / f"{case_id}.b2nd",
        use_full_image_chunk=use_full_image_chunk,
    )


def _has_preprocessed_image_outputs(out_images_dir: Path) -> bool:
    """Return whether preprocessed image files already exist in a directory."""
    return any(out_images_dir.glob("*.npy")) or any(out_images_dir.glob("*.b2nd"))


def _validate_image_output_options(
    target_image_size: PatchSize | None,
    target_voxel_size: VoxelSize | None,
) -> None:
    """Ensure mutually exclusive image-size and voxel-size options."""
    if target_image_size is not None and target_voxel_size is not None:
        msg = "Only one of target_image_size or target_voxel_size can be specified."
        raise ValueError(msg)


def _validate_existing_outputs(
    out_images_dir: Path,
    out_masks_dir: Path | None,
    overwrite: bool,
) -> None:
    """Validate output directories against overwrite behavior."""
    if (
        out_images_dir.exists()
        and _has_preprocessed_image_outputs(out_images_dir)
        and not overwrite
    ):
        msg = (
            "Output images directory already contains preprocessed image files at "
            f"{out_images_dir}. Use overwrite=True to allow replacing files."
        )
        raise FileExistsError(msg)
    if (
        out_masks_dir
        and out_masks_dir.exists()
        and (any(out_masks_dir.glob("*.npz")) or any(out_masks_dir.glob("*.b2nd")))
        and not overwrite
    ):
        msg = (
            f"Output masks directory already contains preprocessed mask files at {out_masks_dir}. "
            "Use overwrite=True to allow replacing files."
        )
        raise FileExistsError(msg)


def _convert_case_to_blosc2(
    case_id: str,
    image_path: Path,
    masks_dir: Path,
    out_images_dir: Path,
    out_masks_dir: Path | None,
    config: PreprocessCaseConfig,
) -> None:
    """Preprocess one case and store image and optional mask as Blosc2 arrays."""
    image_tensor, voxel_size = _load_nifti_tensor(image_path)
    original_ndim = image_tensor.ndim
    image_tensor, _, resized_shape = preprocess_image(
        image_tensor,
        config.normalization,
        target_shape=config.target_image_size,
        input_voxel_size=voxel_size,
        target_voxel_size=config.target_voxel_size,
        output_dtype=resolve_torch_dtype(config.image_dtype),
    )
    _save_preprocessed_image(
        _tensor_to_saved_numpy(image_tensor, original_ndim).astype(
            resolve_numpy_dtype(config.image_dtype),
            copy=False,
        ),
        out_images_dir,
        case_id,
        use_full_image_chunk=config.target_image_size is not None,
    )

    if out_masks_dir is None:
        return

    mask_path = masks_dir / f"{case_id}.nii.gz"
    if mask_path.exists():
        mask_tensor, _ = _load_nifti_tensor(mask_path, squeeze=True)
        mask_resampled = resample_mask(
            mask_tensor.to(dtype=torch.uint8),
            PatchSize.from_zyx(resized_shape),
        )
        _save_preprocessed_mask(
            _tensor_to_saved_numpy(mask_resampled, mask_tensor.ndim).astype(
                np.uint8,
                copy=False,
            ),
            out_masks_dir,
            case_id,
            use_full_image_chunk=config.target_image_size is not None,
        )


def _preprocess_cases_serially(
    cases: list[tuple[str, Path]],
    masks_dir: Path,
    out_images_dir: Path,
    out_masks_dir: Path | None,
    case_config: PreprocessCaseConfig,
) -> None:
    """Preprocess all cases sequentially with progress reporting."""
    progress_bar = tqdm(total=len(cases), desc="Preprocessing", unit="case", smoothing=0.05)
    try:
        for case_id, image_path in cases:
            try:
                _convert_case_to_blosc2(
                    case_id,
                    image_path,
                    masks_dir,
                    out_images_dir,
                    out_masks_dir,
                    case_config,
                )
            except Exception as exc:
                msg = f"Failed to preprocess case '{case_id}'"
                raise RuntimeError(msg) from exc
            progress_bar.update(1)
    finally:
        progress_bar.close()


def _preprocess_cases_in_process_pool(
    cases: list[tuple[str, Path]],
    masks_dir: Path,
    out_images_dir: Path,
    out_masks_dir: Path | None,
    case_config: PreprocessCaseConfig,
    max_workers: int,
) -> None:
    """Preprocess all cases in a process pool with per-case error wrapping."""
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _convert_case_to_blosc2,
                case_id,
                image_path,
                masks_dir,
                out_images_dir,
                out_masks_dir,
                case_config,
            ): case_id
            for case_id, image_path in cases
        }
        progress_bar = tqdm(total=len(futures), desc="Preprocessing", unit="case", smoothing=0.05)
        try:
            for future in as_completed(futures):
                case_id = futures[future]
                try:
                    future.result()
                except BrokenProcessPool:
                    raise
                except Exception as exc:
                    msg = f"Failed to preprocess case '{case_id}'"
                    raise RuntimeError(msg) from exc
                progress_bar.update(1)
        finally:
            progress_bar.close()


def _preprocess_cases(
    cases: list[tuple[str, Path]],
    masks_dir: Path,
    out_images_dir: Path,
    out_masks_dir: Path | None,
    case_config: PreprocessCaseConfig,
    max_workers: int,
) -> None:
    """Dispatch preprocessing serially or in parallel with pool fallback."""
    if max_workers <= 1:
        _preprocess_cases_serially(
            cases,
            masks_dir,
            out_images_dir,
            out_masks_dir,
            case_config,
        )
        return

    try:
        _preprocess_cases_in_process_pool(
            cases,
            masks_dir,
            out_images_dir,
            out_masks_dir,
            case_config,
            max_workers,
        )
    except BrokenProcessPool:
        LOGGER.warning("Process pool preprocessing failed; retrying serially.")
        _preprocess_cases_serially(
            cases,
            masks_dir,
            out_images_dir,
            out_masks_dir,
            case_config,
        )


def preprocess_dataset(  # noqa: PLR0913
    dataset_path: Path,
    output_path: Path,
    workers: int = 0,
    overwrite: bool = False,
    target_image_size: PatchSize | None = None,
    target_voxel_size: VoxelSize | None = None,
    normalization: NormalizationSpec | None = None,
    image_dtype: ImageArrayDType = "float32",
    split_file_name: str = DEFAULT_SPLIT_FILE_NAME,
) -> Path:
    """Convert NIfTI images to array stores in a mirrored dataset folder structure.

    The output dataset keeps labels unchanged and stores images and optional masks as
    `.b2nd` arrays.
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    normalization = {"type": "none"} if normalization is None else normalization

    _validate_image_output_options(
        target_image_size=target_image_size,
        target_voxel_size=target_voxel_size,
    )

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
    _validate_existing_outputs(
        out_images_dir=out_images_dir,
        out_masks_dir=out_masks_dir,
        overwrite=overwrite,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    out_images_dir.mkdir(parents=True, exist_ok=True)
    if out_masks_dir:
        out_masks_dir.mkdir(parents=True, exist_ok=True)

    shutil.copytree(labels_dir, output_path / "labels", dirs_exist_ok=True)
    _copy_tree_if_exists(dataset_path / "labels_image", output_path / "labels_image")
    _copy_tree_if_exists(dataset_path / "inputs_tabular", output_path / "inputs_tabular")

    _copy_file_if_exists(dataset_path / "dataset.yaml", output_path / "dataset.yaml")
    _copy_file_if_exists(
        get_split_file_path(dataset_path, split_file_name),
        get_split_file_path(output_path, split_file_name),
    )

    max_workers = os.cpu_count() or 1
    if workers > 0:
        max_workers = workers

    case_config = PreprocessCaseConfig(
        target_image_size=target_image_size,
        target_voxel_size=target_voxel_size,
        normalization=normalization,
        image_dtype=image_dtype,
    )

    LOGGER.info(
        "Preprocessing %d cases from %s to %s using %d workers",
        len(cases),
        dataset_path,
        output_path,
        max_workers,
    )

    _preprocess_cases(
        cases,
        masks_dir,
        out_images_dir,
        out_masks_dir,
        case_config,
        max_workers,
    )

    return output_path
