from __future__ import annotations

import contextlib
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import nibabel as nib
import numpy as np
from tqdm.auto import tqdm

from radyolo.config.spatial import PatchSize, VoxelSize

if TYPE_CHECKING:
    from collections.abc import Iterable

LOGGER = logging.getLogger(__name__)


class ImageStats(NamedTuple):
    """Per-image foreground statistics and sampled intensities for each channel."""

    per_channel: list[dict[str, float]]
    image_size: PatchSize
    voxel_size: VoxelSize
    has_objects: bool
    samples_per_channel: list[np.ndarray]


class LabelStats(NamedTuple):
    """Per-class object counts and box sizes on the resampled image grid."""

    n_by_class: dict[str, int]
    sizes_by_class_vox: dict[str, list[VoxelSize]]


@dataclass
class DatasetFingerprint:
    """Aggregated dataset statistics written to dataset_fingerprint.json."""

    foreground_channel_stats: list[dict[str, float]]
    median_image_size: PatchSize
    median_voxel_size_mm: VoxelSize
    voxel_size_p10_mm: VoxelSize
    voxel_size_p90_mm: VoxelSize
    median_resampled_imagesize: PatchSize
    resampled_imagesize_p10_vox: PatchSize
    resampled_imagesize_p90_vox: PatchSize
    boxes_per_class: dict[str, int]
    box_size_median_vox: dict[str, VoxelSize]
    box_size_p95_vox: VoxelSize | None
    n_images_with_masks: int
    n_images_with_objects: int
    n_images_without_objects: int


def _serialize_dataset_fingerprint(fingerprint: DatasetFingerprint) -> dict[str, object]:
    return {
        "foreground_channel_stats": fingerprint.foreground_channel_stats,
        "median_image_size": list(fingerprint.median_image_size.get_xyz()),
        "median_voxel_size_mm": list(fingerprint.median_voxel_size_mm.get_xyz()),
        "voxel_size_p10_mm": list(fingerprint.voxel_size_p10_mm.get_xyz()),
        "voxel_size_p90_mm": list(fingerprint.voxel_size_p90_mm.get_xyz()),
        "median_resampled_imagesize": list(fingerprint.median_resampled_imagesize.get_xyz()),
        "resampled_imagesize_p10_vox": list(fingerprint.resampled_imagesize_p10_vox.get_xyz()),
        "resampled_imagesize_p90_vox": list(fingerprint.resampled_imagesize_p90_vox.get_xyz()),
        "boxes_per_class": fingerprint.boxes_per_class,
        "box_size_median_vox": {
            cls: list(size.get_xyz()) for cls, size in fingerprint.box_size_median_vox.items()
        },
        "box_size_p95_vox": (
            list(fingerprint.box_size_p95_vox.get_xyz())
            if fingerprint.box_size_p95_vox is not None
            else None
        ),
        "n_images_with_masks": fingerprint.n_images_with_masks,
        "n_images_with_objects": fingerprint.n_images_with_objects,
        "n_images_without_objects": fingerprint.n_images_without_objects,
    }


def _foreground_channel_stats_and_samples(
    arr: np.ndarray, foreground_mask: np.ndarray, sample_cap: int = 10_000
) -> tuple[dict[str, float], np.ndarray]:
    """Compute stats and foreground samples for one channel.

    Args:
        arr: channel array in (X, Y, Z) order, any numeric dtype.
        foreground_mask: boolean array (X, Y, Z), True for foreground voxels.
        sample_cap: max number of intensities to sample from foreground for percentiles.

    Returns:
        (stats dict, samples 1D float32 array)
    """
    if foreground_mask is None or not np.any(foreground_mask):
        return {"min": np.nan, "max": np.nan, "sum": 0.0, "sumsq": 0.0, "count": 0.0}, np.array(
            [], dtype=np.float32
        )

    fg = arr[foreground_mask].astype(np.float32, copy=False)
    n = int(fg.size)
    s = float(fg.sum(dtype=np.float64))
    ss = float(np.square(fg, dtype=np.float64).sum(dtype=np.float64))
    foreground_stats = {
        "min": float(np.min(fg)),
        "max": float(np.max(fg)),
        "sum": s,
        "sumsq": ss,
        "count": float(n),
    }

    if n <= sample_cap:
        samples = fg
    else:
        idx = np.random.default_rng().choice(n, size=sample_cap, replace=False)
        samples = fg[idx]
    return foreground_stats, samples


def _load_nifti(path: Path) -> tuple[np.ndarray, VoxelSize]:
    """Load a NIfTI image and return its array data with spatial voxel size."""

    img = nib.loadsave.load(str(path))
    if not isinstance(img, nib.nifti1.Nifti1Image):
        msg = f"Unsupported NIfTI at {path}"
        raise TypeError(msg)
    data = np.asanyarray(img.dataobj)
    zooms = img.header.get_zooms()
    spatial_dims = 3
    if len(zooms) >= spatial_dims:
        voxel_size = VoxelSize.from_xyz(zooms[:3])
    else:
        LOGGER.warning(f"Could not extract spacing from nifti header in file {path}")
        voxel_size = VoxelSize(x=1.0, y=1.0, z=1.0)
    return data, voxel_size


def _labels_to_mask(
    entries: list[tuple[int, float, float, float, float, float, float]],
    image_size: PatchSize,
) -> np.ndarray:
    """Create a boolean union mask of all boxes on a grid of shape (X, Y, Z).

    Label format: class x y z w h d (normalized to [0,1] relative to X,Y,Z).
    """
    x_shape, y_shape, z_shape = image_size.get_xyz()
    mask = np.zeros((x_shape, y_shape, z_shape), dtype=bool)
    if not entries:
        return mask
    for _cls, x, y, z, w, h, d in entries:
        cx = x * x_shape
        cy = y * y_shape
        cz = z * z_shape
        wx = max(w * x_shape, 0.0)
        hy = max(h * y_shape, 0.0)
        dz = max(d * z_shape, 0.0)

        x0 = int(np.floor(cx - wx / 2.0))
        x1 = int(np.ceil(cx + wx / 2.0))
        y0 = int(np.floor(cy - hy / 2.0))
        y1 = int(np.ceil(cy + hy / 2.0))
        z0 = int(np.floor(cz - dz / 2.0))
        z1 = int(np.ceil(cz + dz / 2.0))

        x0 = max(0, min(x_shape, x0))
        x1 = max(0, min(x_shape, x1))
        y0 = max(0, min(y_shape, y0))
        y1 = max(0, min(y_shape, y1))
        z0 = max(0, min(z_shape, z0))
        z1 = max(0, min(z_shape, z1))
        if x1 <= x0 or y1 <= y0 or z1 <= z0:
            continue
        mask[x0:x1, y0:y1, z0:z1] = True
    return mask


def _image_stats(image_path: Path, labels_path: Path, sample_cap: int = 10_000) -> ImageStats:
    """Compute per-channel foreground statistics for a single image."""

    num_only_spatial_dims = 3
    num_spatial_and_channel_dims = 4
    heuristic_max_channel_number = 8
    data, voxel_size = _load_nifti(image_path)

    if data.ndim == num_only_spatial_dims:
        data_per_channel = [data]
        image_size = PatchSize.from_xyz((data.shape[0], data.shape[1], data.shape[2]))
    elif data.ndim == num_spatial_and_channel_dims:
        if data.shape[0] <= heuristic_max_channel_number and data.shape[0] < data.shape[-1]:
            data_per_channel = [data[c, ...] for c in range(data.shape[0])]
            image_size = PatchSize.from_xyz((data.shape[1], data.shape[2], data.shape[3]))
        else:
            data_per_channel = [data[..., c] for c in range(data.shape[-1])]
            image_size = PatchSize.from_xyz((data.shape[0], data.shape[1], data.shape[2]))
    else:
        msg = f"Unsupported image ndim {data.ndim} at {image_path}"
        raise ValueError(msg)

    entries = _labels_for_case(labels_path)
    mask = (
        _labels_to_mask(entries, image_size)
        if entries
        else np.zeros(image_size.get_xyz(), dtype=bool)
    )

    per_channel_stats: list[dict[str, float]] = []
    samples_per_channel: list[np.ndarray] = []
    for channel in data_per_channel:
        stats, samples = _foreground_channel_stats_and_samples(channel, mask, sample_cap=sample_cap)
        per_channel_stats.append(stats)
        samples_per_channel.append(samples)

    return ImageStats(
        per_channel=per_channel_stats,
        image_size=image_size,
        voxel_size=voxel_size,
        has_objects=bool(entries),
        samples_per_channel=samples_per_channel,
    )


def _parse_label_line(
    line: str,
) -> tuple[int, float, float, float, float, float, float]:
    """Parse one YOLO-style 3D box annotation line."""

    expected_line_parts = 7
    parts = line.strip().split()
    if len(parts) != expected_line_parts:
        msg = (
            "Label line must have 7 entries: class x y z w h d (normalized), "
            f"but has {len(parts)} entries."
        )
        raise ValueError(msg)
    cls = int(float(parts[0]))
    x, y, z, w, h, d = map(float, parts[1:])
    return cls, x, y, z, w, h, d


def _labels_for_case(
    labels_path: Path,
) -> list[tuple[int, float, float, float, float, float, float]]:
    """Load all non-empty label entries for one case."""

    if not labels_path.exists():
        return []
    label_lines = labels_path.read_text(encoding="utf-8").strip().splitlines()
    entries: list[tuple[int, float, float, float, float, float, float]] = []
    for line in label_lines:
        if not line.strip():
            continue
        cls, x, y, z, w, h, d = _parse_label_line(line)
        if np.isclose(w, 0.0) or np.isclose(h, 0.0) or np.isclose(d, 0.0):
            LOGGER.warning(
                f"Zero-extent bounding box found in {labels_path}: "
                f"class={cls} extents=(w={w:.6f}, h={h:.6f}, d={d:.6f})"
            )
        entries.append((cls, x, y, z, w, h, d))
    return entries


def _label_stats_for_case(
    labels_file: Path,
    image_size: PatchSize,
    voxel_size: VoxelSize,
    target_voxel_size: VoxelSize,
) -> LabelStats:
    """Compute per-class box counts and sizes for one label file."""

    entries = _labels_for_case(labels_file)
    n_by_class: dict[str, int] = {}
    sizes_by_class_vox: dict[str, list[VoxelSize]] = {}
    if not entries:
        return LabelStats(n_by_class, sizes_by_class_vox)

    resampled_image_size = np.array(
        _resolve_resampled_image_size(
            image_size,
            voxel_size,
            target_voxel_size,
        ).get_xyz(),
        dtype=np.float64,
    )

    for cls, _x, _y, _z, w, h, d in entries:
        cls_key = str(cls)
        xyz_norm = np.array([w, h, d], dtype=np.float64)
        size_vox = xyz_norm * resampled_image_size
        sizes_by_class_vox.setdefault(cls_key, []).append(
            VoxelSize(x=float(size_vox[0]), y=float(size_vox[1]), z=float(size_vox[2]))
        )
        n_by_class[cls_key] = n_by_class.get(cls_key, 0) + 1

    return LabelStats(n_by_class=n_by_class, sizes_by_class_vox=sizes_by_class_vox)


def _has_mask(mask_path: Path) -> bool:
    return mask_path.exists()


def _resolve_resampled_image_size(
    image_size: PatchSize,
    source_voxel_size: VoxelSize,
    target_voxel_size: VoxelSize,
) -> PatchSize:
    """Project an image size onto a target voxel spacing."""

    return PatchSize.from_xyz(
        [
            max(1, round(size * source_spacing / target_spacing))
            for size, source_spacing, target_spacing in zip(
                image_size.get_xyz(),
                source_voxel_size.get_xyz(),
                target_voxel_size.get_xyz(),
                strict=True,
            )
        ]
    )


def _patch_size_from_percentile(values_xyz: np.ndarray, percentile: float) -> PatchSize:
    return PatchSize.from_xyz(np.percentile(values_xyz, percentile, axis=0).astype(int).tolist())


def _voxel_size_from_percentile(values_xyz: np.ndarray, percentile: float) -> VoxelSize:
    return VoxelSize.from_xyz(np.percentile(values_xyz, percentile, axis=0).astype(float).tolist())


def _iter_cases(dataset_path: Path) -> Iterable[tuple[str, Path]]:
    """Yield case ids and image paths for all dataset images."""

    images_dir = dataset_path / "images"
    if not images_dir.exists():
        msg = f"Missing images directory at {images_dir}"
        raise FileNotFoundError(msg)
    for image_path in sorted(images_dir.glob("*.nii.gz")):
        image_name = image_path.name
        case_id = image_name[:-7] if image_name.endswith(".nii.gz") else image_path.stem
        yield case_id, image_path


def analyze_dataset(dataset_path: Path, workers: int = 0) -> Path:
    """Compute dataset statistics and save dataset_fingerprint.json in dataset folder.

    Expects subfolders images/ (.nii.gz), labels/ (.txt), optional masks/ (.nii.gz).
    """
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    masks_dir = dataset_path / "masks"

    if not images_dir.exists():
        msg = f"Missing images directory: {images_dir}"
        raise FileNotFoundError(msg)
    if not labels_dir.exists():
        msg = f"Missing labels directory: {labels_dir}"
        raise FileNotFoundError(msg)

    cases = list(_iter_cases(dataset_path))
    if not cases:
        msg = "No cases found under images/"
        raise RuntimeError(msg)

    max_workers = os.cpu_count() or 1
    if workers and workers > 0:
        max_workers = workers

    image_stats_results: list[ImageStats] = []
    label_stats_inputs: list[tuple[Path, PatchSize, VoxelSize]] = []
    n_masks = 0
    n_images_with_objects = 0
    n_images_without_objects = 0

    total_cases = len(cases)
    LOGGER.info("Starting dataset analysis: %d cases, %d workers", total_cases, max_workers)
    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = {}
    processed = 0
    pbar = tqdm(total=total_cases, desc="Analyzing", unit="case")
    try:
        for case_id, image_path in cases:
            label_path = labels_dir / f"{case_id}.txt"
            mask_path = masks_dir / f"{case_id}.nii.gz"
            future = executor.submit(_image_stats, image_path, label_path)
            futures[future] = (case_id, image_path, label_path, mask_path)
        for future in as_completed(futures):
            case_id, image_path, label_path, mask_path = futures[future]
            try:
                image_stats = future.result()
                image_stats_results.append(image_stats)
            except Exception:
                LOGGER.exception("Failed to read image %s", image_path)
            else:
                label_stats_inputs.append(
                    (label_path, image_stats.image_size, image_stats.voxel_size)
                )
                if image_stats.has_objects:
                    n_images_with_objects += 1
                else:
                    n_images_without_objects += 1

                if _has_mask(mask_path):
                    n_masks += 1
            processed += 1
            pbar.update(1)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user. Cancelling pending tasks...")
        for f in futures:
            f.cancel()
        with contextlib.suppress(Exception):
            executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        with contextlib.suppress(Exception):
            pbar.close()
        with contextlib.suppress(Exception):
            executor.shutdown(wait=False, cancel_futures=True)

    if not image_stats_results:
        msg = (
            "No images could be read. Check if image filenames end with .nii.gz and are accessible."
        )
        raise RuntimeError(msg)

    n_channels = max(len(stats.per_channel) for stats in image_stats_results)

    aggregated = [
        {
            "min": np.inf,
            "max": -np.inf,
            "sum": 0.0,
            "sumsq": 0.0,
            "count": 0.0,
            "samples": [],
        }
        for _ in range(n_channels)
    ]

    for stats in image_stats_results:
        for ch_idx, ch_stats in enumerate(stats.per_channel):
            count = ch_stats.get("count", 0.0)
            if count and np.isfinite(ch_stats.get("min", np.nan)):
                aggregated[ch_idx]["min"] = min(aggregated[ch_idx]["min"], ch_stats["min"])  # type: ignore[index]
                aggregated[ch_idx]["max"] = max(aggregated[ch_idx]["max"], ch_stats["max"])  # type: ignore[index]
            aggregated[ch_idx]["sum"] += ch_stats.get("sum", 0.0)  # type: ignore[index]
            aggregated[ch_idx]["sumsq"] += ch_stats.get("sumsq", 0.0)  # type: ignore[index]
            aggregated[ch_idx]["count"] += count  # type: ignore[index]

        for ch_idx, samples in enumerate(stats.samples_per_channel):
            if samples.size:
                aggregated[ch_idx]["samples"].append(samples)  # type: ignore[index]

    foreground_channel_stats: list[dict[str, float]] = []
    for channel in aggregated:
        count = channel["count"] or 0.0
        if count > 0:
            mean = channel["sum"] / count
            var = max(channel["sumsq"] / count - mean * mean, 0.0)
            std = float(np.sqrt(var))
        else:
            mean = float("nan")
            std = float("nan")
        if channel["samples"]:
            pooled = np.concatenate(channel["samples"])  # type: ignore[index]
            p005, p995 = np.percentile(pooled, [0.5, 99.5])
            p005 = float(p005)
            p995 = float(p995)
        else:
            p005 = float("nan")
            p995 = float("nan")

        ch_min = float(channel["min"]) if np.isfinite(channel["min"]) else float("nan")
        ch_max = float(channel["max"]) if np.isfinite(channel["max"]) else float("nan")

        foreground_channel_stats.append(
            {
                "min": ch_min,
                "p00_5": p005,
                "p99_5": p995,
                "mean": float(mean),
                "std": float(std),
                "max": ch_max,
            }
        )

    images_sizes = np.array(
        [stats.image_size.get_xyz() for stats in image_stats_results], dtype=np.int64
    )
    voxel_sizes = np.array(
        [stats.voxel_size.get_xyz() for stats in image_stats_results],
        dtype=np.float64,
    )
    median_image_size = PatchSize.from_xyz(np.median(images_sizes, axis=0).astype(int).tolist())
    median_voxel_size = VoxelSize.from_xyz(np.median(voxel_sizes, axis=0).astype(float).tolist())
    voxel_size_p10 = _voxel_size_from_percentile(voxel_sizes, 10)
    voxel_size_p90 = _voxel_size_from_percentile(voxel_sizes, 90)

    resampled_image_sizes = np.array(
        [
            _resolve_resampled_image_size(
                stats.image_size,
                stats.voxel_size,
                median_voxel_size,
            ).get_xyz()
            for stats in image_stats_results
        ],
        dtype=np.int64,
    )
    median_resampled_imagesize = PatchSize.from_xyz(np.median(resampled_image_sizes, axis=0).astype(int).tolist())
    resampled_imagesize_p10 = _patch_size_from_percentile(resampled_image_sizes, 10)
    resampled_imagesize_p90 = _patch_size_from_percentile(resampled_image_sizes, 90)

    label_stats_results: list[LabelStats] = []
    for label_path, image_size, voxel_size in label_stats_inputs:
        try:
            label_stats = _label_stats_for_case(
                label_path,
                image_size,
                voxel_size,
                median_voxel_size,
            )
            label_stats_results.append(label_stats)
        except Exception:
            LOGGER.exception("Failed to read labels for %s", label_path)

    total_by_class: dict[str, int] = {}
    all_sizes_by_class: dict[str, list[VoxelSize]] = {}
    for label_stats in label_stats_results:
        for cls, count in label_stats.n_by_class.items():
            total_by_class[cls] = total_by_class.get(cls, 0) + count
        for cls, lst in label_stats.sizes_by_class_vox.items():
            all_sizes_by_class.setdefault(cls, []).extend(lst)

    box_size_median_vox: dict[str, VoxelSize] = {}
    all_box_sizes: list[tuple[float, float, float]] = []
    for cls, lst in all_sizes_by_class.items():
        arr = np.array([size.get_xyz() for size in lst], dtype=np.float64)
        if arr.size == 0:
            continue
        all_box_sizes.extend([tuple(size.get_xyz()) for size in lst])
        median = np.median(arr, axis=0)
        box_size_median_vox[cls] = VoxelSize(
            x=float(median[0]),
            y=float(median[1]),
            z=float(median[2]),
        )

    box_size_p95_vox: VoxelSize | None = None
    if all_box_sizes:
        p95 = np.percentile(np.array(all_box_sizes, dtype=np.float64), 95, axis=0)
        box_size_p95_vox = VoxelSize(
            x=float(p95[0]),
            y=float(p95[1]),
            z=float(p95[2]),
        )

    finger_print = DatasetFingerprint(
        foreground_channel_stats=foreground_channel_stats,
        median_image_size=median_image_size,
        median_voxel_size_mm=median_voxel_size,
        voxel_size_p10_mm=voxel_size_p10,
        voxel_size_p90_mm=voxel_size_p90,
        median_resampled_imagesize=median_resampled_imagesize,
        resampled_imagesize_p10_vox=resampled_imagesize_p10,
        resampled_imagesize_p90_vox=resampled_imagesize_p90,
        boxes_per_class=total_by_class,
        box_size_median_vox=box_size_median_vox,
        box_size_p95_vox=box_size_p95_vox,
        n_images_with_masks=n_masks,
        n_images_with_objects=n_images_with_objects,
        n_images_without_objects=n_images_without_objects,
    )

    out_path = dataset_path / "dataset_fingerprint.json"
    out_path.write_text(
        json.dumps(_serialize_dataset_fingerprint(finger_print), indent=2), encoding="utf-8"
    )
    return out_path
