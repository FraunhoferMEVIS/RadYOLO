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

if TYPE_CHECKING:
    from collections.abc import Iterable

LOGGER = logging.getLogger(__name__)


class ImageStats(NamedTuple):
    # Per-channel statistics for an image, within FOREGROUND only.
    # Contains: min,max,sum,sumsq,count (no per-image percentiles anymore)
    per_channel: list[dict[str, float]]
    # image size in voxels (z, y, x)
    shape: tuple[int, ...]
    # voxel spacing in mm (z, y, x)
    spacing: tuple[float, float, float]
    # foreground intensity samples per channel (1D float arrays, up to sample_cap per image)
    samples_per_channel: list[np.ndarray]


class LabelStats(NamedTuple):
    n_by_class: dict[str, int]
    # sizes per class in mm for each axis [z, y, x]
    sizes_by_class_mm: dict[str, list[tuple[float, float, float]]]


@dataclass
class DatasetFingerprint:
    # Per-channel stats aggregated
    foreground_channel_stats: list[dict[str, float]]
    median_image_size: list[int]
    median_voxel_size_mm: list[float]
    # Box stats per class
    boxes_per_class: dict[str, int]
    box_size_median_mm: dict[str, list[float]]
    box_size_p10_mm: dict[str, list[float]]
    # Masks
    n_masks_available: int


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

    # Extract foreground intensities
    fg = arr[foreground_mask].astype(np.float32, copy=False)
    n = int(fg.size)
    # Stats
    s = float(fg.sum(dtype=np.float64))
    ss = float(np.square(fg, dtype=np.float64).sum(dtype=np.float64))
    foreground_stats = {
        "min": float(np.min(fg)),
        "max": float(np.max(fg)),
        "sum": s,
        "sumsq": ss,
        "count": float(n),
    }

    # Sampling for global percentiles
    if n <= sample_cap:
        samples = fg
    else:
        # uniform random sample without replacement
        idx = np.random.default_rng().choice(n, size=sample_cap, replace=False)
        samples = fg[idx]
    return foreground_stats, samples


def _load_nifti(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.loadsave.load(str(path))
    if not isinstance(img, nib.nifti1.Nifti1Image):
        msg = f"Unsupported NIfTI at {path}"
        raise TypeError(msg)
    data = np.asanyarray(img.dataobj)
    # spacing: nib loads affine, voxel sizes are diagonal of Zooms or from header
    zooms = img.header.get_zooms()
    # Expect at least (x,y,z)
    spatial_dims = 3
    if len(zooms) >= spatial_dims:
        spacing_xyz = (float(zooms[2]), float(zooms[1]), float(zooms[0]))  # to (z,y,x)
    else:
        LOGGER.warning(f"Could not extract spacing from nifti header in file {path}")
        spacing_xyz = (1.0, 1.0, 1.0)
    return data, spacing_xyz


def _labels_to_mask(
    entries: list[tuple[int, float, float, float, float, float, float]],
    shape_xyz: tuple[int, int, int],
) -> np.ndarray:
    """Create a boolean union mask of all boxes on a grid of shape (X, Y, Z).

    Label format: class x y z w h d (normalized to [0,1] relative to X,Y,Z).
    """
    x_shape, y_shape, z_shape = shape_xyz
    mask = np.zeros((x_shape, y_shape, z_shape), dtype=bool)
    if not entries:
        return mask
    for _cls, x, y, z, w, h, d in entries:
        # Convert normalized center/size to voxel indices (half-open slices)
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

        # clamp to bounds and skip empty
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
    num_only_spatial_dims = 3
    num_spatial_and_channel_dims = 4
    heuristic_max_channel_number = 8
    data, spacing_zyx = _load_nifti(image_path)

    # Build per-channel arrays in (X, Y, Z) order
    if data.ndim == num_only_spatial_dims:
        data_per_channel = [data]  # (X, Y, Z)
        shape_zyx = (data.shape[2], data.shape[1], data.shape[0])  # (z,y,x)
        shape_xyz = (data.shape[0], data.shape[1], data.shape[2])  # (X,Y,Z)
    elif data.ndim == num_spatial_and_channel_dims:
        # either (X,Y,Z,C) or (C,X,Y,Z); heuristic: if first dim small (<=8) treat as channels-first
        if data.shape[0] <= heuristic_max_channel_number and data.shape[0] < data.shape[-1]:
            # (C,X,Y,Z)
            data_per_channel = [data[c, ...] for c in range(data.shape[0])]
            shape_zyx = (data.shape[3], data.shape[2], data.shape[1])  # (z,y,x)
            shape_xyz = (data.shape[1], data.shape[2], data.shape[3])  # (X,Y,Z)
        else:
            # (X,Y,Z,C)
            data_per_channel = [data[..., c] for c in range(data.shape[-1])]
            shape_zyx = (data.shape[2], data.shape[1], data.shape[0])
            shape_xyz = (data.shape[0], data.shape[1], data.shape[2])
    else:
        msg = f"Unsupported image ndim {data.ndim} at {image_path}"
        raise ValueError(msg)

    # Foreground mask from labels (in XYZ order)
    entries = _labels_for_case(labels_path)
    mask = _labels_to_mask(entries, shape_xyz) if entries else np.zeros(shape_xyz, dtype=bool)

    per_channel_stats: list[dict[str, float]] = []
    samples_per_channel: list[np.ndarray] = []
    for channel in data_per_channel:
        stats, samples = _foreground_channel_stats_and_samples(channel, mask, sample_cap=sample_cap)
        per_channel_stats.append(stats)
        samples_per_channel.append(samples)

    return ImageStats(
        per_channel=per_channel_stats,
        shape=shape_zyx,
        spacing=spacing_zyx,
        samples_per_channel=samples_per_channel,
    )


def _parse_label_line(
    line: str,
) -> tuple[int, float, float, float, float, float, float]:
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
    if not labels_path.exists():
        return []
    txt = labels_path.read_text(encoding="utf-8").strip().splitlines()
    entries: list[tuple[int, float, float, float, float, float, float]] = []
    for line in txt:
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
    image_shape_zyx: tuple[int, int, int],
    spacing_zyx: tuple[float, float, float],
) -> LabelStats:
    entries = _labels_for_case(labels_file)
    n_by_class: dict[str, int] = {}
    sizes_by_class_mm: dict[str, list[tuple[float, float, float]]] = {}
    if not entries:
        return LabelStats(n_by_class, sizes_by_class_mm)

    shape_z, shape_y, shape_x = image_shape_zyx
    image_size = np.array([shape_z, shape_y, shape_x], dtype=np.float64)
    image_spacing = np.array(spacing_zyx, dtype=np.float64)  # (z,y,x) mm

    for cls, _x, _y, _z, w, h, d in entries:
        cls_key = str(cls)
        # labels are normalized in [0,1], sizes relative to image size; convert to mm
        whd_norm = np.array([w, h, d], dtype=np.float64)
        size_vox = whd_norm * image_size
        size_mm = size_vox * image_spacing
        tup = (float(size_mm[0]), float(size_mm[1]), float(size_mm[2]))
        sizes_by_class_mm.setdefault(cls_key, []).append(tup)
        n_by_class[cls_key] = n_by_class.get(cls_key, 0) + 1

    return LabelStats(n_by_class=n_by_class, sizes_by_class_mm=sizes_by_class_mm)


def _has_mask(mask_path: Path) -> bool:
    return mask_path.exists()


def _iter_cases(dataset_path: Path) -> Iterable[tuple[str, Path]]:
    images_dir = dataset_path / "images"
    if not images_dir.exists():
        msg = f"Missing images directory at {images_dir}"
        raise FileNotFoundError(msg)
    for p in sorted(images_dir.glob("*.nii.gz")):
        name = p.name
        # case id = filename without .nii.gz
        case_id = name[:-7] if name.endswith(".nii.gz") else p.stem
        yield case_id, p


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
    label_stats_results: list[LabelStats] = []
    n_masks = 0

    total_cases = len(cases)
    LOGGER.info("Starting dataset analysis: %d cases, %d workers", total_cases, max_workers)
    # Run image and label/mask computations in parallel per case to minimize IO bottlenecks
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
                try:
                    label_stats = _label_stats_for_case(
                        label_path, image_stats.shape[:3], image_stats.spacing
                    )
                    label_stats_results.append(label_stats)
                except Exception:
                    LOGGER.exception("Failed to read labels for %s", label_path)

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

    # Prepare accumulators (foreground-only)
    aggregated = [
        {
            "min": np.inf,
            "max": -np.inf,
            "sum": 0.0,
            "sumsq": 0.0,
            "count": 0.0,
            "samples": [],  # list of np.ndarray
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

        # Collect global percentile samples per channel
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
        # Global percentiles from pooled foreground samples
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

    # Median image size and voxel size
    shapes = np.array([s.shape[:3] for s in image_stats_results], dtype=np.int64)  # (N,3)
    spacings = np.array([s.spacing for s in image_stats_results], dtype=np.float64)  # (N,3)
    median_shape = np.median(shapes, axis=0).astype(int).tolist()
    median_spacing = np.median(spacings, axis=0).astype(float).tolist()

    # Aggregate label stats
    total_by_class: dict[str, int] = {}
    all_sizes_by_class: dict[str, list[tuple[float, float, float]]] = {}
    for label_stats in label_stats_results:
        for cls, count in label_stats.n_by_class.items():
            total_by_class[cls] = total_by_class.get(cls, 0) + count
        for cls, lst in label_stats.sizes_by_class_mm.items():
            all_sizes_by_class.setdefault(cls, []).extend(lst)

    # Compute medians and 10th percentiles per class per axis
    box_size_median_mm: dict[str, list[float]] = {}
    box_size_p10_mm: dict[str, list[float]] = {}
    for cls, lst in all_sizes_by_class.items():
        arr = np.array(lst, dtype=np.float64)  # (M,3) z,y,x in mm
        if arr.size == 0:
            continue
        median = np.median(arr, axis=0)
        p10 = np.percentile(arr, 10, axis=0)
        box_size_median_mm[cls] = [float(median[0]), float(median[1]), float(median[2])]
        box_size_p10_mm[cls] = [float(p10[0]), float(p10[1]), float(p10[2])]

    finger_print = DatasetFingerprint(
        foreground_channel_stats=foreground_channel_stats,
        median_image_size=median_shape,
        median_voxel_size_mm=median_spacing,
        boxes_per_class=total_by_class,
        box_size_median_mm=box_size_median_mm,
        box_size_p10_mm=box_size_p10_mm,
        n_masks_available=n_masks,
    )

    out_path = dataset_path / "dataset_fingerprint.json"
    out_path.write_text(json.dumps(finger_print.__dict__, indent=2), encoding="utf-8")
    return out_path
