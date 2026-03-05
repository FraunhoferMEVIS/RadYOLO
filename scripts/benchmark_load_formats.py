from __future__ import annotations

import argparse
import csv
import gc
import logging
import math
import os
import shutil
import statistics
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    import blosc2  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    blosc2 = None


LOGGER = logging.getLogger("benchmark_load_formats")


@dataclass
class BenchmarkResult:
    image_type: str
    format_name: str
    file_path: Path
    file_size_mb: float
    repeats: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure and compare loading times for .nii.gz, .nii, .npy, and blosc2 array files "
            "using synthetic 3D volumes."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "load_benchmark",
        help="Directory where benchmark files and CSV results are written.",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        default=(192, 192, 192),
        metavar=("D", "H", "W"),
        help="Volume shape (default: 192 192 192).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64", "int16", "uint16"],
        help="Data type for generated arrays.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Number of timed load iterations per file.",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=3,
        help="Number of warm-up loads before timing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible random noise.",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="loading_benchmark_results.csv",
        help="CSV filename written under --output-dir.",
    )
    return parser.parse_args()


def build_random_noise(
    shape: tuple[int, int, int],
    rng: np.random.Generator,
    dtype: np.dtype,
) -> np.ndarray:
    return rng.standard_normal(size=shape).astype(dtype, copy=False)


def build_simple_pattern(
    shape: tuple[int, int, int],
    dtype: np.dtype,
    rng: np.random.Generator,
) -> np.ndarray:
    d, h, w = shape
    z = np.linspace(-1.0, 1.0, d, dtype=np.float32)[:, None, None]
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[None, :, None]
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, None, :]
    z_full = np.broadcast_to(z, shape)
    y_full = np.broadcast_to(y, shape)
    x_full = np.broadcast_to(x, shape)

    radial = np.sqrt(x_full**2 + y_full**2 + 0.5 * z_full**2)
    head_mask = radial <= 0.95
    brain_mask = (x_full**2 + (y_full / 0.9) ** 2 + (z_full / 0.8) ** 2) <= 0.70
    ventricle_mask = (
        (x_full / 0.12) ** 2 + (y_full / 0.25) ** 2 + (z_full / 0.18) ** 2
    ) <= 1.0

    gyri_pattern = np.sin(18.0 * x_full) * np.cos(15.0 * y_full) * np.cos(8.0 * z_full)
    bias_field = 0.25 * x_full - 0.15 * y_full + 0.2 * z_full
    vessel_pattern = np.sin(35.0 * x_full + 0.4) * np.sin(28.0 * y_full - 0.2)
    vessel_pattern *= np.cos(24.0 * z_full)

    image = np.zeros(shape, dtype=np.float32)
    image[head_mask] = -0.15
    image[brain_mask] = 0.35
    image[brain_mask] += 0.12 * gyri_pattern[brain_mask]
    image[brain_mask] += 0.08 * vessel_pattern[brain_mask]
    image[ventricle_mask] = -0.35

    ring = head_mask & (~brain_mask)
    image[ring] = 0.08 + 0.05 * np.sin(20.0 * z_full[ring])

    image[head_mask] += bias_field[head_mask]
    image[head_mask] += rng.normal(0.0, 0.03, size=int(head_mask.sum())).astype(np.float32)

    image = np.clip(image, -1.0, 1.0)
    image -= float(image.min())
    max_val = float(image.max())
    if max_val > 0:
        image /= max_val

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        scaled = image * info.max
        return np.rint(scaled).astype(dtype, copy=False)

    return image.astype(dtype, copy=False)


def build_constant(shape: tuple[int, int, int], dtype: np.dtype, value: float = 7) -> np.ndarray:
    return np.full(shape, value, dtype=dtype)


def write_nii(path: Path, array: np.ndarray) -> None:
    image = nib.Nifti1Image(array, affine=np.eye(4, dtype=np.float32))
    nib.save(image, str(path))


def write_npy(path: Path, array: np.ndarray) -> None:
    np.save(path, array)


def comp_blosc2_params(
    image_size: tuple[int, int, int, int],
    patch_size: tuple[int, int, int] | tuple[int, int],
    bytes_per_pixel: int = 4,
    l1_cache_size_per_core_in_bytes: int = 393216//6,
    l3_cache_size_per_core_in_bytes: int = 1048576//6,
    safety_factor: float = 0.8,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    num_channels = image_size[0]
    if len(patch_size) == 2:
        patch_size = (1, *patch_size)
    patch_size_np = np.array(patch_size)
    block_size = np.array(
        (num_channels, *[2 ** (max(0, math.ceil(math.log2(i)))) for i in patch_size_np])
    )

    estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel
    while estimated_nbytes_block > (l1_cache_size_per_core_in_bytes * safety_factor):
        axis_order = np.argsort(block_size[1:] / patch_size_np)[::-1]
        idx = 0
        picked_axis = axis_order[idx]
        while block_size[picked_axis + 1] == 1:
            idx += 1
            picked_axis = axis_order[idx]
        block_size[picked_axis + 1] = 2 ** (
            max(0, math.floor(math.log2(block_size[picked_axis + 1] - 1)))
        )
        block_size[picked_axis + 1] = min(block_size[picked_axis + 1], image_size[picked_axis + 1])
        estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel

    block_size = np.array([min(i, j) for i, j in zip(image_size, block_size, strict=False)])

    chunk_size = deepcopy(block_size)
    estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
    while estimated_nbytes_chunk < (l3_cache_size_per_core_in_bytes * safety_factor):
        if patch_size_np[0] == 1 and all(
            i == j for i, j in zip(chunk_size[2:], image_size[2:], strict=False)
        ):
            break
        if all(i == j for i, j in zip(chunk_size, image_size, strict=False)):
            break
        axis_order = np.argsort(chunk_size[1:] / block_size[1:])
        idx = 0
        picked_axis = axis_order[idx]
        while (
            chunk_size[picked_axis + 1] == image_size[picked_axis + 1]
            or patch_size_np[picked_axis] == 1
        ):
            idx += 1
            picked_axis = axis_order[idx]
        chunk_size[picked_axis + 1] += block_size[picked_axis + 1]
        chunk_size[picked_axis + 1] = min(chunk_size[picked_axis + 1], image_size[picked_axis + 1])
        estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
        if np.mean([i / j for i, j in zip(chunk_size[1:], patch_size_np, strict=False)]) > 1.5:
            chunk_size[picked_axis + 1] -= block_size[picked_axis + 1]
            break

    chunk_size = [min(i, j) for i, j in zip(image_size, chunk_size, strict=False)]
    return tuple(block_size), tuple(chunk_size)


def write_blosc2(path: Path, array: np.ndarray) -> None:
    if blosc2 is None:
        msg = "blosc2 is not installed. Install with: uv pip install blosc2"
        raise RuntimeError(msg)
    blosc2.set_nthreads(1)
    array_with_channel = np.ascontiguousarray(array[None, ...])
    image_size = tuple(array_with_channel.shape)
    block_size, chunk_size = comp_blosc2_params(
        image_size=image_size,
        patch_size=array.shape,
        bytes_per_pixel=array.itemsize,
    )
    cparams = {
        "codec": blosc2.Codec.ZSTD,
        "clevel": 8,
    }
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    blosc2.asarray(
        array_with_channel,
        urlpath=str(path),
        chunks=chunk_size,
        blocks=block_size,
        cparams=cparams,
    )


def load_nii(path: Path) -> np.ndarray:
    image = nib.load(str(path))
    return np.asanyarray(image.dataobj).copy()


def load_npy(path: Path) -> np.ndarray:
    return np.load(path)


def load_blosc2(path: Path) -> np.ndarray:
    if blosc2 is None:
        msg = "blosc2 is not installed. Install with: uv pip install blosc2"
        raise RuntimeError(msg)
    blosc2.set_nthreads(1)
    dparams = {"nthreads": 1}
    mmap_kwargs = {} if os.name == "nt" else {"mmap_mode": "r"}
    array_obj = blosc2.open(urlpath=str(path), mode="r", dparams=dparams, **mmap_kwargs)
    array = np.asarray(array_obj)
    if array.ndim == 4 and array.shape[0] == 1:
        return array[0]
    return array


def benchmark_load(
    load_fn: Callable[[Path], np.ndarray],
    path: Path,
    repeats: int,
    warmups: int,
) -> tuple[float, float, float, float]:
    for _ in range(warmups):
        _ = load_fn(path)

    durations_ms: list[float] = []
    checksum = 0.0

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            t0 = time.perf_counter()
            arr = load_fn(path)
            t1 = time.perf_counter()
            checksum += float(arr.reshape(-1)[0])
            durations_ms.append((t1 - t0) * 1000)
    finally:
        if gc_was_enabled:
            gc.enable()

    mean_ms = statistics.fmean(durations_ms)
    std_ms = statistics.stdev(durations_ms) if len(durations_ms) > 1 else 0.0
    return mean_ms, std_ms, min(durations_ms), max(durations_ms)


def print_results(results: list[BenchmarkResult]) -> None:
    header = (
        f"{'image_type':<16} {'format':<8} {'size_mb':>9} {'mean_ms':>10} "
        f"{'std_ms':>10} {'min_ms':>10} {'max_ms':>10}"
    )
    LOGGER.info(header)
    LOGGER.info("-" * len(header))
    for row in results:
        LOGGER.info(
            f"{row.image_type:<16} {row.format_name:<8} {row.file_size_mb:>9.2f} "
            f"{row.mean_ms:>10.2f} {row.std_ms:>10.2f} {row.min_ms:>10.2f} {row.max_ms:>10.2f}"
        )


def write_csv(results: list[BenchmarkResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "image_type",
                "format",
                "file_path",
                "file_size_mb",
                "repeats",
                "mean_ms",
                "std_ms",
                "min_ms",
                "max_ms",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row.image_type,
                    row.format_name,
                    str(row.file_path),
                    f"{row.file_size_mb:.6f}",
                    row.repeats,
                    f"{row.mean_ms:.6f}",
                    f"{row.std_ms:.6f}",
                    f"{row.min_ms:.6f}",
                    f"{row.max_ms:.6f}",
                ]
            )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    shape = tuple(args.shape)
    dtype = np.dtype(args.dtype)
    rng = np.random.default_rng(args.seed)

    volumes: dict[str, np.ndarray] = {
        "random_noise": build_random_noise(shape, rng, dtype),
        "simple_pattern": build_simple_pattern(shape, dtype, rng),
        "constant": build_constant(shape, dtype),
    }

    results: list[BenchmarkResult] = []

    for image_type, array in volumes.items():
        image_dir = output_dir / image_type
        image_dir.mkdir(parents=True, exist_ok=True)

        nii_gz_path = image_dir / f"{image_type}.nii.gz"
        nii_path = image_dir / f"{image_type}.nii"
        npy_path = image_dir / f"{image_type}.npy"

        write_nii(nii_gz_path, array)
        write_nii(nii_path, array)
        write_npy(npy_path, array)

        load_targets: list[tuple[str, Path, Callable[[Path], np.ndarray]]] = [
            ("nii.gz", nii_gz_path, load_nii),
            ("nii", nii_path, load_nii),
            ("npy", npy_path, load_npy),
        ]

        if blosc2 is not None:
            blosc2_path = image_dir / f"{image_type}.b2nd"
            write_blosc2(blosc2_path, array)
            load_targets.append(("blosc2", blosc2_path, load_blosc2))

        for format_name, path, load_fn in load_targets:
            mean_ms, std_ms, min_ms, max_ms = benchmark_load(
                load_fn=load_fn,
                path=path,
                repeats=args.repeats,
                warmups=args.warmups,
            )
            results.append(
                BenchmarkResult(
                    image_type=image_type,
                    format_name=format_name,
                    file_path=path,
                    file_size_mb=path.stat().st_size / (1024**2),
                    repeats=args.repeats,
                    mean_ms=mean_ms,
                    std_ms=std_ms,
                    min_ms=min_ms,
                    max_ms=max_ms,
                )
            )

    print_results(results)
    csv_path = output_dir / args.csv_name
    write_csv(results, csv_path)
    LOGGER.info("\nCSV written to: %s", csv_path)

    if blosc2 is None:
        LOGGER.info("blosc2 results skipped because package is not installed.")
        LOGGER.info("Install with: uv pip install blosc2")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
