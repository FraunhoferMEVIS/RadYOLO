from __future__ import annotations

import argparse
import csv
import gc
import logging
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from radyolo.data.augmentations import (
    GINNet,
    batch_generators_intensity_augmentations,
    random_channel_dropout,
    random_mirroring,
    random_rotation,
    random_zoom,
    tensor_cutout,
)
from radyolo.data.normalization import normalize_image

if TYPE_CHECKING:
    from collections.abc import Callable


LOGGER = logging.getLogger("benchmark_data_loading_and_augmentations")

DEFAULT_SIZES = (64, 128, 256, 512)
DEFAULT_AUGMENTATIONS = (
    "data_loading",
    "resampling",
    "normalization",
    "gaussian_noise",
    "gaussian_blur",
    "multiplicative_brightness",
    "additive_brightness",
    "contrast",
    "low_resolution",
    "gamma",
    "gin",
    "channel_dropout",
    "random_zoom_uniform",
    "random_zoom_anisotropic",
    "random_rotation",
    "tensor_cutout",
    "random_mirroring",
)

UNSUPPORTED_AUGMENTATIONS_BY_DEVICE: dict[str, dict[str, str]] = {
    "cuda": {
        "gaussian_noise": "batchgenerators Gaussian noise mixes CPU and CUDA tensors",
        "gaussian_blur": "batchgenerators Gaussian blur keeps convolution weights on CPU",
    }
}


@dataclass
class BenchmarkResult:
    size: int
    augmentation: str
    device: str
    repeats: int
    warmups: int
    mean_ms: float | None
    std_ms: float | None
    min_ms: float | None
    max_ms: float | None
    status: str
    error: str = ""


def get_unsupported_reason(augmentation: str, device: torch.device) -> str | None:
    return UNSUPPORTED_AUGMENTATIONS_BY_DEVICE.get(device.type, {}).get(augmentation)


def make_unsupported_result(
    size: int,
    augmentation: str,
    device: torch.device,
    repeats: int,
    warmups: int,
    reason: str,
) -> BenchmarkResult:
    return BenchmarkResult(
        size=size,
        augmentation=augmentation,
        device=device.type,
        repeats=repeats,
        warmups=warmups,
        mean_ms=None,
        std_ms=None,
        min_ms=None,
        max_ms=None,
        status="unsupported",
        error=reason,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure dummy-data loading time and individual augmentation runtimes for "
            "different cubic image sizes."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "augmentation_benchmark",
        help="Directory where generated dummy files and CSV results are written.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="Cubic edge lengths to benchmark (default: 64 128 256 512).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help=(
            "Number of image channels in the dummy tensor. Defaults to 1 to keep the 512^3 "
            "case feasible on typical CPUs."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed iterations per augmentation and size.",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="Number of warm-up iterations executed before timing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible dummy inputs.",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="data_loading_and_augmentation_benchmark.csv",
        help="CSV filename written below --output-dir.",
    )
    parser.add_argument(
        "--keep-generated-files",
        action="store_true",
        help="Keep the generated dummy .npy volumes after benchmarking finishes.",
    )
    parser.add_argument(
        "--augmentations",
        type=str,
        nargs="+",
        choices=list(DEFAULT_AUGMENTATIONS),
        default=list(DEFAULT_AUGMENTATIONS),
        help="Subset of augmentations to benchmark.",
    )
    parser.add_argument(
        "--normalization-type",
        type=str,
        choices=["none", "zscore", "ct", "255_to_01"],
        default="zscore",
        help="Normalization scheme used for the normalization benchmark row.",
    )
    parser.add_argument(
        "--normalization-mean",
        type=float,
        default=0.0,
        help="Mean used by ct normalization.",
    )
    parser.add_argument(
        "--normalization-std",
        type=float,
        default=1.0,
        help="Standard deviation used by ct normalization.",
    )
    parser.add_argument(
        "--normalization-clip-min",
        type=float,
        default=-1024.0,
        help="Minimum clip value used by ct normalization.",
    )
    parser.add_argument(
        "--normalization-clip-max",
        type=float,
        default=3071.0,
        help="Maximum clip value used by ct normalization.",
    )
    parser.add_argument(
        "--resampling-source-scale",
        type=float,
        default=1.25,
        help="Cubic edge-length factor used to build the source volume for resampling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Execution device for supported operations (default: cpu).",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            msg = "--device cuda was requested, but CUDA is not available"
            raise RuntimeError(msg)
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_normalization_config(args: argparse.Namespace) -> dict[str, float | str]:
    if args.normalization_type != "ct":
        return {"type": args.normalization_type}

    return {
        "type": "ct",
        "mean": args.normalization_mean,
        "std": args.normalization_std,
        "clip_min": args.normalization_clip_min,
        "clip_max": args.normalization_clip_max,
    }


def build_resampling_source_shape(size: int, scale: float) -> tuple[int, int, int]:
    if scale <= 0:
        msg = "--resampling-source-scale must be > 0"
        raise ValueError(msg)
    source_edge = max(2, round(size * scale))
    return (source_edge, source_edge, source_edge)


def build_dummy_volume(
    shape: tuple[int, int, int],
    channels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    base = rng.random((channels, *shape), dtype=np.float32)
    z_gradient = np.linspace(0.0, 1.0, shape[0], dtype=np.float32).reshape(1, shape[0], 1, 1)
    base += z_gradient
    base /= float(base.max())
    return base


def build_dummy_labels(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    labels_norm_xyzwhd = torch.tensor(
        [
            [0.0, 0.25, 0.25, 0.25, 0.18, 0.16, 0.14],
            [1.0, 0.55, 0.55, 0.50, 0.22, 0.20, 0.24],
            [2.0, 0.75, 0.35, 0.70, 0.16, 0.18, 0.20],
        ],
        dtype=torch.float32,
    )
    centers = labels_norm_xyzwhd[:, 1:4] * size
    extents = labels_norm_xyzwhd[:, 4:7] * size
    starts = centers - extents / 2.0
    ends = centers + extents / 2.0
    labels_abs_xyzxyz = torch.cat((labels_norm_xyzwhd[:, :1], starts, ends), dim=1)
    return labels_norm_xyzwhd, labels_abs_xyzxyz


def write_dummy_npy(path: Path, volume: np.ndarray) -> None:
    np.save(path, volume)


def time_operation(
    operation: Callable[[], Any],
    warmups: int,
    repeats: int,
    device: torch.device,
) -> tuple[float, float, float, float]:
    for _ in range(warmups):
        synchronize_device(device)
        result = operation()
        synchronize_device(device)
        del result
        gc.collect()

    elapsed_ms: list[float] = []
    for _ in range(repeats):
        gc.collect()
        synchronize_device(device)
        start = time.perf_counter()
        result = operation()
        synchronize_device(device)
        elapsed_ms.append((time.perf_counter() - start) * 1000.0)
        del result
        gc.collect()

    mean_ms = statistics.mean(elapsed_ms)
    std_ms = statistics.stdev(elapsed_ms) if len(elapsed_ms) > 1 else 0.0
    return mean_ms, std_ms, min(elapsed_ms), max(elapsed_ms)


def benchmark_stage(
    size: int,
    augmentation: str,
    operation: Callable[[], Any],
    warmups: int,
    repeats: int,
    device: torch.device,
) -> BenchmarkResult:
    try:
        mean_ms, std_ms, min_ms, max_ms = time_operation(operation, warmups, repeats, device)
    except Exception as exc:  # noqa: BLE001
        gc.collect()
        error_text = str(exc).replace("\n", " ").strip()
        LOGGER.warning("%s failed for %s^3: %s", augmentation, size, error_text)
        return BenchmarkResult(
            size=size,
            augmentation=augmentation,
            device=device.type,
            repeats=repeats,
            warmups=warmups,
            mean_ms=None,
            std_ms=None,
            min_ms=None,
            max_ms=None,
            status="failed",
            error=error_text,
        )

    return BenchmarkResult(
        size=size,
        augmentation=augmentation,
        device=device.type,
        repeats=repeats,
        warmups=warmups,
        mean_ms=mean_ms,
        std_ms=std_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        status="ok",
    )


def make_intensity_operation(
    volume: torch.Tensor,
    **kwargs: float | bool | tuple[float, float],
) -> Callable[[], torch.Tensor]:
    def operation() -> torch.Tensor:
        with torch.inference_mode():
            return batch_generators_intensity_augmentations(
                volume.clone(),
                **kwargs,
            )

    return operation


def make_gin_operation(
    volume: torch.Tensor,
    device: torch.device,
    rng: np.random.Generator,
    hidden_layers: int = 4,
    hidden_channels: int = 2,
    p_kernel_size_1: float = 0.8,
) -> Callable[[], torch.Tensor]:
    def operation() -> torch.Tensor:
        with torch.inference_mode():
            gin_model = GINNet(
                hidden_layers,
                hidden_channels,
                p_kernel_size_1,
            ).to(device=device, dtype=volume.dtype)
            for layer in gin_model.layers_kernel_size_3:
                torch.nn.init.normal_(layer.weight)
                torch.nn.init.normal_(layer.bias)
            for layer in gin_model.layers_kernel_size_1:
                torch.nn.init.normal_(layer.weight)
                torch.nn.init.normal_(layer.bias)

            output = torch.zeros_like(volume)
            for channel in range(volume.shape[0]):
                input_i = x = volume[channel].unsqueeze(0)
                num_layers = len(gin_model.layers_kernel_size_3)
                for index in range(num_layers):
                    if rng.random() <= p_kernel_size_1:
                        x = gin_model.layers_kernel_size_1[index](x)
                    else:
                        x = gin_model.layers_kernel_size_3[index](x)
                    if index < num_layers - 1:
                        x = F.leaky_relu(x)

                alpha = float(rng.random())
                output_i = alpha * input_i + (1.0 - alpha) * x
                output_i = output_i * (
                    torch.norm(input_i, p="fro") / (torch.norm(output_i, p="fro") + 1e-6)
                )
                output[channel] = output_i.squeeze(0)

            return output

    return operation


def make_resampling_operation(
    source_volume: torch.Tensor,
    target_shape: tuple[int, int, int],
) -> Callable[[], torch.Tensor]:
    def operation() -> torch.Tensor:
        with torch.inference_mode():
            return F.interpolate(
                source_volume.unsqueeze(0),
                size=target_shape,
                mode="trilinear",
            ).squeeze(0)

    return operation


def make_normalization_operation(
    volume: torch.Tensor,
    normalization_config: dict[str, float | str],
) -> Callable[[], torch.Tensor]:
    def operation() -> torch.Tensor:
        with torch.inference_mode():
            return normalize_image(volume.clone(), normalization_config)

    return operation


def make_stage_operations(
    volume: torch.Tensor,
    resampling_source_volume: torch.Tensor,
    labels_norm_xyzwhd: torch.Tensor,
    labels_abs_xyzxyz: torch.Tensor,
    device: torch.device,
    rng: np.random.Generator,
    normalization_config: dict[str, float | str],
) -> dict[str, Callable[[], Any]]:
    def channel_dropout() -> torch.Tensor:
        with torch.inference_mode():
            return random_channel_dropout(
                volume.clone(),
                drop_probabilities=[1.0] * volume.shape[0],
                fill_value=0.0,
            )

    def random_zoom_uniform() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        with torch.inference_mode():
            return random_zoom(
                volume.clone(),
                labels_abs_xyzxyz.clone(),
                masks=None,
                max_zoom=1.5,
                min_zoom=0.7,
                p=1.0,
                out_zoom_padding="constant",
                padding_fill_value=0.0,
                uniform_zoom=True,
            )

    def random_zoom_anisotropic() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        with torch.inference_mode():
            return random_zoom(
                volume.clone(),
                labels_abs_xyzxyz.clone(),
                masks=None,
                max_zoom=1.5,
                min_zoom=0.7,
                p=1.0,
                out_zoom_padding="constant",
                padding_fill_value=0.0,
                uniform_zoom=False,
            )

    def random_rotation_stage() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        with torch.inference_mode():
            return random_rotation(
                volume.clone(),
                labels_abs_xyzxyz.clone(),
                masks=None,
                angle_ranges=((-20.0, 20.0), (-20.0, 20.0), (-20.0, 20.0)),
                p=1.0,
                keep_box_volume=True,
                fill_value=0.0,
            )

    def tensor_cutout_stage() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        with torch.inference_mode():
            return tensor_cutout(
                volume.clone(),
                labels_abs_xyzxyz.clone(),
                masks=None,
                cutout_params=[(0.5, 1), (0.25, 2), (0.125, 4)],
                p=1.0,
            )

    def random_mirroring_stage() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        with torch.inference_mode():
            return random_mirroring(
                volume.clone(),
                labels_norm_xyzwhd.clone(),
                masks=None,
                mirror_axes=("z", "y", "x"),
                p=1.0,
            )

    intensity_operations = {
        "gaussian_noise": make_intensity_operation(
            volume,
            noise_prob=1.0,
            blur_prob=0.0,
            mult_brightness_prob=0.0,
            add_brightness_prob=0.0,
            contrast_prob=0.0,
            lowres_prob=0.0,
            gamma_prob=0.0,
        ),
        "gaussian_blur": make_intensity_operation(
            volume,
            noise_prob=0.0,
            blur_prob=1.0,
            mult_brightness_prob=0.0,
            add_brightness_prob=0.0,
            contrast_prob=0.0,
            lowres_prob=0.0,
            gamma_prob=0.0,
        ),
        "multiplicative_brightness": make_intensity_operation(
            volume,
            noise_prob=0.0,
            blur_prob=0.0,
            mult_brightness_prob=1.0,
            add_brightness_prob=0.0,
            contrast_prob=0.0,
            lowres_prob=0.0,
            gamma_prob=0.0,
        ),
        "additive_brightness": make_intensity_operation(
            volume,
            noise_prob=0.0,
            blur_prob=0.0,
            mult_brightness_prob=0.0,
            add_brightness_prob=1.0,
            contrast_prob=0.0,
            lowres_prob=0.0,
            gamma_prob=0.0,
        ),
        "contrast": make_intensity_operation(
            volume,
            noise_prob=0.0,
            blur_prob=0.0,
            mult_brightness_prob=0.0,
            add_brightness_prob=0.0,
            contrast_prob=1.0,
            lowres_prob=0.0,
            gamma_prob=0.0,
        ),
        "low_resolution": make_intensity_operation(
            volume,
            noise_prob=0.0,
            blur_prob=0.0,
            mult_brightness_prob=0.0,
            add_brightness_prob=0.0,
            contrast_prob=0.0,
            lowres_prob=1.0,
            gamma_prob=0.0,
        ),
        "gamma": make_intensity_operation(
            volume,
            noise_prob=0.0,
            blur_prob=0.0,
            mult_brightness_prob=0.0,
            add_brightness_prob=0.0,
            contrast_prob=0.0,
            lowres_prob=0.0,
            gamma_prob=1.0,
        ),
    }

    return {
        "resampling": make_resampling_operation(
            resampling_source_volume,
            tuple(volume.shape[1:]),
        ),
        "normalization": make_normalization_operation(
            volume,
            normalization_config,
        ),
        **intensity_operations,
        "gin": make_gin_operation(volume, device, rng),
        "channel_dropout": channel_dropout,
        "random_zoom_uniform": random_zoom_uniform,
        "random_zoom_anisotropic": random_zoom_anisotropic,
        "random_rotation": random_rotation_stage,
        "tensor_cutout": tensor_cutout_stage,
        "random_mirroring": random_mirroring_stage,
    }


def write_results(path: Path, results: list[BenchmarkResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "size",
                "augmentation",
                "device",
                "repeats",
                "warmups",
                "mean_ms",
                "std_ms",
                "min_ms",
                "max_ms",
                "status",
                "error",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.size,
                    result.augmentation,
                    result.device,
                    result.repeats,
                    result.warmups,
                    "" if result.mean_ms is None else f"{result.mean_ms:.3f}",
                    "" if result.std_ms is None else f"{result.std_ms:.3f}",
                    "" if result.min_ms is None else f"{result.min_ms:.3f}",
                    "" if result.max_ms is None else f"{result.max_ms:.3f}",
                    result.status,
                    result.error,
                ]
            )


def format_summary_cell(result: BenchmarkResult | None) -> str:
    if result is None:
        return "-"
    if result.status == "unsupported":
        return "unsupported"
    if result.status != "ok" or result.mean_ms is None or result.std_ms is None:
        return "failed"
    return f"{result.mean_ms:.1f} +- {result.std_ms:.1f}"


def log_summary(results: list[BenchmarkResult]) -> None:
    device_names = sorted({result.device for result in results})
    LOGGER.info("Benchmark summary (%s):", ", ".join(device_names))
    sizes = sorted({result.size for result in results})

    augmentation_order = [
        augmentation
        for augmentation in DEFAULT_AUGMENTATIONS
        if any(result.augmentation == augmentation for result in results)
    ]
    extra_augmentations = sorted(
        {
            result.augmentation
            for result in results
            if result.augmentation not in DEFAULT_AUGMENTATIONS
        }
    )
    augmentation_order.extend(extra_augmentations)

    result_lookup = {
        (result.augmentation, result.size): result
        for result in results
    }

    headers = ["augmentation", *[f"{size}^3 [ms]" for size in sizes]]
    rows = [
        [
            augmentation,
            *[
                format_summary_cell(result_lookup.get((augmentation, size)))
                for size in sizes
            ],
        ]
        for augmentation in augmentation_order
    ]

    column_widths = [
        max(len(str(cell)) for cell in [header, *[row[index] for row in rows]])
        for index, header in enumerate(headers)
    ]

    def format_row(cells: list[str]) -> str:
        return " | ".join(
            cell.ljust(column_widths[index])
            for index, cell in enumerate(cells)
        )

    separator = "-+-".join("-" * width for width in column_widths)

    LOGGER.info(format_row(headers))
    LOGGER.info(separator)
    for row in rows:
        LOGGER.info(format_row(row))


def benchmark_size(
    size: int,
    args: argparse.Namespace,
    device: torch.device,
    generated_dir: Path,
    normalization_config: dict[str, float | str],
) -> list[BenchmarkResult]:
    shape = (size, size, size)
    resampling_source_shape = build_resampling_source_shape(size, args.resampling_source_scale)
    LOGGER.info("Preparing dummy %d^3 volume with %d channel(s)", size, args.channels)
    rng = np.random.default_rng(args.seed + size)
    volume_np = build_dummy_volume(shape=shape, channels=args.channels, rng=rng)
    resampling_source_np = build_dummy_volume(
        shape=resampling_source_shape,
        channels=args.channels,
        rng=rng,
    )
    dummy_path = generated_dir / f"dummy_{size}.npy"
    write_dummy_npy(dummy_path, volume_np)

    results = [
        benchmark_stage(
            size=size,
            augmentation="data_loading",
            operation=lambda path=dummy_path, benchmark_device=device: torch.from_numpy(
                np.load(path, allow_pickle=False)
            ).to(benchmark_device),
            warmups=args.warmups,
            repeats=args.repeats,
            device=device,
        )
    ]

    volume_torch = torch.from_numpy(volume_np).to(device)
    resampling_source_torch = torch.from_numpy(resampling_source_np).to(device)
    labels_norm_xyzwhd_cpu, labels_abs_xyzxyz_cpu = build_dummy_labels(size)
    labels_norm_xyzwhd = labels_norm_xyzwhd_cpu.to(device)
    labels_abs_xyzxyz = labels_abs_xyzxyz_cpu.to(device)
    stage_operations = make_stage_operations(
        volume_torch,
        resampling_source_torch,
        labels_norm_xyzwhd,
        labels_abs_xyzxyz,
        device,
        rng,
        normalization_config,
    )

    for augmentation in args.augmentations:
        if augmentation == "data_loading":
            continue
        unsupported_reason = get_unsupported_reason(augmentation, device)
        if unsupported_reason is not None:
            LOGGER.info(
                "Skipping %s for %d^3 on %s: %s",
                augmentation,
                size,
                device,
                unsupported_reason,
            )
            results.append(
                make_unsupported_result(
                    size=size,
                    augmentation=augmentation,
                    device=device,
                    repeats=args.repeats,
                    warmups=args.warmups,
                    reason=unsupported_reason,
                )
            )
            continue
        LOGGER.info("Benchmarking %s for %d^3", augmentation, size)
        results.append(
            benchmark_stage(
                size=size,
                augmentation=augmentation,
                operation=stage_operations[augmentation],
                warmups=args.warmups,
                repeats=args.repeats,
                device=device,
            )
        )

    del volume_np
    del resampling_source_np
    del volume_torch
    del resampling_source_torch
    del labels_norm_xyzwhd
    del labels_abs_xyzxyz
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return results


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = resolve_device(args.device)
    normalization_config = build_normalization_config(args)

    output_dir = args.output_dir.resolve()
    generated_dir = output_dir / "generated_npy"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Running benchmark on %s", device)
    LOGGER.info("Writing dummy benchmark inputs to %s", generated_dir)
    results: list[BenchmarkResult] = []

    for size in args.sizes:
        results.extend(
            benchmark_size(
                size=size,
                args=args,
                device=device,
                generated_dir=generated_dir,
                normalization_config=normalization_config,
            )
        )

    csv_path = output_dir / args.csv_name
    write_results(csv_path, results)
    log_summary(results)
    LOGGER.info("Wrote CSV results to %s", csv_path)

    if not args.keep_generated_files:
        shutil.rmtree(generated_dir, ignore_errors=True)
        LOGGER.info("Removed generated dummy files in %s", generated_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
