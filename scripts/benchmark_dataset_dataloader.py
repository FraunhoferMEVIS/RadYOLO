from __future__ import annotations

import argparse
import importlib
import logging
import statistics
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm.auto import tqdm


def _ensure_project_root_on_sys_path(script_path: Path | None = None) -> Path:
    project_root = (script_path or Path(__file__)).resolve().parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


def _ensure_radyolo_package(script_path: Path | None = None) -> Path:
    project_root = _ensure_project_root_on_sys_path(script_path)
    package_root = project_root / "radyolo"
    if "radyolo" not in sys.modules:
        package = types.ModuleType("radyolo")
        package.__file__ = str(package_root / "__init__.py")
        package.__path__ = [str(package_root)]
        sys.modules["radyolo"] = package
    return project_root


_ensure_radyolo_package()

PatchSize = importlib.import_module("radyolo.config.spatial").PatchSize
_data_module = importlib.import_module("radyolo.data.data")
WholeImageDataset = _data_module.WholeImageDataset
build_dataloader = _data_module.build_dataloader

LOGGER = logging.getLogger("benchmark_dataset_dataloader")


@dataclass(frozen=True)
class BenchmarkSummary:
    num_batches: int
    num_samples: int
    total_seconds: float
    batch_mean_seconds: float
    batch_std_seconds: float
    batch_min_seconds: float
    batch_max_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a single full pass over the RadYOLO dataset dataloader and report "
            "total iteration time plus per-batch request statistics."
        )
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to the dataset root containing images/, labels/ and data_splits.json.",
    )
    parser.add_argument(
        "--patch-size",
        nargs=3,
        type=int,
        metavar=("X", "Y", "Z"),
        required=True,
        help="Patch size used when constructing the dataset as x y z.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Batch size used by the dataloader.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        required=True,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold index selected from data_splits.json.",
    )
    parser.add_argument(
        "--group",
        type=str,
        choices=["train", "val", "test"],
        default="train",
        help="Data split group to iterate.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataloader order before iteration.",
    )
    parser.add_argument(
        "--load-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load masks if present in the dataset.",
    )
    return parser.parse_args()


def build_benchmark_dataset(
    dataset_path: Path,
    patch_size_xyz: tuple[int, int, int],
    fold: int,
    group: str,
    load_masks: bool,
) -> WholeImageDataset:
    return WholeImageDataset(
        src_path=dataset_path,
        patch_size=PatchSize.from_xyz(patch_size_xyz),
        augment=False,
        fold=fold,
        group=group,
        normalization={"type": "none"},
        load_masks=load_masks,
    )


def benchmark_dataloader_iteration(dataloader: torch.utils.data.DataLoader) -> BenchmarkSummary:
    batch_durations: list[float] = []
    num_batches = 0
    num_samples = 0
    progress_overhead_seconds = 0.0

    total_start = time.perf_counter()
    dataloader_iterator = iter(dataloader)
    with tqdm(
        total=len(dataloader), desc="Benchmarking", unit="batch", smoothing=0.05
    ) as progress_bar:
        while True:
            batch_start = time.perf_counter()
            try:
                batch = next(dataloader_iterator)
            except StopIteration:
                break
            batch_end = time.perf_counter()

            batch_durations.append(batch_end - batch_start)
            num_batches += 1
            num_samples += int(batch["img"].shape[0])

            progress_start = time.perf_counter()
            progress_bar.update(1)
            progress_overhead_seconds += time.perf_counter() - progress_start
    total_end = time.perf_counter()

    if not batch_durations:
        msg = "The dataloader did not yield any batches. Check the selected dataset split."
        raise RuntimeError(msg)

    return BenchmarkSummary(
        num_batches=num_batches,
        num_samples=num_samples,
        total_seconds=(total_end - total_start) - progress_overhead_seconds,
        batch_mean_seconds=statistics.fmean(batch_durations),
        batch_std_seconds=statistics.stdev(batch_durations) if len(batch_durations) > 1 else 0.0,
        batch_min_seconds=min(batch_durations),
        batch_max_seconds=max(batch_durations),
    )


def log_summary(summary: BenchmarkSummary) -> None:
    LOGGER.info("Batches requested: %d", summary.num_batches)
    LOGGER.info("Samples seen: %d", summary.num_samples)
    LOGGER.info("Full iteration time: %.6f s", summary.total_seconds)
    LOGGER.info("Average batch request time: %.6f s", summary.batch_mean_seconds)
    LOGGER.info("Batch request stddev: %.6f s", summary.batch_std_seconds)
    LOGGER.info("Fastest batch request: %.6f s", summary.batch_min_seconds)
    LOGGER.info("Slowest batch request: %.6f s", summary.batch_max_seconds)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    try:
        dataset = build_benchmark_dataset(
            dataset_path=args.dataset_path,
            patch_size_xyz=tuple(args.patch_size),
            fold=args.fold,
            group=args.group,
            load_masks=args.load_masks,
        )
        dataloader = build_dataloader(
            dataset=dataset,
            batch_size=args.batch_size,
            workers=args.workers,
            shuffle=args.shuffle,
            device=torch.device("cpu"),
        )
        summary = benchmark_dataloader_iteration(dataloader)
    except Exception as exc:
        LOGGER.exception("Failed to benchmark dataset dataloader", exc_info=exc)
        return 1

    LOGGER.info("Dataset: %s", args.dataset_path)
    LOGGER.info("Patch size (x y z): %s %s %s", *args.patch_size)
    LOGGER.info("Batch size: %d", args.batch_size)
    LOGGER.info("Workers: %d", args.workers)
    LOGGER.info("Fold/group: %d/%s", args.fold, args.group)
    log_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
