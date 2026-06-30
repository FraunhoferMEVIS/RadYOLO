from __future__ import annotations

import json
import logging
import math
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

LOGGER = logging.getLogger(__name__)
DEFAULT_SPLIT_FILE_NAME = "data_splits.json"


def get_split_file_path(dataset_path: Path, split_file_name: str = DEFAULT_SPLIT_FILE_NAME) -> Path:
    """Return the output path for the split JSON file inside a dataset directory."""
    return dataset_path / split_file_name


def _find_case_names(images_dir: Path) -> list[str]:
    """Return case base names (without extensions) from an images directory.

    Supports .nii.gz and .nii files. Ignores other files.
    """
    if not images_dir.exists() or not images_dir.is_dir():
        msg = f"Images directory not found: {images_dir}"
        raise FileNotFoundError(msg)

    cases: list[str] = []
    for p in sorted(images_dir.iterdir()):
        if not p.is_file():
            continue
        name = p.name
        if name.endswith(".nii.gz"):
            cases.append(name[: -len(".nii.gz")])
        elif name.endswith(".nii"):
            cases.append(name[: -len(".nii")])
        else:
            continue
    if not cases:
        msg = f"No .nii or .nii.gz files found in {images_dir}"
        raise RuntimeError(msg)
    return cases


def _kfold_chunks(items: list[str], k: int, seed: int, *, min_splits: int = 2) -> list[list[str]]:
    """Shuffle items deterministically and split them into k near-even chunks."""
    if k < min_splits:
        msg = f"k must be >= {min_splits}"
        raise ValueError(msg)
    items = list(items)
    rnd = random.Random(seed)
    rnd.shuffle(items)

    n = len(items)
    base = n // k
    rem = n % k
    sizes = [base + (1 if i < rem else 0) for i in range(k)]
    chunks: list[list[str]] = []
    idx = 0
    for s in sizes:
        chunks.append(items[idx : idx + s])
        idx += s
    return chunks


def _build_folds(
    cases: list[str], k: int = 5, seed: int = 42, *, create_test_group: bool = True
) -> list[dict[str, list[str]]]:
    """Create k folds with train/val/test per fold.

    When test groups are enabled, fold i uses test = chunk i, val = chunk (i+1) mod k,
    and train = remaining. Otherwise val = chunk i, train = remaining and test stays empty.
    """
    chunks = _kfold_chunks(cases, k=k, seed=seed, min_splits=3 if create_test_group else 2)
    folds: list[dict[str, list[str]]] = []
    for i in range(k):
        if create_test_group:
            excluded_chunk_indices = {i, (i + 1) % k}
            test = sorted(chunks[i])
            val = sorted(chunks[(i + 1) % k])
        else:
            excluded_chunk_indices = {i}
            test = []
            val = sorted(chunks[i])

        train = sorted(
            [c for j, chunk in enumerate(chunks) if j not in excluded_chunk_indices for c in chunk]
        )
        folds.append({"train": train, "val": val, "test": test})
    return folds


def _group_cases_by_prefix(cases: list[str], sep: str) -> dict[str, list[str]]:
    """Group case names by the substring before the first separator occurrence."""
    groups: dict[str, list[str]] = {}
    for c in cases:
        key = c.split(sep, 1)[0] if sep else c
        groups.setdefault(key, []).append(c)
    return groups


def _get_case_groups(
    cases: list[str], *, group_by_prefix: bool, prefix_sep: str
) -> list[list[str]]:
    """Return singleton groups or prefix-based groups for the provided case names."""
    if not group_by_prefix:
        return [[case] for case in cases]
    return list(_group_cases_by_prefix(cases, sep=prefix_sep).values())


def _flatten_case_groups(case_groups: list[list[str]]) -> list[str]:
    """Flatten grouped case names into a single sorted list."""
    return sorted(case for group in case_groups for case in group)


def _allocate_split_group_counts(
    total_groups: int,
    train_percentage: float,
    val_percentage: float,
    test_percentage: float,
) -> tuple[int, int, int]:
    """Convert split percentages into integer group counts using largest remainders."""
    percentages = [train_percentage, val_percentage, test_percentage]
    if any(percentage < 0 for percentage in percentages):
        msg = "train, val and test percentages must all be greater than or equal to 0"
        raise ValueError(msg)
    if not math.isclose(sum(percentages), 100.0, rel_tol=0.0, abs_tol=1e-6):
        msg = "train, val and test percentages must sum to 100"
        raise ValueError(msg)
    if total_groups < 1:
        msg = "at least 1 case or case group is required"
        raise ValueError(msg)

    exact_counts = [total_groups * percentage / 100.0 for percentage in percentages]
    counts = [math.floor(exact) for exact in exact_counts]
    remaining_groups = total_groups - sum(counts)

    if remaining_groups:
        remainders = [
            (exact - math.floor(exact), -index) for index, exact in enumerate(exact_counts)
        ]
        for _, negative_index in sorted(remainders, reverse=True)[:remaining_groups]:
            counts[-negative_index] += 1

    return counts[0], counts[1], counts[2]


def k_fold_data_splits(
    dataset_path: Path,
    seed: int = 42,
    k: int = 5,
    *,
    split_file_name: str = DEFAULT_SPLIT_FILE_NAME,
    create_test_group: bool = True,
    group_by_prefix: bool = False,
    prefix_sep: str = "_",
) -> Path:
    """Create k-fold data splits from DATASET_PATH/images and write a split file.

    The output is a JSON list with k items. Each item is a dict with keys
    'train', 'val', 'test', each containing a list of case names (without extensions).
    """
    try:
        images_dir = dataset_path / "images"
        cases = _find_case_names(images_dir)
        case_groups = _get_case_groups(
            cases,
            group_by_prefix=group_by_prefix,
            prefix_sep=prefix_sep,
        )
        if not group_by_prefix:
            folds = _build_folds(cases, k=k, seed=seed, create_test_group=create_test_group)
        else:
            shuffled_group_ids = list(range(len(case_groups)))
            group_chunks = _kfold_chunks(
                shuffled_group_ids,
                k=k,
                seed=seed,
                min_splits=3 if create_test_group else 2,
            )

            folds = []
            for i in range(k):
                if create_test_group:
                    excluded_chunk_indices = {i, (i + 1) % k}
                    test_group_ids = group_chunks[i]
                    val_group_ids = group_chunks[(i + 1) % k]
                else:
                    excluded_chunk_indices = {i}
                    test_group_ids = []
                    val_group_ids = group_chunks[i]

                train_group_ids = [
                    group_id
                    for j, chunk in enumerate(group_chunks)
                    if j not in excluded_chunk_indices
                    for group_id in chunk
                ]

                folds.append(
                    {
                        "train": _flatten_case_groups(
                            [case_groups[group_id] for group_id in train_group_ids]
                        ),
                        "val": _flatten_case_groups(
                            [case_groups[group_id] for group_id in val_group_ids]
                        ),
                        "test": _flatten_case_groups(
                            [case_groups[group_id] for group_id in test_group_ids]
                        ),
                    }
                )

        out_path = get_split_file_path(dataset_path, split_file_name)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(folds, f, indent=2)

        LOGGER.info(
            "Wrote %d-fold train/val/test splits to %s (total cases: %d)",
            k,
            out_path,
            len(cases),
        )
    except Exception as e:
        LOGGER.exception("Failed to create data splits", exc_info=e)
        raise
    else:
        return out_path


def train_val_test_data_split(  # noqa: PLR0913
    dataset_path: Path,
    seed: int = 42,
    *,
    split_file_name: str = DEFAULT_SPLIT_FILE_NAME,
    train_percentage: float = 70.0,
    val_percentage: float = 15.0,
    test_percentage: float = 15.0,
    group_by_prefix: bool = False,
    prefix_sep: str = "_",
) -> Path:
    """Create a single train/val/test split from DATASET_PATH/images and write a split file."""
    try:
        images_dir = dataset_path / "images"
        cases = _find_case_names(images_dir)
        case_groups = _get_case_groups(
            cases,
            group_by_prefix=group_by_prefix,
            prefix_sep=prefix_sep,
        )

        train_group_count, val_group_count, test_group_count = _allocate_split_group_counts(
            len(case_groups),
            train_percentage=train_percentage,
            val_percentage=val_percentage,
            test_percentage=test_percentage,
        )

        shuffled_group_ids = list(range(len(case_groups)))
        random.Random(seed).shuffle(shuffled_group_ids)

        train_group_ids = shuffled_group_ids[:train_group_count]
        val_start = train_group_count
        val_stop = val_start + val_group_count
        val_group_ids = shuffled_group_ids[val_start:val_stop]
        test_group_ids = shuffled_group_ids[val_stop : val_stop + test_group_count]

        folds = [
            {
                "train": _flatten_case_groups(
                    [case_groups[group_id] for group_id in train_group_ids]
                ),
                "val": _flatten_case_groups([case_groups[group_id] for group_id in val_group_ids]),
                "test": _flatten_case_groups(
                    [case_groups[group_id] for group_id in test_group_ids]
                ),
            }
        ]

        out_path = get_split_file_path(dataset_path, split_file_name)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(folds, f, indent=2)

        LOGGER.info(
            "Wrote single train/val/test split to %s (train=%d, val=%d, test=%d, total cases: %d)",
            out_path,
            len(folds[0]["train"]),
            len(folds[0]["val"]),
            len(folds[0]["test"]),
            len(cases),
        )
    except Exception as e:
        LOGGER.exception("Failed to create train/val/test split", exc_info=e)
        raise
    else:
        return out_path
