from __future__ import annotations

import json
import logging
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

LOGGER = logging.getLogger(__name__)


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


def _kfold_chunks(items: list[str], k: int, seed: int) -> list[list[str]]:
    min_splits = 2
    if k < min_splits:
        msg = "k must be >= 2"
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
    cases: list[str], k: int = 5, seed: int = 42
) -> list[dict[str, list[str]]]:
    """Create k folds with train/val/test per fold.

    For fold i: test = chunk i, val = chunk (i+1) mod k, train = remaining.
    """
    chunks = _kfold_chunks(cases, k=k, seed=seed)
    folds: list[dict[str, list[str]]] = []
    for i in range(k):
        test = sorted(chunks[i])
        val = sorted(chunks[(i + 1) % k])
        train = sorted(
            [
                c
                for j, chunk in enumerate(chunks)
                if j not in (i, (i + 1) % k)
                for c in chunk
            ]
        )
        folds.append({"train": train, "val": val, "test": test})
    return folds


def _group_cases_by_prefix(cases: list[str], sep: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for c in cases:
        key = c.split(sep, 1)[0] if sep else c
        groups.setdefault(key, []).append(c)
    return groups


def k_fold_data_splits(
    dataset_path: Path,
    seed: int = 42,
    k: int = 5,
    *,
    group_by_prefix: bool = False,
    prefix_sep: str = "_",
) -> Path:
    """Create k-fold data splits from DATASET_PATH/images and write data_splits.json.

    The output is a JSON list with k items. Each item is a dict with keys
    'train', 'val', 'test', each containing a list of case names (without extensions).
    """
    try:
        images_dir = dataset_path / "images"
        cases = _find_case_names(images_dir)
        if not group_by_prefix:
            folds = _build_folds(cases, k=k, seed=seed)
        else:
            # Group cases by prefix and split on group level so that all cases
            # with the same prefix end up in the same fold.
            groups = _group_cases_by_prefix(cases, sep=prefix_sep)
            group_keys = list(groups.keys())
            group_chunks = _kfold_chunks(group_keys, k=k, seed=seed)

            folds = []
            for i in range(k):
                test_groups = group_chunks[i]
                val_groups = group_chunks[(i + 1) % k]
                train_groups = [
                    g
                    for j, ch in enumerate(group_chunks)
                    if j not in (i, (i + 1) % k)
                    for g in ch
                ]

                test = sorted([c for g in test_groups for c in groups[g]])
                val = sorted([c for g in val_groups for c in groups[g]])
                train = sorted([c for g in train_groups for c in groups[g]])

                folds.append({"train": train, "val": val, "test": test})

        out_path = dataset_path / "data_splits.json"
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
