from __future__ import annotations

import logging
import random
from pathlib import Path
import json

LOGGER = logging.getLogger(__name__)


def _find_case_names(images_dir: Path) -> list[str]:
    """Return case base names (without extensions) from an images directory.

    Supports .nii.gz and .nii files. Ignores other files.
    """
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

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
        raise RuntimeError(f"No .nii or .nii.gz files found in {images_dir}")
    return cases


def _kfold_chunks(items: list[str], k: int, seed: int) -> list[list[str]]:
    if k <= 1:
        raise ValueError("k must be >= 3")
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


def _build_folds(cases: list[str], k: int = 5, seed: int = 42) -> list[dict[str, list[str]]]:
    """Create k folds with train/val/test per fold.

    For fold i: test = chunk i, val = chunk (i+1) mod k, train = remaining.
    """
    chunks = _kfold_chunks(cases, k=k, seed=seed)
    folds: list[dict[str, list[str]]] = []
    for i in range(k):
        test = list(chunks[i])
        val = list(chunks[(i + 1) % k])
        train = [c for j, chunk in enumerate(chunks) if j not in (i, (i + 1) % k) for c in chunk]
        folds.append({"train": train, "val": val, "test": test})
    return folds


def k_fold_data_splits(dataset_path: Path, seed: int = 42, k: int = 5) -> Path:
    """Create k-fold data splits from DATASET_PATH/images and write data_splits.json.

    The output is a JSON list with k items. Each item is a dict with keys
    'train', 'val', 'test', each containing a list of case names (without extensions).
    """
    try:
        images_dir = dataset_path / "images"
        cases = _find_case_names(images_dir)
        folds = _build_folds(cases, k=k, seed=seed)

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
