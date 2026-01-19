from __future__ import annotations

import contextlib
import csv
import json
import logging
import os
import pickle
import re
import shutil
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import yaml
from nibabel.filebasedimages import ImageFileError
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterable

LOGGER = logging.getLogger(__name__)


@dataclass
class SplitSets:
    train: list[str]
    val: list[str]
    test: list[str]


def _sanitize_name(name: str) -> str:
    """Return a filesystem and nnDetection friendly name (no dots)."""
    # Replace dots and spaces with underscores; collapse repeats
    name = re.sub(r"[\s.]+", "_", name.strip())
    return re.sub(r"_+", "_", name)


def _load_dataset_names(dataset_yaml: Path) -> dict[int, str] | None:
    try:
        with dataset_yaml.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return None

    names = data.get("names")
    if names is None:
        # also support list-style ['a','b']
        class_list = data.get("class_names") or data.get("classes")
        if isinstance(class_list, list):
            return {i: str(n) for i, n in enumerate(class_list)}
        return None

    if isinstance(names, dict):
        # keys may be strings in some datasets
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return None


def _read_splits(dataset_dir: Path, fold: int) -> SplitSets:
    splits_path = dataset_dir / "data_splits.json"
    if not splits_path.exists():
        # Fallback: all images -> train; none for val/test
        images = sorted((dataset_dir / "images").glob("*.nii.gz"))
        case_ids = [p.stem for p in images]
        LOGGER.warning(
            "data_splits.json not found; assigning all %d cases to train set.", len(case_ids)
        )
        return SplitSets(train=case_ids, val=[], test=[])

    with splits_path.open("r", encoding="utf-8") as f:
        splits = json.load(f)
    if not isinstance(splits, list) or fold >= len(splits):
        number_of_splits = len(splits) if isinstance(splits, list) else "invalid"
        msg = f"Invalid fold {fold}; data_splits.json contains {number_of_splits} entries"
        raise ValueError(msg)
    entry = splits[fold]
    return SplitSets(
        train=entry.get("train", []), val=entry.get("val", []), test=entry.get("test", [])
    )


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _copy_image_to_nndet(image_path: Path, out_images_dir: Path, case_id: str) -> Path:
    out_path = out_images_dir / f"{case_id}_0000.nii.gz"
    _ensure_dir(out_images_dir)
    # Prefer copying over re-saving to keep headers identical
    shutil.copy2(image_path, out_path)
    return out_path


def _labels_for_case(
    labels_dir: Path, case_id: str
) -> list[tuple[int, float, float, float, float, float, float]]:
    label_file = labels_dir / f"{case_id}.txt"
    if not label_file.exists():
        return []
    lines: list[str] = label_file.read_text(encoding="utf-8").strip().splitlines()
    tuples: list[tuple[int, float, float, float, float, float, float]] = []
    required_label_parts = 7
    for ln in lines:
        if not ln.strip():
            continue
        parts = ln.split()
        if len(parts) != required_label_parts:
            msg = f"Invalid label line in {label_file}: '{ln}'"
            raise ValueError(msg)
        c, xc, yc, zc, sx, sy, sz = parts
        tuples.append(
            (int(float(c)), float(xc), float(yc), float(zc), float(sx), float(sy), float(sz))
        )
    return tuples


def _try_load_masks(masks_dir: Path, case_id: str) -> np.ndarray | None:
    mask_path = masks_dir / f"{case_id}.nii.gz"
    if not mask_path.exists():
        return None
    try:
        nii = nib.load(mask_path)
        return np.asanyarray(nii.dataobj)
    except (FileNotFoundError, ImageFileError) as e:
        LOGGER.warning("Failed to load mask %s: %s", mask_path, e)
        return None


def _compose_instance_seg_from_masks(mask_arr: np.ndarray, num_instances: int) -> np.ndarray | None:
    """Return a single-channel integer labelmap from multi-channel mask array.

    Supports both channel-first (K, X, Y, Z) and channel-last (X, Y, Z, K) layouts.
    Returns None if dimensions are not compatible.
    """
    single_mask_dims = 3
    multiple_mask_dims = 4
    if mask_arr.ndim == single_mask_dims:
        # Single mask provided but multiple instances expected; skip
        if num_instances <= 1:
            labelmap = (mask_arr > 0).astype(np.uint16)
            labelmap[labelmap > 0] = 1
            return labelmap
        return None
    if mask_arr.ndim != multiple_mask_dims:
        return None

    k_last = mask_arr.shape[-1]
    k_first = mask_arr.shape[0]
    if k_last == num_instances:
        # channels-last
        spatial = mask_arr[..., 0].shape
        out = np.zeros(spatial, dtype=np.uint16)
        for i in range(num_instances):
            m = mask_arr[..., i] > 0
            out[m] = i + 1
        return out
    if k_first == num_instances:
        # channels-first
        spatial = mask_arr[0].shape
        out = np.zeros(spatial, dtype=np.uint16)
        for i in range(num_instances):
            m = mask_arr[i] > 0
            out[m] = i + 1
        return out
    return None


def _compose_instance_seg_from_boxes(
    image_shape: tuple[int, int, int],
    bboxes: Iterable[tuple[float, float, float, float, float, float]],
) -> np.ndarray:
    """Create integer instance segmentation by filling axis-aligned cuboids.

    image_shape is (X, Y, Z) in nibabel array convention.
    bboxes are normalized (xc, yc, zc, sx, sy, sz) in [0, 1] relative to image size.
    """
    width, height, depth = image_shape
    out = np.zeros((width, height, depth), dtype=np.uint16)
    for idx, (xc, yc, zc, sx, sy, sz) in enumerate(bboxes, start=1):
        cx = xc * width
        cy = yc * height
        cz = zc * depth
        rx = (sx * width) / 2.0
        ry = (sy * height) / 2.0
        rz = (sz * depth) / 2.0
        x0 = max(0, int(np.floor(cx - rx)))
        x1 = min(width, int(np.ceil(cx + rx)))
        y0 = max(0, int(np.floor(cy - ry)))
        y1 = min(height, int(np.ceil(cy + ry)))
        z0 = max(0, int(np.floor(cz - rz)))
        z1 = min(depth, int(np.ceil(cz + rz)))
        if x1 <= x0 or y1 <= y0 or z1 <= z0:
            continue
        out[x0:x1, y0:y1, z0:z1] = idx
    return out


def _write_label_files(
    out_labels_dir: Path,
    case_id: str,
    labelmap: np.ndarray,
    instance_classes: list[int],
    ref_nii: nib.Nifti1Image,
) -> None:
    _ensure_dir(out_labels_dir)
    lab_path = out_labels_dir / f"{case_id}.nii.gz"
    lab_img = nib.Nifti1Image(
        labelmap.astype(np.uint16), affine=ref_nii.affine, header=ref_nii.header
    )
    lab_img.set_data_dtype(np.uint16)
    nib.save(lab_img, lab_path)

    properties = {str(i + 1): int(c) for i, c in enumerate(instance_classes)}
    json_path = out_labels_dir / f"{case_id}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"instances": properties}, f, indent=4)


def convert_dataset_to_nndetection(
    dataset_dir: Path,
    out_root: Path,
    task_id: int = 100,
    task_name: str | None = None,
    fold: int = 0,
    include_val_in_train: bool = True,
    modality_name: str = "CT",
    workers: int = 0,
) -> Path:
    """Convert YOLO11_3D dataset to nnDetection format.

    Returns the path to the created Task folder.
    """
    dataset_dir = dataset_dir.resolve()
    out_root = out_root.resolve()
    if task_name is None:
        task_name = _sanitize_name(dataset_dir.name)
    task_name = _sanitize_name(task_name)
    task_dir = out_root / f"Task{task_id:03d}_{task_name}"

    raw = task_dir / "raw_splitted"
    imagesTr = raw / "imagesTr"  # noqa: N806
    labelsTr = raw / "labelsTr"  # noqa: N806
    imagesTs = raw / "imagesTs"  # noqa: N806
    labelsTs = raw / "labelsTs"  # noqa: N806
    _ensure_dir(imagesTr)
    _ensure_dir(labelsTr)
    # test folders are created only if needed later

    names = _load_dataset_names(dataset_dir / "dataset.yaml")
    splits = _read_splits(dataset_dir, fold=fold)

    # Helper: choose train set
    train_cases: list[str] = list(splits.train)
    if include_val_in_train:
        train_cases += list(splits.val)
    train_cases = sorted(set(train_cases))

    # Process Train
    label_dir = dataset_dir / "labels"

    total_cases = len(train_cases) + len(splits.test)
    max_workers = os.cpu_count() or 1
    if workers and workers > 0:
        max_workers = workers

    def _submit_cases(
        executor: ProcessPoolExecutor, cases: Iterable[str], out_img: Path, out_lbl: Path
    ) -> dict:
        futures = {}
        for case_id in cases:
            fut = executor.submit(
                _convert_one_case, str(dataset_dir), case_id, str(out_img), str(out_lbl)
            )
            futures[fut] = case_id
        return futures

    processed_train = 0
    processed_test = 0
    mapping_rows: list[dict[str, str]] = []
    if total_cases > 0:
        pbar = tqdm(total=total_cases, desc="Converting", unit="case")
        executor = ProcessPoolExecutor(max_workers=max_workers)
        try:
            futures = {}
            _ensure_dir(imagesTr)
            _ensure_dir(labelsTr)
            futures.update(_submit_cases(executor, train_cases, imagesTr, labelsTr))
            if splits.test:
                _ensure_dir(imagesTs)
                _ensure_dir(labelsTs)
                futures.update(_submit_cases(executor, splits.test, imagesTs, labelsTs))

            for fut in as_completed(futures):
                case_id = futures[fut]
                try:
                    ok, split, src_image_name, dst_image_names, src_case_id, dst_case_id = (
                        fut.result()
                    )
                except Exception:
                    LOGGER.exception("Failed to convert case %s", case_id)
                else:
                    if ok:
                        if split == "train":
                            processed_train += 1
                        elif split == "test":
                            processed_test += 1
                        # Write one row per nnDetection image (one per modality/channel)
                        for dn in dst_image_names or []:
                            mapping_rows.append(  # noqa: PERF401
                                {
                                    "yolo_case_id": src_case_id,
                                    "yolo_image": src_image_name,
                                    "nndet_case_id": dst_case_id,
                                    "nndet_image": dn,
                                    "split": split,
                                }
                            )
                finally:
                    pbar.update(1)
        finally:
            with contextlib.suppress(Exception):
                pbar.close()
            with contextlib.suppress(Exception):
                executor.shutdown(wait=False, cancel_futures=True)
    n_tr = processed_train
    n_ts = processed_test

    # Write dataset.yaml (strings as keys as per nnDetection)
    labels_map = {str(k): str(v) for k, v in (names or {}).items()} if names else None
    if not labels_map:
        # Infer number of classes from train labels if names missing
        cls_set: set[int] = set()
        for case_id in train_cases:
            for c, *_ in _labels_for_case(label_dir, case_id):
                cls_set.add(int(c))
        if cls_set:
            labels_map = {str(i): f"class_{i}" for i in sorted(cls_set)}
        else:
            labels_map = {"0": "class_0"}

    dataset_yaml = {
        "task": f"Task{task_id:03d}_{task_name}",
        "name": task_name,
        "dim": 3,
        "test_labels": bool(n_ts > 0),
        "labels": labels_map,
        "modalities": {"0": modality_name},
    }
    with (task_dir / "dataset.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False)

    # Write mapping CSV at task root
    if mapping_rows:
        mapping_csv = task_dir / "mapping.csv"
        with mapping_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "yolo_case_id",
                    "yolo_image",
                    "nndet_case_id",
                    "nndet_image",
                    "split",
                ],
            )
            writer.writeheader()
            writer.writerows(mapping_rows)

    # Write splits_final.pkl (list of 5 OrderedDicts with 'train' and 'val')
    try:
        with (dataset_dir / "data_splits.json").open("r", encoding="utf-8") as f:
            all_splits = json.load(f)
        if not isinstance(all_splits, list):
            msg = "data_splits.json malformed: expected a list"
            raise TypeError(msg)
        n_folds = len(all_splits)
        if n_folds < 5:
            LOGGER.warning("Only %d folds found in data_splits.json; writing %d.", n_folds, n_folds)
        result_list: list[OrderedDict[str, np.ndarray]] = []
        for i, fold_entry in enumerate(all_splits[:5]):
            train_list = [_sanitize_name(cid) for cid in fold_entry.get("train", [])]
            val_list = [_sanitize_name(cid) for cid in fold_entry.get("val", [])]
            result_list.append(OrderedDict([("train", train_list), ("val", val_list)]))
        splits_pkl = task_dir / "splits_final.pkl"
        with splits_pkl.open("wb") as f:
            pickle.dump(result_list, f)
    except Exception:
        LOGGER.exception("Failed to write splits_final.pkl from data_splits.json")

    LOGGER.info(
        "nnDetection conversion done: %d train cases%s -> %s",
        n_tr,
        f", {n_ts} test cases" if n_ts else "",
        task_dir,
    )
    return task_dir


__all__ = ["convert_dataset_to_nndetection"]


def _convert_one_case(
    dataset_dir_str: str, cid: str, out_img_dir_str: str, out_lbl_dir_str: str
) -> tuple[bool, str, str, list[str], str, str]:
    """Worker to convert a single case. Returns (success, split) where split is 'train' or 'test'.

    The split is inferred from target directory name.
    """
    dataset_dir = Path(dataset_dir_str)
    out_img_dir = Path(out_img_dir_str)
    out_lbl_dir = Path(out_lbl_dir_str)

    # Infer split from target images directory name
    split = "test" if out_img_dir.name.endswith("imagesTs") else "train"

    img_dir = dataset_dir / "images"
    lbl_dir = dataset_dir / "labels"
    msk_dir = dataset_dir / "masks"

    img_path = img_dir / f"{cid}.nii.gz"
    if not img_path.exists():
        return (False, split, f"{cid}.nii.gz", [], cid, _sanitize_name(cid))

    case_id = _sanitize_name(cid)
    try:
        _ensure_dir(out_img_dir)
        _ensure_dir(out_lbl_dir)

        # Load source image once
        ref_nii = nib.load(img_path)
        ref_data = np.asanyarray(ref_nii.dataobj)
        ref_shape = tuple(ref_data.shape)

        # Write one nnDetection file per channel/modality
        dst_names: list[str] = []
        if ref_data.ndim == 3:
            # Single channel
            dst_name = f"{case_id}_0000.nii.gz"
            nib.save(
                nib.Nifti1Image(
                    ref_data.astype(np.float32, copy=False),
                    affine=ref_nii.affine,
                    header=ref_nii.header,
                ),
                out_img_dir / dst_name,
            )
            dst_names.append(dst_name)
        elif ref_data.ndim == 4:
            # Could be (C,X,Y,Z) or (X,Y,Z,C)
            if ref_data.shape[0] <= 5 and ref_data.shape[0] < ref_data.shape[-1]:
                # (C,X,Y,Z)
                channels = ref_data.shape[0]
                for c in range(channels):
                    dst_name = f"{case_id}_{c:04d}.nii.gz"
                    nib.save(
                        nib.Nifti1Image(
                            ref_data[c, ...].astype(np.float32, copy=False),
                            affine=ref_nii.affine,
                            header=ref_nii.header,
                        ),
                        out_img_dir / dst_name,
                    )
                    dst_names.append(dst_name)
                # Use spatial shape for labels
                ref_shape = tuple(ref_data.shape[1:4])
            else:
                # (X,Y,Z,C)
                channels = ref_data.shape[-1]
                for c in range(channels):
                    dst_name = f"{case_id}_{c:04d}.nii.gz"
                    nib.save(
                        nib.Nifti1Image(
                            ref_data[..., c].astype(np.float32, copy=False),
                            affine=ref_nii.affine,
                            header=ref_nii.header,
                        ),
                        out_img_dir / dst_name,
                    )
                    dst_names.append(dst_name)
                ref_shape = tuple(ref_data.shape[0:3])
        else:
            # Unexpected dimensionality; copy as single channel fallback
            dst_name = f"{case_id}_0000.nii.gz"
            nib.save(
                nib.Nifti1Image(
                    ref_data.astype(np.float32, copy=False),
                    affine=ref_nii.affine,
                    header=ref_nii.header,
                ),
                out_img_dir / dst_name,
            )
            dst_names.append(dst_name)

        # Gather labels
        labels_tuples = _labels_for_case(lbl_dir, cid)
        instance_classes = [c for c, *_ in labels_tuples]
        boxes = [(xc, yc, zc, sx, sy, sz) for _, xc, yc, zc, sx, sy, sz in labels_tuples]

        # Compose labelmap from masks if possible, else from boxes
        labelmap: np.ndarray | None = None
        mask_arr = _try_load_masks(msk_dir, cid)
        if mask_arr is not None and len(instance_classes) > 0:
            labelmap_candidate = _compose_instance_seg_from_masks(mask_arr, len(instance_classes))
            if labelmap_candidate is not None and labelmap_candidate.shape == ref_shape:
                labelmap = labelmap_candidate

        if labelmap is None:
            labelmap = _compose_instance_seg_from_boxes(ref_shape, boxes)

        _write_label_files(out_lbl_dir, case_id, labelmap, instance_classes, ref_nii)
    except Exception:
        # Let caller log the exception if needed
        return (False, split, img_path.name, [], cid, case_id)
    else:
        return (True, split, img_path.name, dst_names, cid, case_id)

