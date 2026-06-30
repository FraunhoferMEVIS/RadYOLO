from __future__ import annotations

import contextlib
import csv
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
from tqdm.auto import tqdm

from radyolo.data.convert_to_nndetection import (
    _labels_for_case,
    _load_dataset_names,
    _read_splits,
    _sanitize_name,
    _try_load_masks,
)
from radyolo.data.data_splits import DEFAULT_SPLIT_FILE_NAME
from radyolo.data.mask_compositing import (
    compose_labelmap_from_boxes as _compose_semantic_seg_from_boxes,
)
from radyolo.data.mask_compositing import (
    compose_labelmap_from_masks as _compose_semantic_seg_from_masks,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

LOGGER = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_label_file(
    out_labels_dir: Path,
    case_id: str,
    labelmap: np.ndarray,
    ref_nii: nib.Nifti1Image,
) -> None:
    _ensure_dir(out_labels_dir)
    label_path = out_labels_dir / f"{case_id}.nii.gz"
    label_img = nib.Nifti1Image(
        labelmap.astype(np.uint16, copy=False), affine=ref_nii.affine, header=ref_nii.header
    )
    label_img.set_data_dtype(np.uint16)
    nib.save(label_img, label_path)


def _infer_channel_count(dataset_dir: Path, case_ids: Iterable[str]) -> int:
    image_dir = dataset_dir / "images"
    for case_id in case_ids:
        image_path = image_dir / f"{case_id}.nii.gz"
        if not image_path.exists():
            continue
        image = nib.load(image_path)
        image_data = np.asanyarray(image.dataobj)
        if image_data.ndim != 4:
            return 1
        if image_data.shape[0] <= 5 and image_data.shape[0] < image_data.shape[-1]:
            return int(image_data.shape[0])
        return int(image_data.shape[-1])
    return 1


def _build_label_map(
    dataset_names: dict[int, str] | None,
    dataset_dir: Path,
    train_cases: Iterable[str],
) -> dict[str, int]:
    labels: dict[str, int] = {"background": 0}
    if dataset_names:
        for class_id, class_name in sorted(dataset_names.items()):
            labels[str(class_name)] = int(class_id) + 1
        return labels

    class_ids: set[int] = set()
    label_dir = dataset_dir / "labels"
    for case_id in train_cases:
        for class_id, *_ in _labels_for_case(label_dir, case_id):
            class_ids.add(int(class_id))

    for class_id in sorted(class_ids):
        labels[f"class_{class_id}"] = class_id + 1
    return labels


def _write_split_file(
    dataset_dir: Path,
    out_dataset_dir: Path,
    input_split_file_name: str,
    output_split_file_name: str,
    include_val_in_train: bool,
) -> None:
    with (dataset_dir / input_split_file_name).open("r", encoding="utf-8") as file_handle:
        all_splits = json.load(file_handle)
    if not isinstance(all_splits, list):
        msg = f"{input_split_file_name} malformed: expected a list"
        raise TypeError(msg)

    output_splits: list[dict[str, list[str]]] = []
    for split_entry in all_splits:
        train_list = sorted({_sanitize_name(case_id) for case_id in split_entry.get("train", [])})
        val_source = sorted({_sanitize_name(case_id) for case_id in split_entry.get("val", [])})
        if val_source and not include_val_in_train:
            LOGGER.warning(
                "Validation cases are omitted from exported %s because --no-include-val-in-train "
                "was requested.",
                output_split_file_name,
            )
        output_splits.append(
            {
                "train": train_list,
                "val": val_source if include_val_in_train else [],
            }
        )

    with (out_dataset_dir / output_split_file_name).open("w", encoding="utf-8") as file_handle:
        json.dump(output_splits, file_handle, indent=2)


def convert_dataset_to_nnunet(
    dataset_dir: Path,
    out_root: Path,
    dataset_id: int = 100,
    dataset_name: str | None = None,
    fold: int = 0,
    include_val_in_train: bool = True,
    modality_name: str = "CT",
    workers: int = 0,
    split_file_name: str = DEFAULT_SPLIT_FILE_NAME,
    export_split_file_name: str = "splits_final.json",
) -> Path:
    """Convert a RadYOLO dataset to nnU-Net raw dataset format."""
    dataset_dir = dataset_dir.resolve()
    out_root = out_root.resolve()
    if dataset_name is None:
        dataset_name = _sanitize_name(dataset_dir.name)
    dataset_name = _sanitize_name(dataset_name)
    out_dataset_dir = out_root / f"Dataset{dataset_id:03d}_{dataset_name}"

    images_tr_dir = out_dataset_dir / "imagesTr"
    labels_tr_dir = out_dataset_dir / "labelsTr"
    images_ts_dir = out_dataset_dir / "imagesTs"
    labels_ts_dir = out_dataset_dir / "labelsTs"

    _ensure_dir(images_tr_dir)
    _ensure_dir(labels_tr_dir)

    dataset_names = _load_dataset_names(dataset_dir / "dataset.yaml")
    splits = _read_splits(dataset_dir, fold=fold, split_file_name=split_file_name)

    train_cases: list[str] = list(splits.train)
    if include_val_in_train:
        train_cases += list(splits.val)
    train_cases = sorted(set(train_cases))

    total_cases = len(train_cases) + len(splits.test)
    max_workers = workers if workers and workers > 0 else (os.cpu_count() or 1)

    def _submit_cases(
        executor: ProcessPoolExecutor,
        cases: Iterable[str],
        out_image_dir: Path,
        out_label_dir: Path,
    ) -> dict:
        futures = {}
        for case_id in cases:
            future = executor.submit(
                _convert_one_case,
                str(dataset_dir),
                case_id,
                str(out_image_dir),
                str(out_label_dir),
            )
            futures[future] = case_id
        return futures

    processed_train = 0
    processed_test = 0
    mapping_rows: list[dict[str, str]] = []
    observed_channel_counts: set[int] = set()
    if total_cases > 0:
        pbar = tqdm(total=total_cases, desc="Converting", unit="case", smoothing=0.05)
        executor = ProcessPoolExecutor(max_workers=max_workers)
        try:
            futures = {}
            futures.update(_submit_cases(executor, train_cases, images_tr_dir, labels_tr_dir))
            if splits.test:
                _ensure_dir(images_ts_dir)
                _ensure_dir(labels_ts_dir)
                futures.update(_submit_cases(executor, splits.test, images_ts_dir, labels_ts_dir))

            for future in as_completed(futures):
                case_id = futures[future]
                try:
                    (
                        ok,
                        split,
                        src_image_name,
                        dst_image_names,
                        src_case_id,
                        dst_case_id,
                        channel_count,
                    ) = future.result()
                except Exception:
                    LOGGER.exception("Failed to convert case %s", case_id)
                else:
                    if ok:
                        observed_channel_counts.add(channel_count)
                        if split == "train":
                            processed_train += 1
                        elif split == "test":
                            processed_test += 1
                        for dst_name in dst_image_names:
                            mapping_rows.append(
                                {
                                    "yolo_case_id": src_case_id,
                                    "yolo_image": src_image_name,
                                    "nnunet_case_id": dst_case_id,
                                    "nnunet_image": dst_name,
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

    channel_count = (
        max(observed_channel_counts)
        if observed_channel_counts
        else _infer_channel_count(dataset_dir, [*train_cases, *splits.test])
    )
    if len(observed_channel_counts) > 1:
        LOGGER.warning(
            "Observed inconsistent channel counts across cases: %s. Writing channel_names for %d "
            "channels.",
            sorted(observed_channel_counts),
            channel_count,
        )

    channel_names = {
        str(channel_index): (
            modality_name if channel_index == 0 else f"{modality_name}_{channel_index}"
        )
        for channel_index in range(channel_count)
    }
    dataset_json = {
        "channel_names": channel_names,
        "labels": _build_label_map(dataset_names, dataset_dir, train_cases),
        "numTraining": processed_train,
        "file_ending": ".nii.gz",
    }
    with (out_dataset_dir / "dataset.json").open("w", encoding="utf-8") as file_handle:
        json.dump(dataset_json, file_handle, indent=2)

    if mapping_rows:
        mapping_csv = out_dataset_dir / "mapping.csv"
        with mapping_csv.open("w", encoding="utf-8", newline="") as file_handle:
            writer = csv.DictWriter(
                file_handle,
                fieldnames=[
                    "yolo_case_id",
                    "yolo_image",
                    "nnunet_case_id",
                    "nnunet_image",
                    "split",
                ],
            )
            writer.writeheader()
            writer.writerows(mapping_rows)

    try:
        _write_split_file(
            dataset_dir,
            out_dataset_dir,
            input_split_file_name=split_file_name,
            output_split_file_name=export_split_file_name,
            include_val_in_train=include_val_in_train,
        )
    except FileNotFoundError:
        LOGGER.warning(
            "Input split file %s not found; skipped exporting %s.",
            split_file_name,
            export_split_file_name,
        )
    except Exception:
        LOGGER.exception("Failed to write %s from %s", export_split_file_name, split_file_name)

    LOGGER.info(
        "nnU-Net conversion done: %d training cases%s -> %s",
        processed_train,
        f", {processed_test} test cases" if processed_test else "",
        out_dataset_dir,
    )
    return out_dataset_dir


def _convert_one_case(
    dataset_dir_str: str,
    case_id: str,
    out_image_dir_str: str,
    out_label_dir_str: str,
) -> tuple[bool, str, str, list[str], str, str, int]:
    dataset_dir = Path(dataset_dir_str)
    out_image_dir = Path(out_image_dir_str)
    out_label_dir = Path(out_label_dir_str)

    split = "test" if out_image_dir.name.endswith("imagesTs") else "train"
    image_path = dataset_dir / "images" / f"{case_id}.nii.gz"
    if not image_path.exists():
        return (False, split, f"{case_id}.nii.gz", [], case_id, _sanitize_name(case_id), 0)

    sanitized_case_id = _sanitize_name(case_id)
    try:
        _ensure_dir(out_image_dir)
        _ensure_dir(out_label_dir)

        ref_nii = nib.load(image_path)
        ref_data = np.asanyarray(ref_nii.dataobj)
        ref_shape = tuple(ref_data.shape[:3]) if ref_data.ndim >= 3 else tuple(ref_data.shape)

        dst_names: list[str] = []
        channel_count = 1
        if ref_data.ndim == 3:
            dst_name = f"{sanitized_case_id}_0000.nii.gz"
            nib.save(
                nib.Nifti1Image(
                    ref_data.astype(np.float32, copy=False),
                    affine=ref_nii.affine,
                    header=ref_nii.header,
                ),
                out_image_dir / dst_name,
            )
            dst_names.append(dst_name)
        elif ref_data.ndim == 4:
            if ref_data.shape[0] <= 5 and ref_data.shape[0] < ref_data.shape[-1]:
                channel_count = int(ref_data.shape[0])
                ref_shape = tuple(ref_data.shape[1:4])
                for channel_index in range(channel_count):
                    dst_name = f"{sanitized_case_id}_{channel_index:04d}.nii.gz"
                    nib.save(
                        nib.Nifti1Image(
                            ref_data[channel_index, ...].astype(np.float32, copy=False),
                            affine=ref_nii.affine,
                            header=ref_nii.header,
                        ),
                        out_image_dir / dst_name,
                    )
                    dst_names.append(dst_name)
            else:
                channel_count = int(ref_data.shape[-1])
                ref_shape = tuple(ref_data.shape[0:3])
                for channel_index in range(channel_count):
                    dst_name = f"{sanitized_case_id}_{channel_index:04d}.nii.gz"
                    nib.save(
                        nib.Nifti1Image(
                            ref_data[..., channel_index].astype(np.float32, copy=False),
                            affine=ref_nii.affine,
                            header=ref_nii.header,
                        ),
                        out_image_dir / dst_name,
                    )
                    dst_names.append(dst_name)
        else:
            dst_name = f"{sanitized_case_id}_0000.nii.gz"
            nib.save(
                nib.Nifti1Image(
                    ref_data.astype(np.float32, copy=False),
                    affine=ref_nii.affine,
                    header=ref_nii.header,
                ),
                out_image_dir / dst_name,
            )
            dst_names.append(dst_name)

        labels_tuples = _labels_for_case(dataset_dir / "labels", case_id)
        mask_arr = _try_load_masks(dataset_dir / "masks", case_id)
        labelmap = (
            _compose_semantic_seg_from_masks(mask_arr, labels_tuples)
            if mask_arr is not None
            else None
        )
        if labelmap is None:
            labelmap = _compose_semantic_seg_from_boxes(ref_shape, labels_tuples)
        _write_label_file(out_label_dir, sanitized_case_id, labelmap, ref_nii)
    except Exception:
        return (False, split, image_path.name, [], case_id, sanitized_case_id, 0)
    return (True, split, image_path.name, dst_names, case_id, sanitized_case_id, channel_count)
