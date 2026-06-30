from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import nibabel as nib
import torch

from radyolo.config.config import DefaultConfig, get_training_config, get_dataset_cfg
from radyolo.data.data import (
    PatchWiseDataset,
    WholeImageDataset,
    apply_configured_augmentations_to_batch,
    build_dataloader,
)
from radyolo.utils.ops import xyzwhd2xyzxyz

LOGGER = logging.getLogger("test_augmentation_from_dataset")

DatasetExportType = PatchWiseDataset | WholeImageDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export samples from the configured training or validation dataset pipeline"
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to training config.yaml (same as used for training)",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to dataset root (with images/, labels/, masks/, data_splits.json)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write augmented samples",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of augmented samples to export",
    )
    p.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Optional override for fold (else from config)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (affects augmentations)",
    )
    p.add_argument(
        "--dataset-type",
        choices=("auto", "wholeimage", "patchwise"),
        default="auto",
        help="Dataset implementation to export from. 'auto' uses config.training_mode.",
    )
    p.add_argument(
        "--group",
        choices=("train", "val", "test"),
        default="train",
        help="Dataset split to export from.",
    )
    augment_group = p.add_mutually_exclusive_group()
    augment_group.add_argument(
        "--augment",
        dest="augment",
        action="store_true",
        help="Force dataset augmentation on.",
    )
    augment_group.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        help="Force dataset augmentation off.",
    )
    p.set_defaults(augment=None)
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional dataloader batch size override. Defaults to config.batch_size.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional dataloader worker override. Defaults to config.workers.",
    )
    return p.parse_args()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_nifti(data: torch.Tensor, target_path: Path) -> None:
    # data expected shape (C,D,W,H) or (D,W,H); convert to numpy in original orientation.
    if data.ndim == 4 and data.shape[0] == 1:
        data = data[0]
    # Reverse permutation done during load, so save back as (W,H,D) -> nib expects (X,Y,Z)
    arr = data.permute(2, 1, 0).cpu().numpy().astype("float32")
    img = nib.Nifti1Image(arr, affine=torch.eye(4).numpy())
    nib.save(img, str(target_path))


def _save_combined_masks(masks: torch.Tensor, target_path: Path) -> str:
    """Save all instance masks into one NIfTI file.

    Input shapes encountered (after dataset augmentation) are expected to be
    (N,1,D,W,H) or (N,D,W,H). We'll convert to either:
      - 4D (N,D,W,H) uint8 if overlapping instances allowed, OR
      - 3D (D,W,H) uint16 label map if masks are non-overlapping (perfect partition).

    Heuristic: if sum over channel dimension equals at most 1 everywhere, we
    collapse to a label map (argmax + 1, background=0). Otherwise keep stacked.
    """
    if masks.ndim == 5 and masks.shape[1] == 1:  # (N,1,D,W,H) -> (N,D,W,H)
        masks = masks[:, 0]
    if masks.ndim != 4:
        msg = f"Unexpected mask tensor shape {tuple(masks.shape)}"
        raise ValueError(msg)

    masks_bin = (masks > 0.5).to(torch.uint8)
    overlap = masks_bin.sum(dim=0)  # (D,W,H)
    non_overlapping = bool(torch.all(overlap <= 1))

    if non_overlapping:
        labels = torch.zeros_like(overlap, dtype=torch.int32)
        for i in range(masks_bin.shape[0]):
            labels[masks_bin[i] == 1] = i + 1
        arr = labels.permute(2, 1, 0).cpu().numpy().astype("uint16")
    else:
        arr = masks_bin.permute(0, 3, 2, 1).cpu().numpy().astype("uint8")  # (N,W,H,D)

    img = nib.Nifti1Image(arr, affine=torch.eye(4).numpy())
    nib.save(img, str(target_path))
    return target_path.name


def _sample_stem(sample_index: int, case_name: str) -> str:
    return f"sample_{sample_index:04d}_{case_name}"


def _image_filename(sample_index: int, case_name: str) -> str:
    return f"{_sample_stem(sample_index, case_name)}_image.nii.gz"


def _boxes_filename(sample_index: int, case_name: str) -> str:
    return f"{_sample_stem(sample_index, case_name)}_boxes.json"


def _masks_filename(sample_index: int, case_name: str) -> str:
    return f"{_sample_stem(sample_index, case_name)}_masks.nii.gz"


def _resolve_dataset_type(cfg: DefaultConfig, args: argparse.Namespace) -> str:
    if args.dataset_type != "auto":
        return str(args.dataset_type)
    return str(cfg.training_mode)


def _resolve_augment(args: argparse.Namespace) -> bool:
    if args.augment is not None:
        return bool(args.augment)
    return args.group == "train"


def _serialize_ratio_pad(
    ratio_pad: tuple[tuple[float, float, float], tuple[int, int, int]] | None,
) -> dict[str, list[float] | list[int]] | None:
    if ratio_pad is None:
        return None
    ratio, pad = ratio_pad
    return {
        "ratio": [float(value) for value in ratio],
        "pad": [int(value) for value in pad],
    }


def _build_dataset(
    cfg: DefaultConfig, args: argparse.Namespace
) -> DatasetExportType:
    dataset_class = (
        PatchWiseDataset
        if _resolve_dataset_type(cfg, args) == "patchwise"
        else WholeImageDataset
    )
    return dataset_class(
        src_path=args.data_root,
        patch_size=cfg.patch_size,
        augment=_resolve_augment(args),
        min_stride=8,
        fold=args.fold if args.fold is not None else cfg.fold,
        group=args.group,
        normalization=cfg.normalization,
        load_masks=cfg.segmentation_output,
        foreground_oversampling=(
            cfg.foreground_oversampling
            if args.group == "train" or dataset_class is PatchWiseDataset
            else 0.0
        ),
        min_box_crop_fraction=cfg.min_box_crop_fraction,
        config=cfg,
    )


def _load_class_names(data_root: Path) -> dict[int, str] | None:
    dataset_cfg_path = data_root / "dataset.yaml"
    if not dataset_cfg_path.exists():
        return None

    try:
        ds_cfg = get_dataset_cfg(dataset_cfg_path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not load dataset class names: %s", exc)
        return None
    return ds_cfg.names


def _apply_remaining_batch_augmentations(
    batch: dict[str, Any],
    cfg: DefaultConfig,
    *,
    augment: bool,
) -> dict[str, Any]:
    if not augment or not cfg.uses_gpu_augmentations():
        return batch

    # Mirror the training pipeline's second augmentation stage. In training these
    # augmentations are assigned to "gpu", but they operate on whichever device
    # the tensors live on, so we can run them on CPU for sample export.
    apply_configured_augmentations_to_batch(
        batch,
        cfg,
        cfg.patch_size,
        placement="gpu",
    )
    return batch


def _write_batch_samples(
    export_batch: dict[str, Any],
    *,
    args: argparse.Namespace,
    dataset: DatasetExportType,
    names: dict[int, str] | None,
    output_dir: Path,
    total_written: int,
) -> int:
    imgs: torch.Tensor = export_batch["img"]
    classes: torch.Tensor = export_batch["cls"]
    bboxes: torch.Tensor = export_batch["bboxes"]
    batch_idx: torch.Tensor = export_batch["batch_idx"]
    masks_tuple = export_batch["masks"]
    case_names = export_batch["case_names"]
    shapes = export_batch["shape"]
    original_shapes = export_batch["ori_shape"]
    ratio_pads = export_batch["ratio_pad"]

    for image_index in range(imgs.shape[0]):
        if total_written >= args.num_samples:
            break

        case_name = case_names[image_index]
        sample_index = total_written
        image_path = output_dir / _image_filename(sample_index, case_name)
        _save_nifti(imgs[image_index], image_path)

        sample_mask = batch_idx == image_index
        cls_i = classes[sample_mask]
        box_whd_norm = bboxes[sample_mask]
        image_shape_xyz = tuple(int(dim) for dim in reversed(imgs[image_index].shape[-3:]))
        scale_xyzxyz = box_whd_norm.new_tensor((*image_shape_xyz, *image_shape_xyz))
        box_xyzxyz_img = xyzwhd2xyzxyz(box_whd_norm) * scale_xyzxyz

        boxes_json: list[dict[str, Any]] = []
        for box_index in range(box_whd_norm.shape[0]):
            class_id = int(cls_i[box_index].item())
            class_name = names.get(class_id, f"class_{class_id}") if names else f"class_{class_id}"
            whd_norm = box_whd_norm[box_index].tolist()
            xyzxyz_img = box_xyzxyz_img[box_index].tolist()
            boxes_json.append(
                {
                    "class": class_id,
                    "class_name": class_name,
                    "box_xyzwhd_normalized": [round(float(x), 6) for x in whd_norm],
                    "box_xyzxyz_image": [round(float(x), 2) for x in xyzxyz_img],
                }
            )

        masks_i = masks_tuple[image_index]
        mask_file: str | None = None
        if masks_i is not None:
            try:
                mask_file = _save_combined_masks(
                    masks_i,
                    output_dir / _masks_filename(sample_index, case_name),
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed saving masks for %s: %s", case_name, exc)

        out_boxes_path = output_dir / _boxes_filename(sample_index, case_name)
        with out_boxes_path.open("w") as file:
            json.dump(
                {
                    "sample_index": sample_index,
                    "case_name": case_name,
                    "dataset_type": type(dataset).__name__,
                    "group": args.group,
                    "augment": bool(dataset.augment),
                    "image_shape": [int(v) for v in shapes[image_index]],
                    "original_shape": [int(v) for v in original_shapes[image_index]],
                    "ratio_pad": _serialize_ratio_pad(ratio_pads[image_index]),
                    "num_boxes": len(boxes_json),
                    "boxes": boxes_json,
                    "mask_path": mask_file,
                    "image_path": image_path.name,
                },
                file,
                indent=2,
            )

        total_written += 1
        LOGGER.info(
            "Wrote augmented sample %d/%d: %s",
            total_written,
            args.num_samples,
            case_name,
        )

    return total_written


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    config = get_training_config(args.config, overrides=None)
    if args.fold is not None:
        config.fold = args.fold
    names = _load_class_names(args.data_root)

    dataset = _build_dataset(config, args)
    LOGGER.info(
        "Exporting %s samples from %s split using %s (augment=%s)",
        args.num_samples,
        args.group,
        type(dataset).__name__,
        dataset.augment,
    )
    loader = build_dataloader(
        dataset,
        batch_size=args.batch_size or config.batch_size,
        workers=args.workers if args.workers is not None else config.workers,
        shuffle=args.group == "train",
        device=torch.device("cpu"),
    )

    _ensure_dir(args.output_dir)

    total_written = 0
    for batch in loader:
        if total_written >= args.num_samples:
            break
        export_batch = _apply_remaining_batch_augmentations(
            batch,
            config,
            augment=bool(dataset.augment),
        )
        total_written = _write_batch_samples(
            export_batch,
            args=args,
            dataset=dataset,
            names=names,
            output_dir=args.output_dir,
            total_written=total_written,
        )

    LOGGER.info(
        "Finished writing %d augmented samples to %s", total_written, args.output_dir
    )
    return 0


"""Generate and persist augmented training samples.

This script reuses the normal dataset and augmentation pipeline to produce image
volumes (as NIfTI), corresponding segmentation masks (if any), and bounding
boxes (JSON) exactly as they would appear when sampled from a dataloader.

Usage (example):
    python -m scripts.test_augmentation_from_dataset \
        --config path/to/config.yaml \
        --data-root path/to/dataset_root \
        --output-dir out/aug_samples \
        --num-samples 20 \
        --dataset-type patchwise \
        --group train \
        --augment \
        --fold 0

Validation example:
    python -m scripts.test_augmentation_from_dataset \
        --config path/to/config.yaml \
        --data-root path/to/dataset_root \
        --output-dir out/val_samples \
        --dataset-type patchwise \
        --group val \
        --no-augment

The dataset root must contain the same structure used for training
 (images/*.nii.gz, labels/*.txt, masks/*.nii.gz, data_splits.json).

The resulting flat directory layout now uses unique sample indices, so repeated
case names from patchwise sampling do not overwrite earlier exports:
    out/aug_samples/
            sample_0000_case_0000_image.nii.gz
            sample_0000_case_0000_boxes.json
            sample_0000_case_0000_masks.nii.gz
            sample_0001_case_0000_image.nii.gz
            sample_0001_case_0000_boxes.json
            sample_0001_case_0000_masks.nii.gz
            ...

Notes:
- Bounding boxes are saved in multiple representations:
    * xyzwhd (normalized as stored in labels after augmentation) as floats
    * xyzxyz_image (voxel coordinates in exported image space)
- Class names are taken from dataset config if available, else fallback to class_<idx>.
- Masks (if present) are exported as ONE NIfTI file per case. Depending on the
    shape, either:
        * (N,D,W,H) if individual binary instance masks are available, or
        * (D,W,H) single label map (if masks are mutually exclusive and reducible).
- JSON metadata includes the sampled split, whether augmentation was enabled,
  the sampled shape/original_shape, and ratio_pad.
"""
if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
