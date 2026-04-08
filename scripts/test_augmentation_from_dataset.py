from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import nibabel as nib
import torch

from radyolo.config.config import DefaultConfig, get_cfg, get_dataset_cfg
from radyolo.data.data import WholeImageDataset, build_dataloader
from radyolo.utils.ops import xyzwhd2xyzxyz

LOGGER = logging.getLogger("test_augmentation_from_dataset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export augmented training samples using existing pipeline"
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


def _image_filename(case: str) -> str:
    return f"{case}_image.nii.gz"


def _boxes_filename(case: str) -> str:
    return f"{case}_boxes.json"


def _masks_filename(case: str) -> str:
    return f"{case}_masks.nii.gz"


def _build_dataset(
    cfg: DefaultConfig, args: argparse.Namespace
) -> WholeImageDataset:
    return WholeImageDataset(
        src_path=args.data_root,
        patch_size=cfg.patch_size,
        augment=True,
        min_stride=8,
        fold=args.fold if args.fold is not None else cfg.fold,
        group="train",
        normalization=cfg.normalization,
        load_masks=cfg.segmentation_output,
        config=cfg,
    )


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    config = get_cfg(args.config, overrides=None)
    if args.fold is not None:
        config.fold = args.fold

    # Try to derive class names if dataset.yaml (or similar) is available
    dataset_cfg_path = args.data_root / "dataset.yaml"
    if dataset_cfg_path.exists():
        try:
            ds_cfg = get_dataset_cfg(dataset_cfg_path)
            names = ds_cfg.names
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("Could not load dataset class names: %s", e)

    dataset = _build_dataset(config, args)
    loader = build_dataloader(
        dataset,
        batch_size=2,
        workers=4,
        shuffle=True,
        device=torch.device("cpu"),
    )

    _ensure_dir(args.output_dir)

    total_written = 0
    for batch in loader:
        if total_written >= args.num_samples:
            break
        imgs: torch.Tensor = batch["img"]  # (B,C,D,W,H)
        classes: torch.Tensor = batch["cls"]
        bboxes: torch.Tensor = batch["bboxes"]  # (N,6) normalized xyzwhd
        batch_idx: torch.Tensor = batch["batch_idx"]  # (N,)
        masks_tuple = batch["masks"]  # tuple of per-image masks (maybe None)
        case_names = batch["case_names"]

        # Convert boxes to xyzxyz image-space (resized) for convenience
        bboxes_xyzxyz = xyzwhd2xyzxyz(bboxes)

        for i_img in range(imgs.shape[0]):
            if total_written >= args.num_samples:
                break
            case_name = case_names[i_img]
            image_path = args.output_dir / _image_filename(case_name)
            _save_nifti(imgs[i_img], image_path)

            # Gather boxes for this image
            mask = batch_idx == i_img
            cls_i = classes[mask]
            box_whd_norm = bboxes[mask]
            box_xyzxyz_img = bboxes_xyzxyz[mask]

            boxes_json: list[dict[str, Any]] = []
            for j in range(box_whd_norm.shape[0]):
                cid = int(cls_i[j].item())
                name = names.get(cid, f"class_{cid}") if names else f"class_{cid}"
                whd_norm = box_whd_norm[j].tolist()
                xyzxyz_img = box_xyzxyz_img[j].tolist()
                boxes_json.append(
                    {
                        "class": cid,
                        "class_name": name,
                        "box_xyzwhd_normalized": [round(float(x), 6) for x in whd_norm],
                        "box_xyzxyz_image": [round(float(x), 2) for x in xyzxyz_img],
                    }
                )

            masks_i = masks_tuple[i_img]
            mask_file: str | None = None
            if masks_i is not None:
                try:
                    mask_file = _save_combined_masks(
                        masks_i, args.output_dir / _masks_filename(case_name)
                    )
                except Exception as e:  # noqa: BLE001
                    LOGGER.warning("Failed saving masks for %s: %s", case_name, e)

            out_boxes_path = args.output_dir / _boxes_filename(case_name)
            with out_boxes_path.open("w") as f:
                json.dump(
                    {
                        "case_name": case_name,
                        "num_boxes": len(boxes_json),
                        "boxes": boxes_json,
                        "mask_path": mask_file,
                        "image_path": image_path.name,
                    },
                    f,
                    indent=2,
                )

            total_written += 1
            LOGGER.info(
                "Wrote augmented sample %d/%d: %s",
                total_written,
                args.num_samples,
                case_name,
            )

        if total_written >= args.num_samples:
            break

    LOGGER.info(
        "Finished writing %d augmented samples to %s", total_written, args.output_dir
    )
    return 0


"""Generate and persist augmented training samples.

This script reuses the normal training dataset & augmentation pipeline to produce
augmented image volumes (as NIfTI), corresponding segmentation masks (if any),
and bounding boxes (JSON) exactly as they would appear *after* augmentation in
an epoch.

Usage (example):
    python -m scripts.test_augmentation_from_dataset \
        --config path/to/config.yaml \
        --data-root path/to/dataset_root \
        --output-dir out/aug_samples \
        --num-samples 20 \
        --fold 0

The dataset root must contain the same structure used for training
 (images/*.nii.gz, labels/*.txt, masks/*.nii.gz, data_splits.json).

The resulting (flat) directory layout now:
    out/aug_samples/
            case_0000_image.nii.gz
            case_0000_boxes.json
            case_0000_masks.nii.gz   (if segmentation present; single 4D or 3D label volume)
            case_0001_image.nii.gz
            case_0001_boxes.json
            case_0001_masks.nii.gz
            ...

Notes:
- Bounding boxes are saved in multiple representations:
    * xyzwhd (normalized as stored in labels after augmentation) as floats
    * xyzxyz_image (pixel/voxel coordinates in resized/augmented space)
- Class names are taken from dataset config if available, else fallback to class_<idx>.
- Masks (if present) are exported as ONE NIfTI file per case. Depending on the
    shape, either:
        * (N,D,W,H) if individual binary instance masks are available, or
        * (D,W,H) single label map (if masks are mutually exclusive and reducible).
"""
if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
