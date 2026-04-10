from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from radyolo.config.config import DefaultConfig, get_cfg, get_dataset_cfg
from radyolo.config.spatial import PatchSize
from radyolo.data.data import (
    PatchWiseDataset,
    WholeImageDataset,
    apply_configured_augmentations_to_batch,
    build_dataloader,
)
from radyolo.utils.ops import xyzwhd2xyzxyz
from radyolo.validation.plotting import (
    Annotator,
    _annotate_projected_boxes,
    _get_case_plot_volume,
    _project_case_volume,
    colors,
)

LOGGER = logging.getLogger("visualize_preprocessed_patches")

DatasetType = Literal["whole-image", "patch-wise"]
Plane = Literal["xy", "xz", "yz"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize samples from a preprocessed RadYOLO dataset using the existing dataloader "
            "and validation plotting helpers."
        )
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to the dataset root containing images/, labels/ and data_splits.json.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional training config used to match augmentation and normalization settings.",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["whole-image", "patch-wise"],
        required=True,
        help="Whether to visualize full-image samples or random patch-wise samples.",
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
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory where projection plots and annotated slice PNGs are written.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Maximum number of samples to visualize.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size used by the dataloader.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
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
        default="val",
        help="Data split group to visualize.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataloader order before visualization.",
    )
    augment_group = parser.add_mutually_exclusive_group()
    augment_group.add_argument(
        "--augment",
        dest="augment",
        action="store_true",
        help="Enable dataset augmentations. Requires --config to reproduce training settings.",
    )
    augment_group.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        help="Disable dataset augmentations.",
    )
    parser.set_defaults(augment=False)
    parser.add_argument(
        "--load-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load segmentation masks if present in the dataset.",
    )
    parser.add_argument(
        "--slices-per-plane",
        type=int,
        default=3,
        help="How many annotated slices to save for each of the XY, XZ and YZ planes.",
    )
    parser.add_argument(
        "--projection-mode",
        type=str,
        choices=["max", "mean"],
        default="max",
        help="Projection reduction used for the overview figure.",
    )
    parser.add_argument(
        "--image-mode",
        type=str,
        choices=["default", "channel_difference"],
        default="default",
        help="Image mode passed through to validation plotting.",
    )
    parser.add_argument(
        "--difference-channels",
        nargs=2,
        type=int,
        metavar=("MINUEND", "SUBTRAHEND"),
        default=(2, 0),
        help="Channels used when --image-mode=channel_difference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for patch-wise sampling and dataloader shuffling.",
    )
    return parser.parse_args()


def _load_class_names(dataset_path: Path) -> dict[int, str] | None:
    dataset_cfg_path = dataset_path / "dataset.yaml"
    if not dataset_cfg_path.exists():
        return None

    try:
        return get_dataset_cfg(dataset_cfg_path).names
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not load class names from %s: %s", dataset_cfg_path, exc)
        return None


def _build_dataset(
    args: argparse.Namespace,
    config: DefaultConfig | None,
) -> WholeImageDataset | PatchWiseDataset:
    patch_size = PatchSize.from_xyz(tuple(args.patch_size))
    if config is not None and config.patch_size != patch_size:
        LOGGER.warning(
            "CLI patch size %s differs from config patch size %s; using config patch size.",
            patch_size.get_xyz(),
            config.patch_size.get_xyz(),
        )
        patch_size = config.patch_size

    dataset_kwargs = {
        "src_path": args.dataset_path,
        "patch_size": patch_size,
        "augment": bool(args.augment),
        "fold": args.fold,
        "group": args.group,
        "normalization": {"type": "none"} if config is None else config.normalization,
        "load_masks": args.load_masks,
        "config": config,
    }
    if args.dataset_type == "whole-image":
        return WholeImageDataset(**dataset_kwargs)
    if config is not None:
        dataset_kwargs["foreground_oversampling"] = (
            config.foreground_oversampling if args.group == "train" else 0.0
        )
        dataset_kwargs["min_box_crop_fraction"] = config.min_box_crop_fraction
    return PatchWiseDataset(**dataset_kwargs)


def _apply_remaining_batch_augmentations(
    batch: dict,
    config: DefaultConfig | None,
    *,
    augment: bool,
) -> dict:
    if config is None or not augment or not config.uses_gpu_augmentations():
        return batch

    apply_configured_augmentations_to_batch(
        batch,
        config,
        config.patch_size,
        placement="gpu",
    )
    return batch


def _denormalize_boxes_xyzxyz(image: torch.Tensor, bboxes_xyzwhd: torch.Tensor) -> torch.Tensor:
    if bboxes_xyzwhd.numel() == 0:
        return bboxes_xyzwhd.new_empty((0, 6))

    image_shape_xyz = tuple(reversed([int(dim) for dim in image.shape[-3:]]))
    scale = bboxes_xyzwhd.new_tensor((*image_shape_xyz, *image_shape_xyz))
    return xyzwhd2xyzxyz(bboxes_xyzwhd.to(dtype=torch.float32)) * scale


def _expand_masks_to_instances(masks: torch.Tensor | None) -> torch.Tensor | None:
    if masks is None:
        return None

    mask_tensor = masks.detach().to(device="cpu")
    if mask_tensor.ndim == 5 and mask_tensor.shape[1] == 1:
        mask_tensor = mask_tensor[:, 0]
    elif mask_tensor.ndim == 3:
        mask_tensor = mask_tensor.unsqueeze(0)

    if mask_tensor.ndim != 4:
        msg = f"Expected masks with shape (N, D, H, W) or (D, H, W), got {tuple(mask_tensor.shape)}"
        raise ValueError(msg)

    if mask_tensor.shape[0] == 1:
        label_map = mask_tensor[0]
        if label_map.dtype == torch.bool or label_map.max().item() <= 1:
            return (mask_tensor > 0).to(dtype=torch.uint8)

        instance_ids = torch.unique(label_map)
        instance_ids = instance_ids[instance_ids > 0]
        if instance_ids.numel() == 0:
            return torch.zeros((0, *label_map.shape), dtype=torch.uint8)
        return torch.stack([(label_map == instance_id) for instance_id in instance_ids], dim=0).to(
            dtype=torch.uint8
        )

    return (mask_tensor > 0).to(dtype=torch.uint8)


def _compute_display_scale(volume: torch.Tensor, upper_quantile: float = 0.995) -> float:
    finite_values = volume[torch.isfinite(volume)]
    if finite_values.numel() == 0:
        return 1.0

    scale = torch.quantile(finite_values.to(dtype=torch.float32), upper_quantile)
    if not torch.isfinite(scale) or scale <= 0:
        scale = finite_values.max()
    if not torch.isfinite(scale) or scale <= 0:
        return 1.0
    return float(scale.item())


def _make_rgb_slice(slice_2d: torch.Tensor, scale: float) -> np.ndarray:
    normalized = slice_2d.to(dtype=torch.float32).div(scale).clamp(0, 1).cpu().numpy()
    grayscale = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    return np.repeat(grayscale[..., None], 3, axis=2)


def _slice_scores(
    plane: Plane,
    size: int,
    boxes_xyzxyz: torch.Tensor,
    masks: torch.Tensor | None,
) -> torch.Tensor:
    scores = torch.zeros(size, dtype=torch.float32)
    if boxes_xyzxyz.numel():
        axis_index = {"xy": 2, "xz": 1, "yz": 0}[plane]
        projected_indices = {"xy": (0, 1, 3, 4), "xz": (0, 2, 3, 5), "yz": (1, 2, 4, 5)}[plane]
        for box in boxes_xyzxyz.to(dtype=torch.float32):
            start = max(int(torch.floor(box[axis_index]).item()), 0)
            end = min(int(torch.ceil(box[axis_index + 3]).item()), size)
            if end <= start:
                continue
            area = max(
                float((box[projected_indices[2]] - box[projected_indices[0]]).item()),
                1.0,
            ) * max(
                float((box[projected_indices[3]] - box[projected_indices[1]]).item()),
                1.0,
            )
            scores[start:end] += area

    if masks is not None and masks.numel():
        reduction_dims = {"xy": (2, 3), "xz": (1, 3), "yz": (1, 2)}[plane]
        mask_scores = masks.to(dtype=torch.float32).sum(dim=reduction_dims).sum(dim=0)
        scores += mask_scores[:size]

    return scores


def _fallback_slice_indices(size: int, count: int) -> list[int]:
    if size <= 0 or count <= 0:
        return []
    if count >= size:
        return list(range(size))
    return sorted({int(index) for index in np.linspace(0, size - 1, num=count)})


def _select_slice_indices(
    plane: Plane,
    size: int,
    boxes_xyzxyz: torch.Tensor,
    masks: torch.Tensor | None,
    count: int,
) -> list[int]:
    if size <= 0 or count <= 0:
        return []

    scores = _slice_scores(plane, size, boxes_xyzxyz, masks)
    nonzero_indices = torch.nonzero(scores > 0, as_tuple=False).flatten()
    if nonzero_indices.numel() == 0:
        return _fallback_slice_indices(size, count)

    k = min(count, int(nonzero_indices.numel()))
    ranked = torch.argsort(scores, descending=True, stable=True)
    selected = sorted(int(index) for index in ranked[:k].tolist() if scores[index] > 0)
    if len(selected) < count:
        for fallback_index in _fallback_slice_indices(size, count):
            if fallback_index not in selected:
                selected.append(fallback_index)
            if len(selected) == count:
                break
    return sorted(selected[:count])


def _boxes_for_slice(
    boxes_xyzxyz: torch.Tensor,
    classes: torch.Tensor,
    plane: Plane,
    slice_index: int,
) -> list[tuple[tuple[float, float, float, float], int]]:
    plane_spec = {
        "xy": {"depth": (2, 5), "box": (0, 1, 3, 4)},
        "xz": {"depth": (1, 4), "box": (0, 2, 3, 5)},
        "yz": {"depth": (0, 3), "box": (1, 2, 4, 5)},
    }[plane]
    slice_boxes: list[tuple[tuple[float, float, float, float], int]] = []
    for box, class_index in zip(boxes_xyzxyz.tolist(), classes.tolist(), strict=False):
        start = int(np.floor(box[plane_spec["depth"][0]]))
        end = int(np.ceil(box[plane_spec["depth"][1]]))
        if slice_index < start or slice_index >= end:
            continue
        x1, y1, x2, y2 = (box[index] for index in plane_spec["box"])
        slice_boxes.append(((x1, y1, x2, y2), int(class_index)))
    return slice_boxes


def _mask_slices_for_plane(
    masks: torch.Tensor | None,
    plane: Plane,
    slice_index: int,
) -> list[np.ndarray]:
    if masks is None or not masks.numel():
        return []
    if plane == "xy":
        return [mask[slice_index].cpu().numpy().astype(bool) for mask in masks]
    if plane == "xz":
        return [mask[:, slice_index, :].cpu().numpy().astype(bool) for mask in masks]
    return [mask[:, :, slice_index].cpu().numpy().astype(bool) for mask in masks]


def _render_annotated_slice(  # noqa: PLR0913
    base_volume: torch.Tensor,
    display_scale: float,
    boxes_xyzxyz: torch.Tensor,
    classes: torch.Tensor,
    masks: torch.Tensor | None,
    plane: Plane,
    slice_index: int,
    names: dict[int, str] | None,
) -> np.ndarray:
    if plane == "xy":
        slice_2d = base_volume[slice_index]
    elif plane == "xz":
        slice_2d = base_volume[:, slice_index, :]
    else:
        slice_2d = base_volume[:, :, slice_index]

    image = _make_rgb_slice(slice_2d, display_scale)
    mask_slices = _mask_slices_for_plane(masks, plane, slice_index)
    if mask_slices:
        overlay = image.astype(np.float32)
        use_box_classes = len(mask_slices) == int(classes.numel())
        for mask_index, mask_slice in enumerate(mask_slices):
            color_index = int(classes[mask_index].item()) if use_box_classes else mask_index
            color = np.asarray(colors(color_index), dtype=np.float32)
            overlay[mask_slice] = overlay[mask_slice] * 0.4 + color * 0.6
        image = np.clip(overlay, 0, 255).astype(np.uint8)

    annotator = Annotator(np.ascontiguousarray(image), pil=True)
    for box, class_index in _boxes_for_slice(boxes_xyzxyz, classes, plane, slice_index):
        label = names.get(class_index, str(class_index)) if names else str(class_index)
        annotator.box_label(box, label=label, color=colors(class_index))
    annotator.text(
        [6, 6],
        f"{plane.upper()} slice {slice_index}",
        txt_color=(255, 255, 255),
        box_style=True,
    )
    return annotator.result()


def _save_projection_overview(  # noqa: PLR0913
    image: torch.Tensor,
    boxes_xyzxyz: torch.Tensor,
    classes: torch.Tensor,
    output_dir: Path,
    names: dict[int, str] | None,
    case_name: str,
    projection_mode: Literal["max", "mean"],
    image_mode: Literal["default", "channel_difference"],
    difference_channels: tuple[int, int],
) -> None:
    output_path = output_dir / f"{case_name}_projections.png"
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, plane in zip(axes, ("xy", "xz", "yz"), strict=True):
        projection = _project_case_volume(
            image,
            plane,
            projection_mode,
            image_mode=image_mode,
            difference_channels=difference_channels,
        )
        axis.imshow(projection, cmap="gray", interpolation="lanczos", resample=True)
        _annotate_projected_boxes(axis, boxes_xyzxyz, classes, plane, names)
        axis.set_title(f"{plane.upper()} {projection_mode}")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(case_name)
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def _save_slice_visualizations(  # noqa: PLR0913
    image: torch.Tensor,
    boxes_xyzxyz: torch.Tensor,
    classes: torch.Tensor,
    masks: torch.Tensor | None,
    output_dir: Path,
    case_name: str,
    names: dict[int, str] | None,
    image_mode: Literal["default", "channel_difference"],
    difference_channels: tuple[int, int],
    slices_per_plane: int,
) -> int:
    base_volume = _get_case_plot_volume(
        image,
        image_mode=image_mode,
        difference_channels=difference_channels,
    )
    display_scale = _compute_display_scale(base_volume)
    output_dir.mkdir(parents=True, exist_ok=True)

    slice_counts = {
        "xy": int(base_volume.shape[0]),
        "xz": int(base_volume.shape[1]),
        "yz": int(base_volume.shape[2]),
    }

    written = 0
    for plane in ("xy", "xz", "yz"):
        indices = _select_slice_indices(
            plane,
            slice_counts[plane],
            boxes_xyzxyz,
            masks,
            slices_per_plane,
        )
        for slice_index in indices:
            rendered = _render_annotated_slice(
                base_volume,
                display_scale,
                boxes_xyzxyz,
                classes,
                masks,
                plane,
                slice_index,
                names,
            )
            Image.fromarray(rendered).save(
                output_dir / f"{case_name}_{plane}_slice_{slice_index:03d}.png"
            )
            written += 1
    return written


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    if args.augment and args.config is None:
        LOGGER.error("--augment requires --config so the training augmentation settings are known.")
        return 2

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = get_cfg(args.config, overrides=None) if args.config is not None else None
    names = _load_class_names(args.dataset_path)
    dataset = _build_dataset(args, config)
    if len(dataset) == 0:
        LOGGER.error("The selected dataset split is empty.")
        return 1

    LOGGER.info(
        "Visualizing %s samples from %s using %s (augment=%s)",
        args.num_samples,
        args.group,
        type(dataset).__name__,
        dataset.augment,
    )

    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=args.shuffle,
        device=torch.device("cpu"),
    )

    num_written = 0
    difference_channels = tuple(args.difference_channels)
    with tqdm(
        total=args.num_samples, desc="Visualizing", unit="sample", smoothing=0.05
    ) as progress_bar:
        for batch in dataloader:
            prepared_batch = _apply_remaining_batch_augmentations(
                batch,
                config,
                augment=bool(dataset.augment),
            )
            images: torch.Tensor = prepared_batch["img"]
            masks_tuple = prepared_batch["masks"]
            case_names = prepared_batch["case_names"]
            batch_indices: torch.Tensor = prepared_batch["batch_idx"]
            batch_classes: torch.Tensor = prepared_batch["cls"].view(-1)
            batch_boxes: torch.Tensor = prepared_batch["bboxes"]

            for sample_index in range(images.shape[0]):
                if num_written >= args.num_samples:
                    break

                case_name = case_names[sample_index]
                sample_mask = batch_indices == sample_index
                sample_classes = batch_classes[sample_mask].to(dtype=torch.int64)
                sample_boxes = _denormalize_boxes_xyzxyz(
                    images[sample_index],
                    batch_boxes[sample_mask],
                )
                sample_masks = _expand_masks_to_instances(masks_tuple[sample_index])

                _save_projection_overview(
                    image=images[sample_index],
                    boxes_xyzxyz=sample_boxes,
                    classes=sample_classes,
                    output_dir=output_dir,
                    names=names,
                    case_name=case_name,
                    projection_mode=args.projection_mode,
                    image_mode=args.image_mode,
                    difference_channels=difference_channels,
                )
                slice_count = _save_slice_visualizations(
                    image=images[sample_index],
                    boxes_xyzxyz=sample_boxes,
                    classes=sample_classes,
                    masks=sample_masks,
                    output_dir=output_dir,
                    case_name=case_name,
                    names=names,
                    image_mode=args.image_mode,
                    difference_channels=difference_channels,
                    slices_per_plane=args.slices_per_plane,
                )
                LOGGER.info(
                    "Saved sample %d/%d for %s to %s (%d slice PNGs)",
                    num_written + 1,
                    args.num_samples,
                    case_name,
                    output_dir,
                    slice_count,
                )
                num_written += 1
                progress_bar.update(1)

            if num_written >= args.num_samples:
                break

    LOGGER.info("Finished writing %d sample visualizations to %s", num_written, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
