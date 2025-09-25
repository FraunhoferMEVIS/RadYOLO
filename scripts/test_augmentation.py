from __future__ import annotations

import random
from pathlib import Path
import sys

import torch

# Ensure local src is imported even if a site-packages version exists
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from yolo11_3d.data.augmentations import (  # type: ignore  # noqa: E402
    batch_generators_intensity_augmentations,
    enlarge_bounding_boxes,
    random_mirroring,
    random_zoom,
    tensor_cutout,
    random_rotation,
    random_channel_dropout,
    gin_augmentation,
)
from yolo11_3d.validation.plotting import plot_images  # type: ignore  # noqa: E402


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def create_base_volume(shape: tuple[int, int, int] = (256, 256, 256)) -> torch.Tensor:
    img = torch.zeros((3, *shape), dtype=torch.float32)
    d, h, w = shape
    img[0, d // 2 :, :, :] = 1
    img[1, :, h // 2 :, :] = 1
    img[2, :, :, w // 2 :] = 1
    return img


def create_boxes(num: int, imgsz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lt_list = []
    rb_list = []
    for _ in range(num):
        lt = [random.uniform(0.05, 0.85) for _ in range(3)]
        rb = [random.uniform(lt[i] + 0.02, 0.95) for i in range(3)]
        lt_list.append(lt)
        rb_list.append(rb)
    # tiny box
    lt_list.append([0.1, 0.1, 0.1])
    rb_list.append([0.12, 0.12, 0.12])
    lt_t = torch.tensor(lt_list, dtype=torch.float32)
    rb_t = torch.tensor(rb_list, dtype=torch.float32)
    cls = torch.arange(len(lt_list)).unsqueeze(-1).float()
    boxes_norm_xyzxyz = torch.cat([lt_t, rb_t], 1)
    scale_xyzxyz = imgsz[[2, 1, 0, 2, 1, 0]].float()
    boxes_abs_xyzxyz = boxes_norm_xyzxyz * scale_xyzxyz
    return cls, boxes_abs_xyzxyz


def create_elliptical_masks(boxes_abs_xyzxyz: torch.Tensor, volume_shape: tuple[int, int, int]) -> torch.Tensor:
    """Create per-box 3D binary masks shaped (N, D, H, W) with an ellipsoid inside each bounding box.

    Args:
        boxes_abs_xyzxyz: (N, 7) tensor [cls, x1, y1, z1, x2, y2, z2]
        volume_shape: (D, H, W)
    Returns:
        masks: (N, D, H, W) torch.bool
    """
    D, H, W = volume_shape
    n = boxes_abs_xyzxyz.shape[0]
    masks = torch.zeros((n, D, H, W))
    for i in range(n):
        _, x1, y1, z1, x2, y2, z2 = boxes_abs_xyzxyz[i]
        x1i, y1i, z1i = map(lambda v: max(int(v.item()), 0), (x1, y1, z1))
        x2i, y2i, z2i = map(lambda v: min(int(v.item()) + 1, W), (x2, y2, z2))
        # ensure at least 1 voxel thickness
        if x2i <= x1i or y2i <= y1i or z2i <= z1i:
            continue
        xs = torch.arange(x1i, x2i)
        ys = torch.arange(y1i, y2i)
        zs = torch.arange(z1i, z2i)
        zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")  # (dz, dy, dx)
        # radii
        rx = max((x2i - x1i) / 2.0, 1.0)
        ry = max((y2i - y1i) / 2.0, 1.0)
        rz = max((z2i - z1i) / 2.0, 1.0)
        cx = x1i + rx
        cy = y1i + ry
        cz = z1i + rz
        ellip = (((xx + 0.5 - cx) / rx) ** 2 + ((yy + 0.5 - cy) / ry) ** 2 + ((zz + 0.5 - cz) / rz) ** 2) <= 1.0
        masks[i, z1i:z2i, y1i:y2i, x1i:x2i] = ellip
    return masks


def abs_xyzxyz_to_abs_xyzwhd(t: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, x2, y2, z2 = [t[:, i] for i in range(1, 7)]
    w = x2 - x1
    h = y2 - y1
    d = z2 - z1
    x = x1 + w / 2
    y = y1 + h / 2
    z = z1 + d / 2
    return torch.stack([t[:, 0], x, y, z, w, h, d], 1)


def abs_xyzwhd_to_abs_xyzxyz(t: torch.Tensor) -> torch.Tensor:
    x, y, z, w, h, d = [t[:, i] for i in range(1, 7)]
    x1 = x - w / 2
    y1 = y - h / 2
    z1 = z - d / 2
    x2 = x + w / 2
    y2 = y + h / 2
    z2 = z + d / 2
    return torch.stack([t[:, 0], x1, y1, z1, x2, y2, z2], 1)


def abs_xyzxyz_to_norm_xyzwhd(t: torch.Tensor, scale_xyzxyz: torch.Tensor) -> torch.Tensor:
    xyzwhd_abs = abs_xyzxyz_to_abs_xyzwhd(t)
    norm = xyzwhd_abs.clone()
    norm[:, 1:7] = norm[:, 1:7] / scale_xyzxyz[[0, 1, 2, 0, 1, 2]]
    return norm


def norm_xyzwhd_to_abs_xyzxyz(t: torch.Tensor, scale_xyzxyz: torch.Tensor) -> torch.Tensor:
    abs_whd = t.clone()
    abs_whd[:, 1:7] = abs_whd[:, 1:7] * scale_xyzxyz[[0, 1, 2, 0, 1, 2]]
    return abs_xyzwhd_to_abs_xyzxyz(abs_whd)


def apply_tensor_cutout(image: torch.Tensor, labels_abs_xyzxyz: torch.Tensor, masks: torch.Tensor, params) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    img_out, labels_out, masks_out = tensor_cutout(image, labels_abs_xyzxyz, masks, params, p=1.0)
    masks_out = masks_out if masks_out is not None else torch.zeros(0, *masks.shape[1:], dtype=masks.dtype)
    return img_out, labels_out, masks_out


def apply_zoom(image: torch.Tensor, labels_abs_xyzxyz: torch.Tensor, masks: torch.Tensor, uniform: bool, max_zoom=1.5, min_zoom=0.7) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    img_out, labels_out, masks_out = random_zoom(
        image,
        labels_abs_xyzxyz,
        masks,
        max_zoom=max_zoom,
        min_zoom=min_zoom,
        p=1.0,
        out_zoom_padding="constant",
        uniform_zoom=uniform,
    )
    masks_out = masks_out if masks_out is not None else torch.zeros(0, *masks.shape[1:], dtype=masks.dtype)
    return img_out, labels_out, masks_out


def apply_mirroring(image: torch.Tensor, labels_abs_xyzxyz: torch.Tensor, masks: torch.Tensor, scale_xyzxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels_norm_xyzwhd = abs_xyzxyz_to_norm_xyzwhd(labels_abs_xyzxyz, scale_xyzxyz)
    img_out, labels_norm_out, masks_out = random_mirroring(image, labels_norm_xyzwhd, masks, p=1.0)
    labels_abs_xyzxyz_out = norm_xyzwhd_to_abs_xyzxyz(labels_norm_out, scale_xyzxyz)
    masks_out = masks_out if masks_out is not None else torch.zeros(0, *masks.shape[1:], dtype=masks.dtype)
    return img_out, labels_abs_xyzxyz_out, masks_out


def apply_intensity(image: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return batch_generators_intensity_augmentations(image, do_intensity_augmentations_per_channel=True), masks


def apply_enlarge(image: torch.Tensor, labels_abs_xyzxyz: torch.Tensor, masks: torch.Tensor, imgsz: torch.Tensor, stride: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels_abs_xyzwhd = abs_xyzxyz_to_abs_xyzwhd(labels_abs_xyzxyz)
    img_out, labels_abs_xyzwhd_out, masks_out = enlarge_bounding_boxes(image, labels_abs_xyzwhd, masks, tuple(imgsz.tolist()), stride=stride)
    labels_abs_xyzxyz_out = abs_xyzwhd_to_abs_xyzxyz(labels_abs_xyzwhd_out)
    masks_out = masks_out if masks_out is not None else torch.zeros(0, *masks.shape[1:], dtype=masks.dtype)
    return img_out, labels_abs_xyzxyz_out, masks_out


def build_grid_for_augmentation(
    aug_name: str,
    base_img: torch.Tensor,
    base_labels_abs_xyzxyz: torch.Tensor,
    imgsz: torch.Tensor,
    n: int = 16,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Return list of images, labels, masks for a single augmentation (original + variants)."""
    scale_xyzxyz = imgsz[[2, 1, 0, 2, 1, 0]].float()
    images: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    masks_list: list[torch.Tensor] = []
    # base masks from original boxes
    base_masks = create_elliptical_masks(base_labels_abs_xyzxyz, tuple(imgsz.tolist()))
    images.append(base_img.clone())
    labels.append(base_labels_abs_xyzxyz.clone())
    masks_list.append(base_masks.clone())

    need = n - 1
    while need > 0:
        if aug_name == "tensor_cutout":
            params = [(0.5, 1), (0.25, 2), (0.125, 4)]
            img_v, lab_v, mask_v = apply_tensor_cutout(base_img.clone(), base_labels_abs_xyzxyz.clone(), base_masks.clone(), params)
        elif aug_name == "random_zoom_uniform":
            img_v, lab_v, mask_v = apply_zoom(base_img.clone(), base_labels_abs_xyzxyz.clone(), base_masks.clone(), True)
        elif aug_name == "random_zoom_aniso":
            img_v, lab_v, mask_v = apply_zoom(base_img.clone(), base_labels_abs_xyzxyz.clone(), base_masks.clone(), False)
        elif aug_name == "random_mirroring":
            img_v, lab_v, mask_v = apply_mirroring(base_img.clone(), base_labels_abs_xyzxyz.clone(), base_masks.clone(), scale_xyzxyz)
        elif aug_name == "intensity":
            img_v, mask_v = apply_intensity(base_img.clone(), base_masks.clone())
            lab_v = base_labels_abs_xyzxyz.clone()
        elif aug_name == "enlarge_bounding_boxes":
            stride = random.choice([4, 8, 12, 16, 24, 32])
            img_v, lab_v, mask_v = apply_enlarge(base_img.clone(), base_labels_abs_xyzxyz.clone(), base_masks.clone(), imgsz, stride)
        elif aug_name == "random_rotation":
            # choose angle ranges variant (wider sometimes)
            if random.random() < 0.5:
                angle_ranges = ((-15, 15), (-10, 10), (-20, 20))
            else:
                angle_ranges = ((-30, 30), (-5, 5), (-10, 10))
            img_v, lab_v, mask_v = random_rotation(
                base_img.clone(),
                base_labels_abs_xyzxyz.clone(),
                base_masks.clone(),
                angle_ranges=angle_ranges,
                p=1.0,
                keep_box_volume=True,
            )
        elif aug_name == "random_channel_dropout":
            # Example per-channel probabilities for 3 channels
            drop_probs = [0.4, 0.3, 0.2]
            img_v, lab_v, mask_v = random_channel_dropout(
                base_img.clone(),
                base_labels_abs_xyzxyz.clone(),
                base_masks.clone(),
                drop_probabilities=drop_probs,
                fill_value=0.0,
            )
        elif aug_name == "gin_augmentation":
            img_v, lab_v, mask_v = gin_augmentation(
                base_img.clone(),
                base_labels_abs_xyzxyz.clone(),
                base_masks.clone(),
                hidden_layers=4,
                hidden_channels=2,
                p_kernel_size_1=0.5,
                p=1.0
            )
        else:
            raise ValueError(f"Unknown augmentation name '{aug_name}'")
        images.append(img_v)
        labels.append(lab_v)
        masks_list.append(mask_v)
        need -= 1
    return images, labels, masks_list


def prepare_for_plot(images: list[torch.Tensor], labels_abs_xyzxyz: list[torch.Tensor], masks_list: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    imgs_tensor = torch.stack(images)
    scale_tensor = torch.as_tensor(imgs_tensor.shape[2:])[ [2,1,0,2,1,0] ]
    norm_xyzwhd_labels = []
    flat_masks_xy: list[torch.Tensor] = []
    flat_masks_xz: list[torch.Tensor] = []
    for labs, masks in zip(labels_abs_xyzxyz, masks_list):
        xyzwhd = abs_xyzxyz_to_abs_xyzwhd(labs)
        norm = xyzwhd.clone()
        norm[:, 1:7] = norm[:, 1:7] / scale_tensor
        norm_xyzwhd_labels.append(norm)
        # masks shape (N,D,H,W)
        # XY projection: max over depth (dim=1) -> (N,H,W)
        masks_xy = masks.amax(dim=1)
        # XZ projection: max over height (dim=2) -> (N,D,W)
        masks_xz = masks.amax(dim=2)
        flat_masks_xy.append(masks_xy)
        flat_masks_xz.append(masks_xz)
    cat_labels = torch.cat(norm_xyzwhd_labels, 0)
    batch_idx = torch.as_tensor([i for i, labs in enumerate(norm_xyzwhd_labels) for _ in range(len(labs))])
    cat_masks_xy = torch.cat(flat_masks_xy, 0)
    cat_masks_xz = torch.cat(flat_masks_xz, 0)
    return imgs_tensor, batch_idx, cat_labels, cat_masks_xy, cat_masks_xz


def save_projections(out_dir: Path, tag: str, imgs_tensor: torch.Tensor, batch_idx: torch.Tensor, labels_norm_xyzwhd: torch.Tensor, masks_xy: torch.Tensor, masks_xz: torch.Tensor) -> None:
    plot_images(
        images=imgs_tensor.amax(-3),
        batch_idx=batch_idx,
        cls=labels_norm_xyzwhd[:, 0:1],
        bboxes=labels_norm_xyzwhd[:, [1, 2, 4, 5]],
        masks=masks_xy,
        fname=str(out_dir / f"{tag}_xy.jpg"),
    )
    plot_images(
        images=imgs_tensor.amax(-2),
        batch_idx=batch_idx,
        cls=labels_norm_xyzwhd[:, 0:1],
        bboxes=labels_norm_xyzwhd[:, [1, 3, 4, 6]],
        masks=masks_xz,
        fname=str(out_dir / f"{tag}_xz.jpg"),
    )


def main() -> None:
    seed_everything(123)
    out_dir = Path.cwd() / "augmentation_tests"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_img = create_base_volume()
    imgsz = torch.as_tensor(base_img.shape[1:])
    cls_ids, boxes_abs_xyzxyz = create_boxes(6, imgsz)
    base_labels_abs_xyzxyz = torch.cat([cls_ids, boxes_abs_xyzxyz], 1)

    augmentations = [
        "tensor_cutout",
        "random_zoom_uniform",
        "random_zoom_aniso",
        "random_mirroring",
        "intensity",
        "enlarge_bounding_boxes",
        "random_rotation",
        "random_channel_dropout",
        "gin_augmentation",
    ]

    for aug in augmentations:
        images, labels_abs_xyzxyz, masks_list = build_grid_for_augmentation(aug, base_img, base_labels_abs_xyzxyz, imgsz)
        imgs_tensor, batch_idx, labels_norm_xyzwhd, masks_xy, masks_xz = prepare_for_plot(images, labels_abs_xyzxyz, masks_list)
        save_projections(out_dir, aug, imgs_tensor, batch_idx, labels_norm_xyzwhd, masks_xy, masks_xz)
        print(f"Saved {aug} grid (with masks) to {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
