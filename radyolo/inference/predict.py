from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from radyolo.config.config import DefaultConfig, DetectionPostprocessConfig
from radyolo.data.preprocessing import preprocess_image
from radyolo.models.model import load_checkpoint_model
from radyolo.utils import ops

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from radyolo.models.layers import ModelOutput

LOGGER = logging.getLogger(__name__)

MASK_THRESHOLD = 0.5


@dataclass
class PredictionConfig:
    model_path: Path
    input_dir: Path
    output_dir: Path
    confidence_threshold: float
    nms_iou_threshold: float
    device: str
    clip_boxes_to_image: bool


def _load_model(cfg: PredictionConfig) -> tuple[torch.nn.Module, DefaultConfig, dict[int, str]]:
    loaded_model = load_checkpoint_model(cfg.model_path, device_override=cfg.device)
    train_config = loaded_model.train_config
    num_classes = train_config.num_classes
    names = {i: f"class_{i}" for i in range(num_classes)}

    return loaded_model.model, train_config, names


def _iter_nifti(input_dir: Path) -> Iterable[Path]:
    return sorted(p for p in input_dir.glob("*.nii.gz") if p.is_file())


def _postprocess_single(
    model_output: ModelOutput,
    train_config: DefaultConfig,
    prediction_config: PredictionConfig,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return NMS-filtered detections with single (argmax) class per box.

    Output tensor columns: x1,y1,z1,x2,y2,z2,conf,class,(optional mask coeffs...)
    """
    detections = ops.postprocess_model_output(
        model_output,
        DetectionPostprocessConfig(
            confidence_threshold=prediction_config.confidence_threshold,
            nms_iou_threshold=prediction_config.nms_iou_threshold,
            agnostic_nms=train_config.agnostic_nms,
            max_det=train_config.max_det,
            nms_time_limit=train_config.nms_time_limit,
            nc=num_classes,
            multi_label=False,
        ),
    )[0].cpu()
    return detections, model_output.prototype_masks


def _relative_box(
    box_image_xyzxyz: torch.Tensor, original_shape: tuple[int, int, int]
) -> list[float]:
    rel_box = box_image_xyzxyz.clone()
    d, w, h = original_shape  # note tensor original ordering (D,W,H)
    rel_box[0] /= w
    rel_box[1] /= h
    rel_box[2] /= d
    rel_box[3] /= w
    rel_box[4] /= h
    rel_box[5] /= d
    return [round(float(x), 6) for x in rel_box.tolist()]


def _reconstruct_and_save_mask(
    det: torch.Tensor,
    proto: torch.Tensor,
    original_shape: tuple[int, int, int],
    nifti_img: nib.Nifti1Image,
    mask_dir: Path,
    det_index: int,
    case_name: str,
) -> str | None:
    nm = det.shape[0] - 8
    if nm <= 0:
        return None
    coeff = det[8 : 8 + nm].to(device=proto.device)
    mask_prob = ops.reconstruct_mask_probabilities(coeff, proto)
    mask_prob_r = mask_prob.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    mask_up = F.interpolate(
        mask_prob_r, size=original_shape, mode="trilinear", align_corners=False
    )[0, 0].permute(2, 1, 0) # (W, H, D)
    mask_bin = (mask_up >= MASK_THRESHOLD).cpu().numpy().astype("uint8")
    mask_affine = nifti_img.affine
    mask_img = nib.Nifti1Image(mask_bin, affine=mask_affine)
    mask_dir.mkdir(exist_ok=True)
    mask_filename = f"{case_name}_det{det_index}.nii.gz"
    mask_path = mask_dir / mask_filename
    nib.save(mask_img, str(mask_path))
    return mask_filename


BOX_COORDS_LEN = 6
NDIM_3D = 3


def _box_image_to_world(
    box_image_xyzxyz: torch.Tensor,
    affine: (torch.Tensor | list[list[float]] | tuple[tuple[float, ...], ...]),
) -> torch.Tensor:
    """Convert an axis-aligned image-space 3D box to world-space using the NIfTI affine.

    The input box is (x1,y1,z1,x2,y2,z2) in voxel coordinates matching the original
    NIfTI indexing order (i=X, j=Y, k=Z). Because an affine can include rotation and
    shear, an axis-aligned box in voxel space may not remain axis-aligned in world
    space. We therefore transform all 8 voxel-space box corners and then take the
    min/max along each world axis to obtain the tightest axis-aligned world-space
    bounding box that encloses the transformed shape.

    Args:
        box_image_xyzxyz: torch.Tensor with shape (6,).
        affine: 4x4 array-like affine from the loaded NIfTI (nifti_img.affine).

    Returns:
        torch.Tensor: (x1,y1,z1,x2,y2,z2) in world coordinates (float32).
    """
    if box_image_xyzxyz.numel() != BOX_COORDS_LEN:
        msg = f"Expected box tensor of shape ({BOX_COORDS_LEN},)"
        raise ValueError(msg)

    # Extract coordinates
    x1, y1, z1, x2, y2, z2 = box_image_xyzxyz.tolist()
    # Generate 8 corners in voxel space
    corners = torch.tensor(
        [[xi, yi, zi] for xi in (x1, x2) for yi in (y1, y2) for zi in (z1, z2)],
        dtype=torch.float32,
    )  # (8,3)
    ones = torch.ones((corners.shape[0], 1), dtype=corners.dtype)
    ijk_h = torch.cat([corners, ones], dim=1)  # (8,4)
    aff = torch.as_tensor(affine, dtype=corners.dtype)
    world = (ijk_h @ aff.T)[:, :3]  # (8,3)
    min_xyz = world.min(dim=0).values
    max_xyz = world.max(dim=0).values
    world_box = torch.stack(
        [
            min_xyz[0],
            min_xyz[1],
            min_xyz[2],
            max_xyz[0],
            max_xyz[1],
            max_xyz[2],
        ]
    )
    return world_box.to(dtype=box_image_xyzxyz.dtype)


def run_prediction(prediction_cfg: PredictionConfig) -> int:
    try:
        if not prediction_cfg.output_dir.exists():
            prediction_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        model, train_config, names = _load_model(prediction_cfg)
        clip_boxes_to_image = prediction_cfg.clip_boxes_to_image
        target_size = train_config.patch_size
        files = list(_iter_nifti(prediction_cfg.input_dir))
        if not files:
            LOGGER.error("No .nii.gz files found in %s", prediction_cfg.input_dir)
            return 1
        for img_path in files:
            case_name = img_path.stem.replace(".nii", "")
            nifti_img = nib.load(str(img_path))
            img_np = np.asanyarray(nifti_img.dataobj, dtype=np.float32)
            img_tensor = torch.from_numpy(img_np).permute(*reversed(range(img_np.ndim)))
            img_preprocessed, original_shape, _ = preprocess_image(
                img_tensor,
                train_config.normalization,
                target_shape=target_size,
                output_dtype=torch.float32,
            )
            device = model.device
            autocast_ctx = (
                torch.autocast(device.type) if device.type != "cpu" else contextlib.nullcontext()
            )
            with torch.no_grad(), autocast_ctx:
                model_output: ModelOutput = model(img_preprocessed.unsqueeze(0).to(device))
            detections, proto_masks = _postprocess_single(
                model_output,
                train_config,
                prediction_cfg,
                len(names),
            )
            image_results: list[dict[str, object]] = []
            if detections.numel():
                ops.scale_boxes(
                    target_size.get_zyx(),
                    detections[:, :6],
                    original_shape,
                    ratio_pad=None,
                    clip=clip_boxes_to_image,
                )
                for det_i, det in enumerate(detections):
                    box_image_xyzxyz = det[:6]
                    try:
                        box_world_xyzxyz = _box_image_to_world(box_image_xyzxyz, nifti_img.affine)
                    except (RuntimeError, ValueError, TypeError):  # pragma: no cover
                        box_world_xyzxyz = box_image_xyzxyz
                    entry: dict[str, object] = {
                        "class": int(det[7].item()),
                        "class_name": names.get(int(det[7].item()), str(int(det[7].item()))),
                        "confidence": round(float(det[6].item()), 6),
                        "box_image_xyzxyz": [round(float(x), 2) for x in box_image_xyzxyz.tolist()],
                        "box_world_xyzxyz": [round(float(x), 2) for x in box_world_xyzxyz.tolist()],
                        "box_relative_xyzxyz": _relative_box(box_image_xyzxyz, original_shape),
                    }
                    if model_output.mask_coefficients is not None and proto_masks is not None:
                        proto = proto_masks[0]
                        mask_filename = _reconstruct_and_save_mask(
                            det,
                            proto,
                            original_shape,
                            nifti_img,
                            prediction_cfg.output_dir / "masks",
                            det_i,
                            case_name,
                        )
                        if mask_filename:
                            entry["mask_path"] = mask_filename
                    image_results.append(entry)
            # Write per-image JSON
            out_json = prediction_cfg.output_dir / f"{case_name}.json"
            with out_json.open("w") as jf:
                json.dump(image_results, jf, indent=2)
        LOGGER.info("Saved predictions for %d volumes to %s", len(files), prediction_cfg.output_dir)
    except Exception as e:  # pragma: no cover - defensive
        LOGGER.exception("Prediction failed", exc_info=e)
        return 1
    else:
        return 0
