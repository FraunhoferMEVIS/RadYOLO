from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import torch
import torch.nn.functional as F  # noqa: N812

from yolo11_3d.config.config import model_checkpoint_from_torch
from yolo11_3d.models.model import HardcodedModel
from yolo11_3d.utils import ops
from yolo11_3d.utils.util import select_device

LOGGER = logging.getLogger(__name__)

MASK_THRESHOLD = 0.5


@dataclass
class PredictionConfig:
    model_path: Path
    input_dir: Path
    output_dir: Path
    conf: float | None
    iou: float | None
    device: str | None = None  # optional device override (e.g. "cpu", "cuda:0")


def _load_model(cfg: PredictionConfig) -> tuple[HardcodedModel, Any, dict[int, str]]:
    checkpoint = model_checkpoint_from_torch(cfg.model_path)
    train_config = checkpoint["train_config"]
    model = checkpoint["ema"]
    if not isinstance(model, HardcodedModel):  # fallbacks
        alt = checkpoint.get("model")
        if isinstance(alt, HardcodedModel):
            model = alt
        else:  # type: ignore[unreachable]
            msg = "Unsupported model type in checkpoint"
            raise TypeError(msg)
    device = select_device(cfg.device)
    model.to(device).float().eval()
    # Attach device attribute expected by downstream code
    try:
        setattr(model, "device", device)
    except Exception:  # pragma: no cover - extremely unlikely
        LOGGER.warning("Could not set model.device attribute; continuing.")
    #ModelEMA(model)  # ensures attributes like stride are set, ignoring returned ema instance
    # TODO(kgeissler): Implement loading of correct class names
    num_classes = model.num_classes
    names = {class_index: f"class_{class_index}" for class_index in range(num_classes)}
    return model, train_config, names


def _iter_nifti(input_dir: Path) -> Iterable[Path]:
    return sorted(p for p in input_dir.glob("*.nii.gz") if p.is_file())


def _resize_image(img: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
    if img.ndim == 3:
        img = img.unsqueeze(0)
    return F.interpolate(img.unsqueeze(0), size=size, mode="trilinear").squeeze(0)


def _postprocess_single(pred_class_box, pred_mask_coeff, pred_prototype_masks, train_config, num_classes: int):
    """Return NMS-filtered detections with single (argmax) class per box.

    Output tensor columns: x1,y1,z1,x2,y2,z2,conf,class,(optional mask coeffs...)
    """
    # Unpack detection outputs (inference_out, loss_out)
    if isinstance(pred_class_box, (list, tuple)) and len(pred_class_box) == 2:
        pred_class_box = pred_class_box[0]

    if isinstance(pred_class_box, tuple) and len(pred_class_box) == 3:  # (dbox, cls_prob, obj_prob)
        dbox, cls_prob, obj_prob = pred_class_box
        # For separate objectness head: joint score = class_prob * objectness
        cls_scores = cls_prob * obj_prob
    elif isinstance(pred_class_box, tuple) and len(pred_class_box) == 2:  # (dbox, cls_prob)
        dbox, cls_prob = pred_class_box
        cls_scores = cls_prob
    else:
        # Already a combined prediction tensor of shape (B, 6 + C (+ M), N)
        prediction = pred_class_box
        detections = ops.non_max_suppression(
            prediction,
            train_config.conf if train_config.conf is not None else 0.001,
            train_config.iou,
            multi_label=False,
            agnostic=train_config.agnostic_nms,
            max_det=train_config.max_det,
            nc=num_classes,
        )[0].cpu()
        return detections, pred_prototype_masks

    # Convert to xyzxyz format if necessary already handled in ops.non_max_suppression
    # Build combined prediction tensor in expected layout: (B, 6 + C, N)
    prediction = torch.cat((dbox, cls_scores), 1)

    # Optionally append mask coefficients (shape (B, M, N)) if counts align
    if isinstance(pred_mask_coeff, torch.Tensor):
        if (
            pred_mask_coeff.shape[0] == prediction.shape[0]
            and pred_mask_coeff.shape[2] == prediction.shape[2]
        ):
            prediction = torch.cat((prediction, pred_mask_coeff), 1)

    # Run NMS with multi_label=False to ensure argmax behaviour (ops handles selecting best class)
    detections = ops.non_max_suppression(
        prediction,
        train_config.conf if train_config.conf is not None else 0.001,
        train_config.iou,
        multi_label=False,
        agnostic=train_config.agnostic_nms,
        max_det=train_config.max_det,
        nc=num_classes,
    )[0].cpu()
    return detections, pred_prototype_masks


def _relative_box(box_xyzxyz: torch.Tensor, original_shape: tuple[int, int, int]) -> list[float]:
    rel_box = ops.xyzxyz2xyzwhd(box_xyzxyz.unsqueeze(0))[0]
    d, w, h = original_shape  # note tensor original ordering (D,W,H)
    rel_box[0] /= h
    rel_box[1] /= w
    rel_box[2] /= d
    rel_box[3] /= h
    rel_box[4] /= w
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
    coeff = det[8:8 + nm].to(dtype=proto.dtype, device=proto.device)
    mask_logits = torch.einsum("n,nhwd->hwd", coeff, proto)
    mask_prob = mask_logits.sigmoid()
    mask_prob_r = mask_prob.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    mask_up = (
        F.interpolate(
            mask_prob_r, size=original_shape, mode="trilinear", align_corners=False
        )[0, 0].permute(1, 2, 0)
    )
    mask_bin = (mask_up >= MASK_THRESHOLD).cpu().numpy().astype("uint8")
    mask_affine = nifti_img.affine
    mask_img = nib.Nifti1Image(mask_bin.transpose(2, 0, 1), affine=mask_affine)
    mask_dir.mkdir(exist_ok=True)
    mask_filename = f"{case_name}_det{det_index}.nii.gz"
    mask_path = mask_dir / mask_filename
    nib.save(mask_img, str(mask_path))
    return mask_filename


def run_prediction(cfg: PredictionConfig) -> int:
    try:
        if not cfg.output_dir.exists():
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
        model, train_config, names = _load_model(cfg)
        target_size = tuple(train_config.imgsz)
        files = list(_iter_nifti(cfg.input_dir))
        if not files:
            LOGGER.error("No .nii.gz files found in %s", cfg.input_dir)
            return 1
        for img_path in files:
            case_name = img_path.stem.replace(".nii", "")
            nifti_img = nib.load(str(img_path))
            img_np = nifti_img.get_fdata().astype("float32")
            img_tensor = torch.from_numpy(img_np).permute(*reversed(range(img_np.ndim)))
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.unsqueeze(0)
            original_shape = img_tensor.shape[1:]
            img_resized = _resize_image(img_tensor, target_size)
            device = model.device
            autocast_ctx = torch.autocast(device.type) if device.type != "cpu" else contextlib.nullcontext()
            with torch.no_grad(), autocast_ctx:
                pred_class_box, pred_mask_coeff, pred_prototype_masks = model(
                    img_resized.unsqueeze(0).to(device)
                )
            detections, proto_masks = _postprocess_single(
                pred_class_box, pred_mask_coeff, pred_prototype_masks, train_config, len(names)
            )
            image_results: list[dict[str, object]] = []
            if detections.numel():
                ops.scale_boxes(target_size, detections[:, :6], original_shape, ratio_pad=None)
                for det_i, det in enumerate(detections):
                    box_xyzxyz = det[:6]
                    entry: dict[str, object] = {
                        "class": int(det[7].item()),
                        "class_name": names.get(int(det[7].item()), str(int(det[7].item()))),
                        "confidence": round(float(det[6].item()), 6),
                        "box_world_xyzxyz": [round(float(x), 2) for x in box_xyzxyz.tolist()],
                        "box_relative_center_whd": _relative_box(box_xyzxyz, original_shape),
                    }
                    if pred_mask_coeff is not None and proto_masks is not None:
                        proto = proto_masks[0]
                        mask_filename = _reconstruct_and_save_mask(
                            det,
                            proto,
                            original_shape,
                            nifti_img,
                            cfg.output_dir / "masks",
                            det_i,
                            case_name,
                        )
                        if mask_filename:
                            entry["mask_path"] = mask_filename
                    image_results.append(entry)
            # Write per-image JSON
            out_json = cfg.output_dir / f"{case_name}.json"
            with out_json.open("w") as jf:
                json.dump(image_results, jf, indent=2)
        LOGGER.info("Saved predictions for %d volumes to %s", len(files), cfg.output_dir)
    except Exception as e:  # pragma: no cover - defensive
        LOGGER.exception("Prediction failed", exc_info=e)
        return 1
    else:
        return 0
