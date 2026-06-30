from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from radyolo.utils import ops

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(slots=True)
class LoadedImageTask:
    """Image volume and metadata handed from loader to inference workers."""

    case_name: str
    affine: np.ndarray
    image: np.ndarray
    voxel_size_xyz: tuple[float, float, float] | None = None


@dataclass(slots=True)
class WriteTask:
    """Prediction payload handed from inference workers to writer workers."""

    case_name: str
    image_results: list[dict[str, object]]
    affine: np.ndarray
    original_shape: tuple[int, int, int]
    prototype_masks: np.ndarray | None
    mask_detections: list[np.ndarray]
    image_label_results: dict[str, object] | None = None
    merged_mask_probabilities: list[np.ndarray] = field(default_factory=list)


def to_numpy_float32_contiguous(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    """Return a CPU float32 contiguous NumPy array."""
    if isinstance(tensor, np.ndarray):
        return np.ascontiguousarray(tensor, dtype=np.float32)
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()


def to_torch_float32_contiguous(array: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Return a CPU float32 contiguous Torch tensor."""
    if isinstance(array, torch.Tensor):
        return array.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return torch.from_numpy(np.ascontiguousarray(array)).to(dtype=torch.float32)


def save_merged_mask_probability(
    mask_probability: torch.Tensor | np.ndarray,
    original_shape: tuple[int, int, int],
    mask_affine: np.ndarray,
    mask_dir: Path,
    det_index: int,
    case_name: str,
) -> str:
    """Threshold and save an already reconstructed mask probability volume."""
    mask_probability_tensor = to_torch_float32_contiguous(mask_probability)
    mask_up = F.interpolate(
        mask_probability_tensor.unsqueeze(0).unsqueeze(0),
        size=original_shape,
        mode="trilinear",
        align_corners=False,
    )[0, 0].permute(2, 1, 0)
    mask_bin = (mask_up >= 0.5).cpu().numpy().astype("uint8")
    mask_img = nib.Nifti1Image(mask_bin, affine=mask_affine)
    mask_dir.mkdir(exist_ok=True)
    mask_filename = f"{case_name}_det{det_index}.nii.gz"
    mask_path = mask_dir / mask_filename
    nib.save(mask_img, str(mask_path))
    return mask_filename


def relative_box(
    box_image_xyzxyz: torch.Tensor, original_shape: tuple[int, int, int]
) -> list[float]:
    """Convert an image-space XYZXYZ box to relative coordinates."""
    rel_box = box_image_xyzxyz.clone()
    d, h, w = original_shape
    rel_box[0] /= w
    rel_box[1] /= h
    rel_box[2] /= d
    rel_box[3] /= w
    rel_box[4] /= h
    rel_box[5] /= d
    return [round(float(x), 6) for x in rel_box.tolist()]


def reconstruct_mask_probability_to_original_shape(
    det: torch.Tensor | np.ndarray,
    proto: torch.Tensor | np.ndarray,
    original_shape: tuple[int, int, int],
) -> torch.Tensor | None:
    """Reconstruct a detection mask probability map at the original image shape."""
    det_tensor = to_torch_float32_contiguous(det)
    proto_tensor = to_torch_float32_contiguous(proto)
    nm = det_tensor.shape[0] - 8
    if nm <= 0:
        return None
    coeff = det_tensor[8 : 8 + nm]
    mask_prob = ops.reconstruct_mask_probabilities(coeff, proto_tensor)
    return F.interpolate(
        mask_prob.unsqueeze(0).unsqueeze(0),
        size=original_shape,
        mode="trilinear",
        align_corners=False,
    )[0, 0].to(dtype=torch.float32)


def reconstruct_and_save_mask(
    det: torch.Tensor | np.ndarray,
    proto: torch.Tensor | np.ndarray,
    original_shape: tuple[int, int, int],
    mask_affine: np.ndarray,
    mask_dir: Path,
    det_index: int,
    case_name: str,
) -> str | None:
    """Reconstruct a detection mask and save it as a thresholded NIfTI file."""
    mask_prob = reconstruct_mask_probability_to_original_shape(det, proto, original_shape)
    if mask_prob is None:
        return None
    mask_up = mask_prob.permute(2, 1, 0)
    mask_bin = (mask_up >= 0.5).cpu().numpy().astype("uint8")
    mask_img = nib.Nifti1Image(mask_bin, affine=mask_affine)
    mask_dir.mkdir(exist_ok=True)
    mask_filename = f"{case_name}_det{det_index}.nii.gz"
    mask_path = mask_dir / mask_filename
    nib.save(mask_img, str(mask_path))
    return mask_filename


def build_prediction_entries(
    detections: torch.Tensor,
    *,
    case_name: str,
    affine: np.ndarray,
    original_shape: tuple[int, int, int],
    names: dict[int, str],
    include_masks: bool,
) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    """Build JSON-ready detection entries and optional mask detections."""
    image_results: list[dict[str, object]] = []
    mask_detections: list[np.ndarray] = []
    for det_i, det in enumerate(detections):
        box_image_xyzxyz = det[:6]
        try:
            box_world_xyzxyz = ops.box_image_to_world(
                box_image_xyzxyz, np.asarray(affine, dtype=np.float32)
            )
        except (RuntimeError, ValueError, TypeError):  # pragma: no cover
            box_world_xyzxyz = box_image_xyzxyz
        entry: dict[str, object] = {
            "class": int(det[7].item()),
            "class_name": names.get(int(det[7].item()), str(int(det[7].item()))),
            "confidence": round(float(det[6].item()), 6),
            "box_image_xyzxyz": [round(float(x), 2) for x in box_image_xyzxyz.tolist()],
            "box_world_xyzxyz": [round(float(x), 2) for x in box_world_xyzxyz.tolist()],
            "box_relative_xyzxyz": relative_box(box_image_xyzxyz, original_shape),
        }
        if include_masks:
            entry["mask_path"] = f"{case_name}_det{det_i}.nii.gz"
            mask_detections.append(to_numpy_float32_contiguous(det))
        image_results.append(entry)
    return image_results, mask_detections


