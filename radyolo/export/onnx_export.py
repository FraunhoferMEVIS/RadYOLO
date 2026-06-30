from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import onnx
import torch
from onnxconverter_common import float16
from torch import nn

from radyolo.models.layers import C2f, Detect

if TYPE_CHECKING:
    from radyolo.config.config import DefaultConfig
    from radyolo.models.model import HardcodedModel


LOGGER = logging.getLogger(__name__)


ExportInternalDType = Literal["float16", "float32"]


class ExportModelWrapper(nn.Module):
    def __init__(self, model: HardcodedModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.model(x).to_export_tensors()


def _looks_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"true", "1", "yes"}


def _checkpoint_kind_from_path(checkpoint_path: Path) -> str | None:
    stem = checkpoint_path.stem.lower()
    if "best" in stem:
        return "best"
    if "last" in stem:
        return "last"
    return None


def _read_best_nms_iou_threshold(
    checkpoint_path: Path,
    *,
    default_threshold: float = 0.1,
) -> float:
    csv_path = checkpoint_path.parent / "nms_optimization.csv"
    if not csv_path.exists():
        return default_threshold

    try:
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        LOGGER.warning("Failed to read NMS optimization CSV at %s", csv_path, exc_info=True)
        return default_threshold

    if not rows:
        return default_threshold

    checkpoint_kind = _checkpoint_kind_from_path(checkpoint_path)
    if checkpoint_kind is not None:
        checkpoint_rows = [
            row for row in rows if str(row.get("checkpoint", "")).strip().lower() == checkpoint_kind
        ]
    else:
        checkpoint_rows = rows

    if not checkpoint_rows:
        return default_threshold

    suggested_rows = [row for row in checkpoint_rows if _looks_truthy(row.get("suggested"))]
    candidate_rows = suggested_rows or checkpoint_rows

    try:
        best_row = max(candidate_rows, key=lambda row: float(row.get("fitness", "-inf")))
        return float(best_row["nms_iou_threshold"])
    except (KeyError, ValueError):
        LOGGER.warning("Failed to parse best NMS IoU threshold from %s", csv_path, exc_info=True)
        return default_threshold


def _to_metadata_string(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def build_export_metadata(
    *,
    train_config: DefaultConfig,
    checkpoint_path: Path,
    default_nms_iou_threshold: float = 0.1,
    metadata_overrides: dict[str, object] | None = None,
) -> dict[str, str]:
    model_config = train_config.model_dump(mode="json")
    metadata = {
        "training_mode": _to_metadata_string(model_config.get("training_mode")),
        "patch_size": _to_metadata_string(model_config.get("patch_size")),
        "voxel_size": _to_metadata_string(model_config.get("voxel_size")),
        "normalization": _to_metadata_string(model_config.get("normalization")),
        "data": _to_metadata_string(model_config.get("data")),
        "fold": _to_metadata_string(model_config.get("fold")),
        "split_file_name": _to_metadata_string(model_config.get("split_file_name")),
        "best_nms_iou_threshold": _to_metadata_string(
            _read_best_nms_iou_threshold(
                checkpoint_path,
                default_threshold=default_nms_iou_threshold,
            )
        ),
    }

    for key, value in (metadata_overrides or {}).items():
        metadata[key] = _to_metadata_string(value)

    return metadata


def _resolve_output_names(model: HardcodedModel) -> list[str]:
    if model.enable_output_masks and not model.separate_objectness_head:
        return ["boxes", "classScores", "maskCoefficients", "prototypeMasks"]
    if not model.enable_output_masks and model.separate_objectness_head:
        return ["boxes", "classScores", "objectnessScores"]
    if model.enable_output_masks and model.separate_objectness_head:
        return [
            "boxes",
            "classScores",
            "objectnessScores",
            "maskCoefficients",
            "prototypeMasks",
        ]
    return ["boxes", "classScores"]


def _write_onnx_metadata(onnx_model: onnx.ModelProto, metadata: dict[str, str]) -> None:
    merged_metadata = {entry.key: entry.value for entry in onnx_model.metadata_props}
    merged_metadata.update(metadata)

    del onnx_model.metadata_props[:]
    for key in sorted(merged_metadata):
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = merged_metadata[key]


def export_model(
    model: HardcodedModel,
    input_shape: tuple[int, ...],
    dest: Path | str,
    internal_dtype: ExportInternalDType = "float16",
    metadata: dict[str, str] | None = None,
) -> Path:
    if internal_dtype not in {"float16", "float32"}:
        msg = f"Unsupported internal ONNX dtype: {internal_dtype}"
        raise ValueError(msg)

    inputs = torch.zeros((1, *input_shape)).to("cpu")
    model.to("cpu")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()

    for _ in range(2):
        model(inputs)

    for m in model.modules():
        if isinstance(m, Detect):
            m.export = True
        if isinstance(m, C2f):
            m.forward = m.forward_split

    onnx_path = Path(dest).with_suffix(".onnx")
    export_wrapper = ExportModelWrapper(model)
    export_wrapper.eval()

    output_names = _resolve_output_names(model)

    torch.onnx.export(
        export_wrapper,
        inputs,
        onnx_path,
        verbose=False,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=None,
        external_data=False,
    )

    if internal_dtype == "float16" or metadata:
        onnx_model = onnx.load(onnx_path)
        if internal_dtype == "float16":
            onnx_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
        if metadata:
            _write_onnx_metadata(onnx_model, metadata)
        onnx.save(onnx_model, onnx_path)

    return onnx_path
