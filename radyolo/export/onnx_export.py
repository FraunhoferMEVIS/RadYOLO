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
    """Wrap `HardcodedModel` to expose export-ready tensor outputs."""

    def __init__(self, model: HardcodedModel) -> None:
        super().__init__()
        self.model = model
        self._image_label_task_names = tuple(sorted(model.image_label_task_specs))
        self._expects_tabular_inputs = (
            model.enable_tabular_image_label_fusion and len(model.tabular_feature_names) > 0
        )

    def forward(
        self,
        x: torch.Tensor,
        tabular_values: torch.Tensor | None = None,
        tabular_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Run the model and return tensors in the export output format."""
        if self._expects_tabular_inputs:
            if tabular_values is None or tabular_mask is None:
                msg = "tabular_values and tabular_mask are required for this export wrapper"
                raise RuntimeError(msg)
            model_output = self.model(x, tabular_values=tabular_values, tabular_mask=tabular_mask)
        else:
            model_output = self.model(x)
        return model_output.to_export_tensors(image_label_task_order=self._image_label_task_names)


def _looks_truthy(value: str | None) -> bool:
    """Return whether a CSV-like string value represents true."""
    if value is None:
        return False
    return value.strip().lower() in {"true", "1", "yes"}


def _checkpoint_kind_from_path(checkpoint_path: Path) -> str | None:
    """Infer checkpoint kind (`best`/`last`) from the checkpoint filename."""
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
    """Read the best suggested NMS IoU threshold from sibling optimization CSV."""
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
    """Serialize metadata values to stable strings for ONNX metadata fields."""
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
    """Build ONNX metadata from training config and export-time overrides."""
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
    """Determine ONNX output names from enabled model heads."""
    if model.enable_output_masks and not model.separate_objectness_head:
        output_names = ["boxes", "classScores", "maskCoefficients", "prototypeMasks"]
    elif not model.enable_output_masks and model.separate_objectness_head:
        output_names = ["boxes", "classScores", "objectnessScores"]
    elif model.enable_output_masks and model.separate_objectness_head:
        output_names = [
            "boxes",
            "classScores",
            "objectnessScores",
            "maskCoefficients",
            "prototypeMasks",
        ]
    else:
        output_names = ["boxes", "classScores"]

    if model.enable_image_label_head and model.image_label_task_specs:
        output_names.extend(
            f"imageLabel_{task_name}" for task_name in sorted(model.image_label_task_specs)
        )

    return output_names


def _resolve_export_inputs(
    model: HardcodedModel,
    input_shape: tuple[int, ...],
) -> tuple[tuple[torch.Tensor, ...], list[str]]:
    """Build deterministic ONNX export inputs and matching input names."""
    images = torch.zeros((1, *input_shape), dtype=torch.float32, device="cpu")

    if model.enable_tabular_image_label_fusion and len(model.tabular_feature_names) > 0:
        num_features = len(model.tabular_feature_names)
        tabular_values = torch.ones((1, num_features), dtype=torch.float32, device="cpu")
        tabular_mask = torch.ones((1, num_features), dtype=torch.bool, device="cpu")
        return (
            images,
            tabular_values,
            tabular_mask,
        ), ["images", "tabularValues", "tabularMask"]

    return (images,), ["images"]


def _write_onnx_metadata(onnx_model: onnx.ModelProto, metadata: dict[str, str]) -> None:
    """Merge and write metadata entries into an ONNX model in sorted key order."""
    merged_metadata = {entry.key: entry.value for entry in onnx_model.metadata_props}
    merged_metadata.update(metadata)

    del onnx_model.metadata_props[:]
    for key in sorted(merged_metadata):
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = merged_metadata[key]


def _export_torch_onnx_with_fallback(
    export_wrapper: ExportModelWrapper,
    export_inputs: tuple[torch.Tensor, ...],
    onnx_path: Path,
    input_names: list[str],
    output_names: list[str],
) -> None:
    """Export ONNX with a fallback to the legacy exporter for unsupported dynamo ops."""
    export_input_argument: torch.Tensor | tuple[torch.Tensor, ...]
    export_input_argument = export_inputs if len(export_inputs) > 1 else export_inputs[0]

    previous_mha_fastpath: bool | None = None
    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "get_fastpath_enabled"):
        previous_mha_fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)

    try:
        try:
            torch.onnx.export(
                export_wrapper,
                export_input_argument,
                onnx_path,
                verbose=False,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=None,
                external_data=False,
                dynamo=True,
            )
        except Exception:
            LOGGER.warning(
                "Dynamo ONNX export failed; retrying with legacy exporter.",
                exc_info=True,
            )
            torch.onnx.export(
                export_wrapper,
                export_input_argument,
                onnx_path,
                verbose=False,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=None,
                external_data=False,
                dynamo=False,
            )
    finally:
        if previous_mha_fastpath is not None:
            torch.backends.mha.set_fastpath_enabled(previous_mha_fastpath)


def export_model(
    model: HardcodedModel,
    input_shape: tuple[int, ...],
    dest: Path | str,
    internal_dtype: ExportInternalDType = "float16",
    metadata: dict[str, str] | None = None,
) -> Path:
    """Export a model to ONNX, optionally convert internal dtype, and attach metadata."""
    if internal_dtype not in {"float16", "float32"}:
        msg = f"Unsupported internal ONNX dtype: {internal_dtype}"
        raise ValueError(msg)

    export_inputs, input_names = _resolve_export_inputs(model, input_shape)
    model.to("cpu")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()

    for _ in range(2):
        if len(export_inputs) > 1:
            model(
                export_inputs[0],
                tabular_values=export_inputs[1],
                tabular_mask=export_inputs[2],
            )
        else:
            model(export_inputs[0])

    for m in model.modules():
        if isinstance(m, Detect):
            m.export = True
        if isinstance(m, C2f):
            m.forward = m.forward_split

    onnx_path = Path(dest).with_suffix(".onnx")
    export_wrapper = ExportModelWrapper(model)
    export_wrapper.eval()

    output_names = _resolve_output_names(model)

    _export_torch_onnx_with_fallback(
        export_wrapper,
        export_inputs,
        onnx_path,
        input_names,
        output_names,
    )

    if internal_dtype == "float16" or metadata:
        onnx_model = onnx.load(onnx_path)
        if internal_dtype == "float16":
            onnx_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
        if metadata:
            _write_onnx_metadata(onnx_model, metadata)
        onnx.save(onnx_model, onnx_path)

    return onnx_path
