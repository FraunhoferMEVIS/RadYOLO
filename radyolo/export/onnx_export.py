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
from torch.export import Dim

from radyolo.data.preprocessing import round_up
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

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run the model and return tensors in the export output format."""
        model_output = self.model(x)
        return model_output.to_export_tensors()


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


def _read_nms_optimization_rows(checkpoint_path: Path) -> list[dict[str, str]]:
    """Read sibling NMS optimization CSV rows, returning an empty list on failures."""
    csv_path = checkpoint_path.parent / "nms_optimization.csv"
    if not csv_path.exists():
        return []

    try:
        with csv_path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        LOGGER.warning("Failed to read NMS optimization CSV at %s", csv_path, exc_info=True)
        return []


def _filter_rows_for_checkpoint(
    rows: list[dict[str, str]],
    checkpoint_path: Path,
) -> list[dict[str, str]]:
    """Filter optimization rows to the checkpoint kind inferred from filename."""
    checkpoint_kind = _checkpoint_kind_from_path(checkpoint_path)
    if checkpoint_kind is None:
        return rows
    return [
        row for row in rows if str(row.get("checkpoint", "")).strip().lower() == checkpoint_kind
    ]


def _parse_float_from_row(row: dict[str, str], key: str) -> float | None:
    """Parse a float from a CSV row key, returning None for missing/invalid values."""
    raw_value = row.get(key)
    if raw_value is None:
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _select_best_row_by_metric(
    rows: list[dict[str, str]],
    metric_key: str,
) -> dict[str, str] | None:
    """Select the row with the best metric value, tie-breaking by fitness."""
    ranked_rows: list[tuple[float, float, dict[str, str]]] = []
    for row in rows:
        metric_value = _parse_float_from_row(row, metric_key)
        if metric_value is None:
            continue
        fitness_value = _parse_float_from_row(row, "fitness")
        ranked_rows.append(
            (
                metric_value,
                float("-inf") if fitness_value is None else fitness_value,
                row,
            )
        )

    if not ranked_rows:
        return None
    ranked_rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked_rows[0][2]


def _read_best_nms_iou_threshold(
    checkpoint_path: Path,
    *,
    default_threshold: float = 0.1,
) -> float:
    """Read the best suggested NMS IoU threshold from sibling optimization CSV."""
    rows = _read_nms_optimization_rows(checkpoint_path)

    if not rows:
        return default_threshold

    checkpoint_rows = _filter_rows_for_checkpoint(rows, checkpoint_path)

    if not checkpoint_rows:
        return default_threshold

    suggested_rows = [row for row in checkpoint_rows if _looks_truthy(row.get("suggested"))]
    candidate_rows = suggested_rows or checkpoint_rows

    try:
        best_row = max(candidate_rows, key=lambda row: float(row.get("fitness", "-inf")))
        return float(best_row["nms_iou_threshold"])
    except (KeyError, ValueError):
        LOGGER.warning(
            "Failed to parse best NMS IoU threshold for checkpoint %s",
            checkpoint_path,
            exc_info=True,
        )
        return default_threshold


def _read_operating_point_confidence_metadata(checkpoint_path: Path) -> dict[str, float]:
    """Read confidence thresholds for selected operating points from optimization CSV."""
    rows = _read_nms_optimization_rows(checkpoint_path)
    if not rows:
        return {}

    checkpoint_rows = _filter_rows_for_checkpoint(rows, checkpoint_path)
    if not checkpoint_rows:
        return {}

    operating_points = (
        (
            "metrics/sensitivity_at_0.1_fppi",
            "metrics/confidence_at_0.1_fppi",
            "confidence_threshold_at_0.1_fp_per_image",
        ),
        (
            "metrics/sensitivity_at_0.5_fppi",
            "metrics/confidence_at_0.5_fppi",
            "confidence_threshold_at_0.5_fp_per_image",
        ),
        (
            "metrics/sensitivity_at_1.0_fppi",
            "metrics/confidence_at_1.0_fppi",
            "confidence_threshold_at_1.0_fp_per_image",
        ),
        (
            "metrics/f1",
            "metrics/confidence_at_max_f1",
            "confidence_threshold_at_max_f1",
        ),
    )

    metadata: dict[str, float] = {}
    for metric_key, confidence_key, metadata_key in operating_points:
        best_row = _select_best_row_by_metric(checkpoint_rows, metric_key)
        if best_row is None:
            continue
        confidence_value = _parse_float_from_row(best_row, confidence_key)
        if confidence_value is None:
            continue
        metadata[metadata_key] = confidence_value

    return metadata


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
    operating_point_metadata = _read_operating_point_confidence_metadata(checkpoint_path)
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
        **{key: _to_metadata_string(value) for key, value in operating_point_metadata.items()},
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

    return output_names


def _convert_to_float16_internal_dtype(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """Convert the graph to float16 while keeping float32 inputs/outputs.

    With dynamic input sizes the anchor grid is built inside the graph from the runtime shape
    (`Cast(int64 -> float)` feeding a `Range`). The converter would relabel the outputs of those
    casts as float16 without changing the casts themselves, producing an invalid graph, and
    `Range` has no float16 kernel in common runtimes. Keeping them in float32 makes the converter
    insert explicit casts at the boundary instead.
    """
    float32_cast_nodes = [
        node.name
        for node in onnx_model.graph.node
        if node.op_type == "Cast"
        and any(
            attribute.name == "to" and attribute.i == onnx.TensorProto.FLOAT
            for attribute in node.attribute
        )
    ]
    return float16.convert_float_to_float16(
        onnx_model,
        keep_io_types=True,
        op_block_list=[*float16.DEFAULT_OP_BLOCK_LIST, "Range"],
        node_block_list=float32_cast_nodes,
    )


def _write_onnx_metadata(onnx_model: onnx.ModelProto, metadata: dict[str, str]) -> None:
    """Merge and write metadata entries into an ONNX model in sorted key order."""
    merged_metadata = {entry.key: entry.value for entry in onnx_model.metadata_props}
    merged_metadata.update(metadata)

    del onnx_model.metadata_props[:]
    for key in sorted(merged_metadata):
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = merged_metadata[key]


MAX_DYNAMIC_BLOCKS = 256  # upper bound (in multiples of the size divisor) per spatial axis
_DYNAMIC_BATCH_AXIS_NAME = "batch"
_DYNAMIC_SPATIAL_AXIS_NAMES = ("depth", "height", "width")


def _build_dynamo_dynamic_shapes(size_divisor: int) -> dict[str, dict[int, object]]:
    """Describe dynamic batch/spatial axes of the wrapper input for `torch.export`.

    Spatial sizes are constrained to multiples of ``size_divisor`` (the coarsest model stride),
    which is exactly the set of input sizes the network can process.
    """
    return {
        "x": {
            0: Dim("batch", min=1, max=1024),
            **{
                axis: size_divisor * Dim(f"{name}_blocks", min=1, max=MAX_DYNAMIC_BLOCKS)
                for axis, name in enumerate(_DYNAMIC_SPATIAL_AXIS_NAMES, start=2)
            },
        }
    }


def _build_legacy_dynamic_axes(
    input_names: list[str],
    output_names: list[str],
) -> dict[str, dict[int, str]]:
    """Describe dynamic axes for the legacy (TorchScript based) ONNX exporter."""
    dynamic_axes: dict[str, dict[int, str]] = {
        input_names[0]: {
            0: _DYNAMIC_BATCH_AXIS_NAME,
            **dict(enumerate(_DYNAMIC_SPATIAL_AXIS_NAMES, start=2)),
        }
    }
    for name in output_names:
        if name == "prototypeMasks":
            dynamic_axes[name] = {
                0: _DYNAMIC_BATCH_AXIS_NAME,
                **{
                    axis: f"prototype_{axis_name}"
                    for axis, axis_name in enumerate(_DYNAMIC_SPATIAL_AXIS_NAMES, start=2)
                },
            }
        else:  # boxes, classScores, objectnessScores, maskCoefficients: (batch, channels, anchors)
            dynamic_axes[name] = {0: _DYNAMIC_BATCH_AXIS_NAME, 2: "num_anchors"}
    return dynamic_axes


def _export_torch_onnx_with_fallback(
    export_wrapper: ExportModelWrapper,
    export_inputs: tuple[torch.Tensor, ...],
    onnx_path: Path,
    input_names: list[str],
    output_names: list[str],
    *,
    size_divisor: int | None = None,
) -> None:
    """Export ONNX with a fallback to the legacy exporter for unsupported dynamo ops.

    If ``size_divisor`` is given, the batch and all spatial input axes are exported as dynamic.
    """
    export_input_argument: torch.Tensor | tuple[torch.Tensor, ...]
    export_input_argument = export_inputs if len(export_inputs) > 1 else export_inputs[0]
    dynamic = size_divisor is not None

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
                dynamic_shapes=_build_dynamo_dynamic_shapes(size_divisor) if dynamic else None,
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
                dynamic_axes=_build_legacy_dynamic_axes(input_names, output_names)
                if dynamic
                else None,
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
    *,
    dynamic_input_size: bool = True,
) -> Path:
    """Export a model to ONNX, optionally convert internal dtype, and attach metadata.

    Args:
        model: Model to export.
        input_shape: Channel-first (C, Z, Y, X) reference shape used to trace the model. With
            ``dynamic_input_size`` it only seeds the example input; the exported model accepts
            any batch size and any spatial size that is a multiple of ``model.size_divisor``.
        dest: Destination path (suffix is replaced with ``.onnx``).
        internal_dtype: Compute dtype inside the graph; inputs and outputs stay float32.
        metadata: Metadata entries to store in the ONNX file.
        dynamic_input_size: If False, the exported model is fixed to ``input_shape`` and batch 1.
    """
    if internal_dtype not in {"float16", "float32"}:
        msg = f"Unsupported internal ONNX dtype: {internal_dtype}"
        raise ValueError(msg)

    size_divisor = int(getattr(model, "size_divisor", 1))
    if dynamic_input_size:
        # Trace with at least two blocks per axis and batch 2 so the exporter does not
        # specialize axes of size one (batch 1 / a single block) to constants.
        example_shape = (
            input_shape[0],
            *(
                max(round_up(dim, size_divisor), 2 * size_divisor)
                for dim in input_shape[1:]
            ),
        )
        export_inputs = (torch.zeros((2, *example_shape), dtype=torch.float32, device="cpu"),)
    else:
        # Models trained with variable input sizes may have a patch size the network cannot
        # process directly, so the fixed size is rounded up to the supported grid.
        fixed_shape = (
            input_shape[0],
            *(round_up(dim, size_divisor) for dim in input_shape[1:]),
        )
        export_inputs = (torch.zeros((1, *fixed_shape), dtype=torch.float32, device="cpu"),)
    input_names = ["images"]
    model.to("cpu")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()

    for _ in range(2):
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
        size_divisor=size_divisor if dynamic_input_size else None,
    )

    export_metadata = {
        "dynamic_input_size": _to_metadata_string(dynamic_input_size),
        "size_divisor": _to_metadata_string(size_divisor),
        **(metadata or {}),
    }
    onnx_model = onnx.load(onnx_path)
    if internal_dtype == "float16":
        onnx_model = _convert_to_float16_internal_dtype(onnx_model)
    _write_onnx_metadata(onnx_model, export_metadata)
    onnx.save(onnx_model, onnx_path)

    return onnx_path
