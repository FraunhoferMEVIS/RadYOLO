from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchinfo import summary
from torchinfo.layer_info import LayerInfo


def _ensure_project_root_on_sys_path(script_path: Path | None = None) -> Path:
    project_root = (script_path or Path(__file__)).resolve().parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


class _TorchinfoModelWrapper(nn.Module):
    """Adapt HardcodedModel output to plain tensors for torchinfo."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.model(x).to_export_tensors()


def _patch_torchinfo_for_model_output(model_output_type: type[Any]) -> None:
    """Allow torchinfo hooks to process RadYOLO ModelOutput values."""
    if getattr(LayerInfo, "radyolo_model_output_patched", False):
        return

    original_calculate_size = LayerInfo.calculate_size

    def _calculate_size_with_model_output(
        inputs: object,
        batch_dim: int | None,
    ) -> tuple[list[int], int]:
        if isinstance(inputs, model_output_type):
            return original_calculate_size(inputs.to_export_tensors(), batch_dim)
        return original_calculate_size(inputs, batch_dim)

    LayerInfo.calculate_size = staticmethod(_calculate_size_with_model_output)
    LayerInfo.radyolo_model_output_patched = True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Load a RadYOLO checkpoint and print a torchinfo summary for the model.")
    )
    parser.add_argument("checkpoint_path", type=Path, help="Path to a .pt checkpoint file.")
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "Device override used to load the checkpoint model (for example: cpu, 0, cuda:0). "
            "Defaults to cpu."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used for the synthetic torchinfo input.",
    )
    parser.add_argument(
        "--input-shape-zyx",
        nargs=3,
        type=int,
        metavar=("Z", "Y", "X"),
        default=None,
        help=(
            "Override input spatial shape as z y x. By default, the checkpoint patch size is used."
        ),
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Maximum module nesting depth shown by torchinfo.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=(0, 1, 2),
        help="torchinfo verbosity level.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _ensure_project_root_on_sys_path()
    from radyolo.models.layers import ModelOutput  # noqa: PLC0415
    from radyolo.models.model import load_checkpoint_model  # noqa: PLC0415

    _patch_torchinfo_for_model_output(ModelOutput)

    if args.batch_size < 1:
        msg = "--batch-size must be >= 1"
        raise ValueError(msg)
    if args.input_shape_zyx is not None and any(dim < 1 for dim in args.input_shape_zyx):
        msg = "--input-shape-zyx requires all dimensions to be >= 1"
        raise ValueError(msg)

    loaded = load_checkpoint_model(args.checkpoint_path, device_override=args.device)
    train_config = loaded.train_config
    input_shape_zyx = (
        tuple(args.input_shape_zyx)
        if args.input_shape_zyx is not None
        else train_config.patch_size.get_zyx()
    )

    input_size = (args.batch_size, train_config.input_channels, *input_shape_zyx)
    wrapped_model = _TorchinfoModelWrapper(loaded.model).eval().to(loaded.device)

    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Device: {loaded.device}")

    with torch.inference_mode():
        summary(
            wrapped_model,
            input_size=input_size,
            device=str(loaded.device),
            depth=args.depth,
            verbose=args.verbose,
            col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
        )


if __name__ == "__main__":
    main()
