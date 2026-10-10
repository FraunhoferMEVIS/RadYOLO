from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import torch
from torch import nn

from radyolo.config.config import (
    DefaultConfig,
    HeadConvolutionType,
    ModelCheckpoint,
    PrototypeLayer,
    load_model_checkpoint,
)
from radyolo.models.layers import (
    C2PSA,
    SPPF,
    C3k2,
    Concat,
    Conv,
    Detect,
    ModelOutput,
    PrototypeMasksFullResolution,
    PrototypeMasksFullResolutionV2,
    PrototypeMasksFullResolutionV3,
    PrototypeMasksHalfResolution,
    PrototypeMasksHalfResolutionV2,
    PrototypeMasksHighResolution,
    build_detection_head_layers,
    resolve_head_hidden_channels,
)
from radyolo.utils.util import intersect_dicts, make_divisible, select_device, unwrap_model

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class LoadedCheckpointModel:
    """Container for a loaded checkpoint, config, model instance, and runtime device."""

    checkpoint: ModelCheckpoint
    train_config: DefaultConfig
    model: HardcodedModel
    device: torch.device


def get_model_min_stride(model: nn.Module | None, default: int = 8) -> int:
    """Return the smallest model stride or a fallback when unavailable."""
    if model is not None:
        model = unwrap_model(model)
    stride = getattr(model, "stride", None)
    if isinstance(stride, torch.Tensor):
        return int(stride.min().item())
    return default


class HardcodedModel(nn.Module):
    """Fixed YOLO-style 3D architecture with optional segmentation head."""

    MODEL_SCALES: ClassVar[dict[str, tuple[float, float, int]]] = {
        "n": (0.50, 0.25, 256),
        "s": (0.50, 0.50, 512),
        "m": (0.50, 1.00, 512),
        "l": (1.00, 1.00, 512),
        "x": (1.00, 1.50, 768),
    }

    # Overwritten at the end of __init__ once the strides are known (1 disables the shape check).
    size_divisor: int = 1

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        scale: Literal["n", "s", "m", "l", "x"],
        dfl_reg_max: int,
        bbox_parametrization: Literal["ltrb", "center_extents"],
        bbox_center_offset_limit: float,
        enable_output_masks: bool = True,
        num_prototype_masks: int = 8,
        prototype_layer: PrototypeLayer = "higher_resolution",
        segmentation_head_convolutions: HeadConvolutionType = "regular",
        segmentation_head_channel_factor: float = 0.25,
        separate_objectness_head: bool = False,
        include_stride4_output: bool = False,
        classification_head_convolutions: HeadConvolutionType = "separable",
        objectness_head_convolutions: HeadConvolutionType = "separable",
        box_head_convolutions: HeadConvolutionType = "regular",
        classification_head_channel_factor: float = 1.0,
        objectness_head_channel_factor: float = 1.0,
        box_head_channel_factor: float = 0.25,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.bbox_parametrization = bbox_parametrization
        self.bbox_center_offset_limit = bbox_center_offset_limit

        depth_multiplier, width_multiplier, max_channels = HardcodedModel.MODEL_SCALES[scale]

        def resolve_channel_size(x: int) -> int:
            return make_divisible(min(x * width_multiplier, max_channels), 8)

        def num_repeats_(n: int) -> int:
            return max(round(n * depth_multiplier), 1)

        self.enable_output_masks = enable_output_masks
        self.num_prototype_masks = num_prototype_masks
        self.prototype_layer_mode = prototype_layer
        self.separate_objectness_head = separate_objectness_head
        self.include_stride4_output = include_stride4_output

        c3k2_use_c3k = scale in ("m", "l", "x")

        # BACKBONE

        out_channels = resolve_channel_size(64)
        self.l0 = Conv(input_channels, out_channels, 3, 2)
        output_channels: list[int] = [out_channels]

        in_channels = output_channels[-1]
        out_channels = resolve_channel_size(128)
        self.l1 = Conv(in_channels, out_channels, 3, 2)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        num_repeats = num_repeats_(2)
        out_channels = resolve_channel_size(256)
        self.l2 = C3k2(in_channels, out_channels, num_repeats, c3k2_use_c3k, 0.25)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        out_channels = resolve_channel_size(256)
        self.l3 = Conv(in_channels, out_channels, 3, 2)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        num_repeats = num_repeats_(2)
        out_channels = resolve_channel_size(512)
        self.l4 = C3k2(in_channels, out_channels, num_repeats, c3k2_use_c3k, 0.25)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        out_channels = resolve_channel_size(512)
        self.l5 = Conv(in_channels, out_channels, 3, 2)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        num_repeats = num_repeats_(2)
        out_channels = resolve_channel_size(512)
        self.l6 = C3k2(in_channels, out_channels, num_repeats, True)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        out_channels = resolve_channel_size(1024)
        self.l7 = Conv(in_channels, out_channels, 3, 2)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        num_repeats = num_repeats_(2)
        out_channels = resolve_channel_size(1024)
        self.l8 = C3k2(in_channels, out_channels, num_repeats, True)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        out_channels = resolve_channel_size(1024)
        self.l9 = SPPF(in_channels, out_channels, 5)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        num_repeats = num_repeats_(2)
        out_channels = resolve_channel_size(1024)
        self.l10 = C2PSA(in_channels, out_channels, num_repeats)
        output_channels.append(out_channels)

        # HEAD

        out_channels = output_channels[-1]
        self.l11 = nn.Upsample(None, 2, "nearest")
        output_channels.append(out_channels)

        out_channels = sum(output_channels[x] for x in [-1, 6])
        self.l12 = Concat(1)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        num_repeats = num_repeats_(2)
        out_channels = resolve_channel_size(512)
        self.l13 = C3k2(in_channels, out_channels, num_repeats, c3k2_use_c3k)
        output_channels.append(out_channels)

        out_channels = output_channels[-1]
        self.l14 = nn.Upsample(None, 2, "nearest")
        output_channels.append(out_channels)

        out_channels = sum(output_channels[x] for x in [-1, 4])
        self.l15 = Concat(1)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        num_repeats = num_repeats_(2)
        out_channels = resolve_channel_size(256)
        self.l16 = C3k2(in_channels, out_channels, num_repeats, c3k2_use_c3k)
        output_channels.append(out_channels)

        # Optional P2 (stride-4) output branch: upsample P3 and fuse with stride-4 backbone (l2)
        # Channel sizes used across pyramid
        c_p2 = resolve_channel_size(128)
        c_p3 = resolve_channel_size(256)
        c_p4 = resolve_channel_size(512)
        c_p5 = resolve_channel_size(1024)
        if self.include_stride4_output:
            # These layers operate on feature maps produced in forward; we define them here
            self.p2_upsample = nn.Upsample(None, 2, "nearest")
            # Concat of upsampled P3 (c_p3) with stride-4 backbone feature (from l2, c_p3)
            self.p2_concat = Concat(1)
            self.p2_c3k2 = C3k2(c_p3 + c_p3, c_p2, num_repeats_(2), c3k2_use_c3k)

        in_channels = output_channels[-1]
        out_channels = resolve_channel_size(256)
        self.l17 = Conv(in_channels, out_channels, 3, 2)
        output_channels.append(out_channels)

        out_channels = sum(output_channels[x] for x in [-1, 13])
        self.l18 = Concat(1)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        num_repeats = num_repeats_(2)
        out_channels = resolve_channel_size(512)
        self.l19 = C3k2(in_channels, out_channels, num_repeats, c3k2_use_c3k)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        out_channels = resolve_channel_size(512)
        self.l20 = Conv(in_channels, out_channels, 3, 2)
        output_channels.append(out_channels)

        out_channels = sum(output_channels[x] for x in [-1, 10])
        self.l21 = Concat(1)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        num_repeats = num_repeats_(2)
        out_channels = resolve_channel_size(1024)
        self.l22 = C3k2(in_channels, out_channels, num_repeats, True)
        output_channels.append(out_channels)

        detect_chs: list[int] = (
            [c_p2, c_p3, c_p4, c_p5] if self.include_stride4_output else [c_p3, c_p4, c_p5]
        )

        if self.enable_output_masks:
            if self.prototype_layer_mode == "full_resolution":
                self.prototype_layer = PrototypeMasksFullResolution(
                    input_channels,
                    output_channels[0],
                    output_channels[2],
                    output_channels[16],
                    output_channels[19],
                    num_prototype_masks,
                )
            elif self.prototype_layer_mode == "full_resolution_v2":
                self.prototype_layer = PrototypeMasksFullResolutionV2(
                    input_channels,
                    output_channels[0],
                    output_channels[2],
                    output_channels[16],
                    output_channels[19],
                    num_prototype_masks,
                )
            elif self.prototype_layer_mode == "full_resolution_v3":
                self.prototype_layer = PrototypeMasksFullResolutionV3(
                    input_channels,
                    output_channels[0],
                    output_channels[2],
                    output_channels[16],
                    output_channels[19],
                    num_prototype_masks,
                )
            elif self.prototype_layer_mode == "half_resolution":
                self.prototype_layer = PrototypeMasksHalfResolution(
                    output_channels[0],
                    output_channels[2],
                    output_channels[16],
                    output_channels[19],
                    num_prototype_masks,
                )
            elif self.prototype_layer_mode == "half_resolution_v2":
                self.prototype_layer = PrototypeMasksHalfResolutionV2(
                    output_channels[0],
                    output_channels[2],
                    output_channels[16],
                    output_channels[19],
                    num_prototype_masks,
                )
            else:
                self.prototype_layer = PrototypeMasksHighResolution(
                    output_channels[0], output_channels[2], output_channels[16], num_prototype_masks
                )

            segmentation_hidden_channels = resolve_head_hidden_channels(
                detect_chs[0],
                segmentation_head_channel_factor,
                "segmentation head",
            )
            # Build coefficient heads dynamically for 3 or 4 detection levels
            self.mask_coefficient_layer = nn.ModuleList(
                build_detection_head_layers(
                    ch,
                    segmentation_hidden_channels,
                    num_prototype_masks,
                    segmentation_head_convolutions,
                )
                for ch in detect_chs
            )

        self.l23 = Detect(
            num_classes,
            detect_chs,
            dfl_reg_max=dfl_reg_max,
            separate_objectness_head=self.separate_objectness_head,
            bbox_parametrization=self.bbox_parametrization,
            center_offset_limit=self.bbox_center_offset_limit,
            classification_head_convolutions=classification_head_convolutions,
            objectness_head_convolutions=objectness_head_convolutions,
            box_head_convolutions=box_head_convolutions,
            classification_head_channel_factor=classification_head_channel_factor,
            objectness_head_channel_factor=objectness_head_channel_factor,
            box_head_channel_factor=box_head_channel_factor,
        )
        s = 256
        sample_output = self.forward(torch.zeros(1, input_channels, s, s, s))
        self.l23.stride = torch.tensor([s / x.shape[-2] for x in sample_output.raw_predictions])
        self.stride = self.l23.stride
        # The anchor grid is derived from the input at runtime, so any input size works as long
        # as every spatial dimension is a multiple of the coarsest stride (otherwise the
        # upsampled feature maps no longer line up with their skip connections).
        self.size_divisor = int(self.stride.max().item())
        self.l23.bias_init()

    def check_input_shape(self, x_input: torch.Tensor) -> None:
        """Raise a descriptive error if the spatial input size is not supported."""
        spatial_shape = tuple(int(dim) for dim in x_input.shape[-3:])
        if any(dim % self.size_divisor for dim in spatial_shape):
            msg = (
                f"Spatial input size (z, y, x) = {spatial_shape} is not supported: every "
                f"dimension has to be a multiple of {self.size_divisor}. Pad the input "
                "(e.g. with radyolo.data.preprocessing.pad_to_multiple) and crop the outputs."
            )
            raise ValueError(msg)

    def forward(
        self,
        x_input: torch.Tensor,
    ) -> ModelOutput:
        is_tracing = (
            torch.jit.is_tracing()
            or torch.compiler.is_compiling()
            or torch.onnx.is_in_onnx_export()
        )
        if not is_tracing:
            self.check_input_shape(x_input)

        # BACKBONE
        x = o0 = self.l0(x_input)
        x = self.l1(x)
        x = o2 = self.l2(x)
        x = self.l3(x)
        x = o4 = self.l4(x)
        x = self.l5(x)
        x = o6 = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = o10 = self.l10(x)

        # HEAD
        x = self.l11(x)
        x = self.l12([x, o6])
        x = o13 = self.l13(x)

        x = self.l14(x)
        x = self.l15([x, o4])
        x = o16 = self.l16(x)

        # Optional P2 (stride-4) feature from upsampled P3 fused with stride-4 backbone feature
        if self.include_stride4_output:
            p2 = self.p2_c3k2(self.p2_concat([self.p2_upsample(o16), o2]))

        x = self.l17(x)
        x = self.l18([x, o13])
        x = o19 = self.l19(x)

        x = self.l20(x)
        x = self.l21([x, o10])
        x = o22 = self.l22(x)

        if self.include_stride4_output:
            output = self.l23([p2, o16, o19, o22])
        else:
            output = self.l23([o16, o19, o22])

        if self.enable_output_masks:
            if self.prototype_layer_mode in {
                "full_resolution",
                "full_resolution_v2",
                "full_resolution_v3",
            }:
                output.prototype_masks = self.prototype_layer(x_input, o0, o2, o16, o19)
            elif self.prototype_layer_mode in {"half_resolution", "half_resolution_v2"}:
                output.prototype_masks = self.prototype_layer(o0, o2, o16, o19)
            else:
                output.prototype_masks = self.prototype_layer(o0, o2, o16)
            if output.prototype_masks is None:
                msg = "Prototype masks are not available"
                raise RuntimeError(msg)
            batch_size = output.prototype_masks.shape[0]
            if self.include_stride4_output:
                output.mask_coefficients = torch.cat(
                    [
                        self.mask_coefficient_layer[0](p2).view(
                            batch_size, self.num_prototype_masks, -1
                        ),
                        self.mask_coefficient_layer[1](o16).view(
                            batch_size, self.num_prototype_masks, -1
                        ),
                        self.mask_coefficient_layer[2](o19).view(
                            batch_size, self.num_prototype_masks, -1
                        ),
                        self.mask_coefficient_layer[3](o22).view(
                            batch_size, self.num_prototype_masks, -1
                        ),
                    ],
                    2,
                )
            else:
                output.mask_coefficients = torch.cat(
                    [
                        self.mask_coefficient_layer[0](o16).view(
                            batch_size, self.num_prototype_masks, -1
                        ),
                        self.mask_coefficient_layer[1](o19).view(
                            batch_size, self.num_prototype_masks, -1
                        ),
                        self.mask_coefficient_layer[2](o22).view(
                            batch_size, self.num_prototype_masks, -1
                        ),
                    ],
                    2,
                )
        return output

    def load(self, state_dict: dict) -> None:
        """Load only checkpoint parameters that intersect the current model state."""
        state_dict_intersected = intersect_dicts(state_dict, self.state_dict())
        self.load_state_dict(state_dict_intersected, strict=False)


def build_model_from_config(
    train_config: DefaultConfig,
    *,
    num_classes: int | None = None,
    state_dict: dict | None = None,
) -> HardcodedModel:
    """Instantiate a model from train config and optionally load a partial state dict."""
    model = HardcodedModel(
        input_channels=train_config.input_channels,
        num_classes=num_classes or train_config.num_classes,
        scale=train_config.model_scale,
        dfl_reg_max=train_config.dfl_reg_max,
        bbox_parametrization=train_config.bbox_parametrization,
        bbox_center_offset_limit=train_config.bbox_center_offset_limit,
        enable_output_masks=train_config.segmentation_output,
        num_prototype_masks=train_config.prototype_masks,
        prototype_layer=train_config.prototype_layer,
        segmentation_head_convolutions=train_config.segmentation_head_convolutions,
        segmentation_head_channel_factor=train_config.segmentation_head_channel_factor,
        separate_objectness_head=train_config.separate_objectness_head,
        include_stride4_output=train_config.include_stride4_output,
        classification_head_convolutions=train_config.classification_head_convolutions,
        objectness_head_convolutions=train_config.objectness_head_convolutions,
        box_head_convolutions=train_config.box_head_convolutions,
        classification_head_channel_factor=train_config.classification_head_channel_factor,
        objectness_head_channel_factor=train_config.objectness_head_channel_factor,
        box_head_channel_factor=train_config.box_head_channel_factor,
    )
    if state_dict is not None:
        model.load(state_dict)
    return model


def load_checkpoint_model(
    checkpoint_path: Path,
    *,
    device_override: str | None = None,
) -> LoadedCheckpointModel:
    """Load checkpoint artifacts and return an eval-ready model bundle on the selected device."""
    checkpoint = load_model_checkpoint(checkpoint_path)
    train_config = checkpoint["train_config"]
    device = select_device(device_override or getattr(train_config, "device", ""))
    model = build_model_from_config(
        train_config,
        state_dict=checkpoint["model_state_dict"],
    )
    model.eval().to(device)
    model.device = device  # type: ignore[attr-defined]
    return LoadedCheckpointModel(
        checkpoint=checkpoint,
        train_config=train_config,
        model=model,
        device=device,
    )
