from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from radyolo.config.config import DefaultConfig, ModelCheckpoint, load_model_checkpoint
from radyolo.models.layers import (
    C2PSA,
    SPPF,
    C3k2,
    Concat,
    Conv,
    Detect,
    ModelOutput,
    PrototypeMasks,
    PrototypeMasksHighResolution,
    PrototypeMasksHighResolutionOneSkip,
)
from radyolo.utils.util import intersect_dicts, make_divisible, select_device, unwrap_model

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class LoadedCheckpointModel:
    checkpoint: ModelCheckpoint
    train_config: DefaultConfig
    model: HardcodedModel
    device: torch.device


def get_model_min_stride(model: nn.Module | None, default: int = 8) -> int:
    if model is not None:
        model = unwrap_model(model)
    stride = getattr(model, "stride", None)
    if isinstance(stride, torch.Tensor):
        return int(stride.min().item())
    return default


class HardcodedModel(nn.Module):
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
        higher_resolution_prototypes: bool = True,
        separate_objectness_head: bool = False,
        include_stride4_output: bool = False,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.bbox_parametrization = bbox_parametrization
        self.bbox_center_offset_limit = bbox_center_offset_limit

        scales: dict[str, tuple[float, float, int]] = {
            "n": (0.50, 0.25, 1024),
            "s": (0.50, 0.50, 1024),
            "m": (0.50, 1.00, 512),
            "l": (1.00, 1.00, 512),
            "x": (1.00, 1.50, 512),
        }

        depth_multiplier, width_multiplier, max_channels = scales[scale]

        def make_divisible_(x: int) -> int:
            return make_divisible(min(x, max_channels) * width_multiplier, 8)

        def num_repeats_(n: int) -> int:
            return max(round(n * depth_multiplier), 1)

        self.enable_output_masks = enable_output_masks
        self.num_prototype_masks = num_prototype_masks
        self.higher_resolution_prototypes = higher_resolution_prototypes
        self.separate_objectness_head = separate_objectness_head
        self.include_stride4_output = include_stride4_output

        c3k2_use_c3k = scale in ("m", "l", "x")

        output_channels: list[int] = [input_channels]

        # BACKBONE

        in_channels = output_channels[-1]
        out_channels = make_divisible_(64)
        self.l0 = Conv(in_channels, out_channels, 3, 2)
        output_channels = [out_channels]

        in_channels = output_channels[-1]
        out_channels = make_divisible_(128)
        self.l1 = Conv(in_channels, out_channels, 3, 2)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        num_repeats = num_repeats_(2)
        out_channels = make_divisible_(256)
        self.l2 = C3k2(in_channels, out_channels, num_repeats, c3k2_use_c3k, 0.25)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        out_channels = make_divisible_(256)
        self.l3 = Conv(in_channels, out_channels, 3, 2)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        num_repeats = num_repeats_(2)
        out_channels = make_divisible_(512)
        self.l4 = C3k2(in_channels, out_channels, num_repeats, c3k2_use_c3k, 0.25)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        out_channels = make_divisible_(512)
        self.l5 = Conv(in_channels, out_channels, 3, 2)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        num_repeats = num_repeats_(2)
        out_channels = make_divisible_(512)
        self.l6 = C3k2(in_channels, out_channels, num_repeats, True)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        out_channels = make_divisible_(1024)
        self.l7 = Conv(in_channels, out_channels, 3, 2)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        num_repeats = num_repeats_(2)
        out_channels = make_divisible_(1024)
        self.l8 = C3k2(in_channels, out_channels, num_repeats, True)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        out_channels = make_divisible_(1024)
        self.l9 = SPPF(in_channels, out_channels, 5)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        num_repeats = num_repeats_(2)
        out_channels = make_divisible_(1024)
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
        out_channels = make_divisible_(512)
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
        out_channels = make_divisible_(256)
        self.l16 = C3k2(in_channels, out_channels, num_repeats, c3k2_use_c3k)
        output_channels.append(out_channels)

        # Optional P2 (stride-4) output branch: upsample P3 and fuse with stride-4 backbone (l2)
        # Channel sizes used across pyramid
        c_p2 = make_divisible_(128)
        c_p3 = make_divisible_(256)
        c_p4 = make_divisible_(512)
        c_p5 = make_divisible_(1024)
        if self.include_stride4_output:
            # These layers operate on feature maps produced in forward; we define them here
            self.p2_upsample = nn.Upsample(None, 2, "nearest")
            # Concat of upsampled P3 (c_p3) with stride-4 backbone feature (from l2, c_p3)
            self.p2_concat = Concat(1)
            self.p2_c3k2 = C3k2(c_p3 + c_p3, c_p2, num_repeats_(2), c3k2_use_c3k)

        in_channels = output_channels[-1]
        out_channels = make_divisible_(256)
        self.l17 = Conv(in_channels, out_channels, 3, 2)
        output_channels.append(out_channels)

        out_channels = sum(output_channels[x] for x in [-1, 13])
        self.l18 = Concat(1)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        num_repeats = num_repeats_(2)
        out_channels = make_divisible_(512)
        self.l19 = C3k2(in_channels, out_channels, num_repeats, c3k2_use_c3k)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        out_channels = make_divisible_(512)
        self.l20 = Conv(in_channels, out_channels, 3, 2)
        output_channels.append(out_channels)

        out_channels = sum(output_channels[x] for x in [-1, 10])
        self.l21 = Concat(1)
        output_channels.append(out_channels)

        in_channels = output_channels[-1]
        num_repeats = num_repeats_(2)
        out_channels = make_divisible_(1024)
        self.l22 = C3k2(in_channels, out_channels, num_repeats, True)
        output_channels.append(out_channels)

        if self.enable_output_masks:
            if self.higher_resolution_prototypes:
                self.prototype_layer = PrototypeMasksHighResolution(
                    output_channels[16], output_channels[1], output_channels[0], num_prototype_masks
                )
            else:
                self.prototype_layer = PrototypeMasks(
                    output_channels[16], num_prototype_masks * 4, num_prototype_masks
                )

            c4 = max(output_channels[16] // 4, num_prototype_masks)
            # Build coefficient heads dynamically for 3 or 4 detection levels
            detect_chs: list[int]
            if self.include_stride4_output:
                detect_chs = [c_p2, c_p3, c_p4, c_p5]
                self.prototype_layer = PrototypeMasksHighResolutionOneSkip(
                    c_p2, output_channels[0], num_prototype_masks
                )
            else:
                detect_chs = [c_p3, c_p4, c_p5]
            self.mask_coefficient_layer = nn.ModuleList(
                nn.Sequential(
                    Conv(ch, c4, 3),
                    Conv(c4, c4, 3),
                    nn.Conv3d(c4, num_prototype_masks, 1),
                )
                for ch in detect_chs
            )

        # Detect head: order from highest resolution to lowest
        detect_chs = [c_p3, c_p4, c_p5]
        if self.include_stride4_output:
            detect_chs = [c_p2, *detect_chs]
        self.l23 = Detect(
            num_classes,
            detect_chs,
            dfl_reg_max=dfl_reg_max,
            separate_objectness_head=self.separate_objectness_head,
            bbox_parametrization=self.bbox_parametrization,
            center_offset_limit=self.bbox_center_offset_limit,
        )
        s = 256
        sample_output = self.forward(torch.zeros(1, input_channels, s, s, s))
        self.l23.stride = torch.tensor(
            [s / x.shape[-2] for x in sample_output.raw_predictions]
        )
        self.stride = self.l23.stride
        self.l23.bias_init()

    def forward(self, x: torch.Tensor) -> ModelOutput:
        # BACKBONE
        x = o0 = self.l0(x)
        x = o1 = self.l1(x)
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
            if self.include_stride4_output:
                output.prototype_masks = self.prototype_layer(p2, o0)
            elif self.higher_resolution_prototypes:
                output.prototype_masks = self.prototype_layer(o16, o1, o0)
            else:
                output.prototype_masks = self.prototype_layer(o16)
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
        state_dict_intersected = intersect_dicts(state_dict, self.state_dict())
        self.load_state_dict(state_dict_intersected, strict=False)


def build_model_from_config(
    train_config: DefaultConfig,
    *,
    num_classes: int | None = None,
    state_dict: dict | None = None,
) -> HardcodedModel:
    model = HardcodedModel(
        input_channels=train_config.input_channels,
        num_classes=num_classes or train_config.num_classes,
        scale=train_config.model_scale,
        dfl_reg_max=train_config.dfl_reg_max,
        bbox_parametrization=train_config.bbox_parametrization,
        bbox_center_offset_limit=train_config.bbox_center_offset_limit,
        enable_output_masks=train_config.segmentation_output,
        num_prototype_masks=train_config.prototype_masks,
        higher_resolution_prototypes=train_config.higher_resolution_prototypes,
        separate_objectness_head=train_config.separate_objectness_head,
        include_stride4_output=train_config.include_stride4_output,
    )
    if state_dict is not None:
        model.load(state_dict)
    return model


def load_checkpoint_model(
    checkpoint_path: Path,
    *,
    device_override: str | None = None,
) -> LoadedCheckpointModel:
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
