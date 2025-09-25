from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import torch
from torch import nn

from yolo11_3d.models.layers import (
    C2PSA,
    SPPF,
    C3k2,
    Concat,
    Conv,
    Detect,
    PrototypeMasks,
    PrototypeMasksHighResolution,
    PrototypeMasksHighResolutionOneSkip,
)
from yolo11_3d.nn.util import intersect_dicts, make_divisible
from yolo11_3d.ultralytics.utils.loss import v8DetectionLoss

if TYPE_CHECKING:
    from yolo11_3d.nn.config import ModelCheckpoint
    from yolo11_3d.data.data import Batch


class HardcodedModel(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        scale: Literal["n", "s", "m", "l", "x"],
        box: float,
        cls: float,
        dfl: float,
        mask_gain: float,
        class_separate_gain: float,
        device: torch.device,
        enable_output_masks: bool = True,
        num_prototype_masks: int = 8,
        higher_resolution_prototypes: bool = True,
        separate_objectness_head: bool = False,
        include_stride4_output: bool = False,
        objectness_loss: Literal["bce", "focal"] = "bce",
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

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
                    output_channels[16],
                    output_channels[1],
                    output_channels[0],
                    num_prototype_masks
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
                    c_p2,
                    output_channels[0],
                    num_prototype_masks
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
            detect_chs = [c_p2] + detect_chs
        self.l23 = Detect(num_classes,
                          detect_chs,
                          separate_objectness_head=self.separate_objectness_head)
        s = 256
        self.l23.stride = torch.tensor(
            [
                s / x.shape[-2]
                for x in self.forward(torch.zeros(1, input_channels, s, s, s))[0]
            ]
        )
        self.stride = self.l23.stride
        self.l23.bias_init()

        self.criterion = v8DetectionLoss(
            self, box, cls, dfl, mask_gain, class_separate_gain, self.l23, device, tal_beta=1.0, tal_fallback=True,
            enable_mask_loss=enable_output_masks, separate_objectness_head=self.separate_objectness_head,
            objectness_loss=objectness_loss
        )

    @overload
    def forward(
        self, x: torch.Tensor, *args, **kwargs
    ) -> (
        list[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]
    ): ...

    @overload
    def forward(
        self, x: Batch, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def forward(
        self, x: torch.Tensor | Batch, *args, **kwargs
    ) -> (
        list[torch.Tensor]
        | torch.Tensor
        | tuple[torch.Tensor, list[torch.Tensor]]
        | tuple[torch.Tensor, torch.Tensor]
    ):
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def loss(
        self,
        batch: Batch,
        preds: (
            list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]] | None
        ) = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if preds is None:
            prediction = self.forward(batch["img"])
            if isinstance(prediction, torch.Tensor):
                raise TypeError(prediction)
            preds = prediction
        return self.criterion(preds, batch)

    def load(self, weights: ModelCheckpoint | nn.Module) -> None:
        model = weights["ema"] if isinstance(weights, dict) else weights
        csd = model.float().state_dict()
        csd = intersect_dicts(csd, self.state_dict())
        self.load_state_dict(csd, strict=False)

    def predict(self, x: torch.Tensor) -> tuple[
        list[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]],
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
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
            detections = self.l23([p2, o16, o19, o22])
        else:
            detections = self.l23([o16, o19, o22])

        if self.enable_output_masks:
            if self.include_stride4_output:
                prototype_masks = self.prototype_layer(p2, o0)
            elif self.higher_resolution_prototypes:
                prototype_masks = self.prototype_layer(o16, o1, o0)
            else:
                prototype_masks = self.prototype_layer(o16)
            batch_size = prototype_masks.shape[0]
            if self.include_stride4_output:
                mask_coefficients = torch.cat(
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
                mask_coefficients = torch.cat(
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
            return detections, mask_coefficients, prototype_masks
        else:
            return detections, None, None

