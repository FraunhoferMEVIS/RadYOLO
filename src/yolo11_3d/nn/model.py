from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, overload

import torch
from torch import nn

from yolo11_3d.nn.layers import (
    C2PSA,
    SPPF,
    C3k2,
    Concat,
    Conv,
    Detect,
)
from yolo11_3d.nn.util import intersect_dicts, make_divisible
from yolo11_3d.ultralytics.utils.loss import v8DetectionLoss

if TYPE_CHECKING:
    from yolo11_3d.nn.config import ModelCheckpoint
    from yolo11_3d.nn.data import Batch


class BaseModel(nn.Module, ABC):
    @overload
    def forward(
        self, x: torch.Tensor, *args, **kwargs
    ) -> list[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]: ...

    @overload
    def forward(self, x: dict[str, Any], *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def forward(
        self, x: torch.Tensor | dict[str, Any], *args, **kwargs
    ) -> (
        list[torch.Tensor]
        | torch.Tensor
        | tuple[torch.Tensor, list[torch.Tensor]]
        | tuple[torch.Tensor, torch.Tensor]
    ): ...

    @abstractmethod
    def loss(
        self,
        batch: Batch,
        preds: list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def load(self, weights: ModelCheckpoint | nn.Module) -> None: ...

    @abstractmethod
    def predict(
        self, x: torch.Tensor, augment: bool = False
    ) -> list[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]: ...


class HardcodedModel(BaseModel):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        scale: Literal["n", "s", "m", "l", "x"],
        box: float,
        cls: float,
        dfl: float,
        device: torch.device,
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

        self.l23 = Detect(num_classes, [output_channels[x] for x in [16, 19, 22]])
        s = 256
        self.l23.stride = torch.tensor(
            [s / x.shape[-2] for x in self.forward(torch.zeros(1, input_channels, s, s, s))]
        )
        self.stride = self.l23.stride
        self.l23.bias_init()

        self.criterion = v8DetectionLoss(self, box, cls, dfl, self.l23, device, tal_beta=1.0, tal_fallback=True)

    @overload
    def forward(
        self, x: torch.Tensor, *args, **kwargs
    ) -> list[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]: ...

    @overload
    def forward(self, x: Batch, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]: ...

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
        preds: list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]] | None = None,
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

    def predict(
        self, x: torch.Tensor, augment: bool = False
    ) -> list[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        # BACKBONE
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
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

        x = self.l17(x)
        x = self.l18([x, o13])
        x = o19 = self.l19(x)

        x = self.l20(x)
        x = self.l21([x, o10])
        x = o22 = self.l22(x)

        return self.l23([o16, o19, o22])
