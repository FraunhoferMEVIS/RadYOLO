from __future__ import annotations

import math
from typing import TYPE_CHECKING, assert_never, cast, overload

import torch
from torch import nn

from radyolo.training.tal import BoxParametrization, decode_bboxes, make_anchors

if TYPE_CHECKING:
    from collections.abc import Sequence


@overload
def autopad[T: int | tuple[int, int] | tuple[int, int, int]](
    kernel_size: T, padding: None, dilation: int
) -> T: ...


@overload
def autopad[U: int | str | tuple[int, int] | tuple[int, int, int]](
    kernel_size: int | tuple[int, int] | tuple[int, int, int], padding: U, dilation: int
) -> U: ...


def autopad[
    T: int | tuple[int, int] | tuple[int, int, int],
    U: int | tuple[int, int] | tuple[int, int, int] | str,
](
    kernel_size: T,
    padding: U | None = None,
    dilation: int = 1,
) -> U | T:
    """Pad to 'same' shape outputs."""
    if padding is not None:
        return padding

    if dilation > 1:

        def dilate(x: int) -> int:
            return dilation * (x - 1) + 1

        if isinstance(kernel_size, int):
            kernel_size = dilate(kernel_size)
        elif isinstance(kernel_size, tuple):
            kernel_size = cast(T, tuple(dilate(x) for x in kernel_size))
        else:
            assert_never(kernel_size)

    def pad(x: int) -> int:
        return x // 2

    if isinstance(kernel_size, int):
        return pad(kernel_size)
    if isinstance(kernel_size, tuple):
        return cast(T, tuple([pad(x) for x in kernel_size]))
    assert_never(kernel_size)


class Conv(nn.Module):
    default_activation = nn.SiLU(inplace=True)

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 1,
        stride: int = 1,
        padding: int | str | None = None,
        n_groups: int = 1,
        dilation: int = 1,
        activation: bool | nn.Module = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            autopad(kernel_size, padding, dilation),
            groups=n_groups,
            dilation=dilation,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.03)
        self.activation = (
            self.default_activation
            if activation is True
            else activation
            if isinstance(activation, nn.Module)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.batch_norm(self.conv(x)))


class DWConv(Conv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 1,
        stride: int = 1,
        dilation: int = 1,
        activation: bool | nn.Module = True,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            n_groups=math.gcd(in_channels, out_channels),
            dilation=dilation,
            activation=activation,
        )


class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, in_channels: int = 16, min_value: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(in_channels, dtype=torch.float) + min_value
        self.conv.weight.data[:] = nn.Parameter(x.view(1, in_channels, 1, 1))
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape
        return self.conv(x.view(b, 6, self.in_channels, a).transpose(2, 1).softmax(1)).view(b, 6, a)


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        n_groups: int = 1,
        kernel_sizes: tuple[int | tuple[int, int, int], int | tuple[int, int, int]] = (3, 3),
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        c_ = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, c_, kernel_sizes[0], 1)
        self.conv2 = Conv(c_, out_channels, kernel_sizes[1], 1, n_groups=n_groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class C2f(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_bottlenecks: int = 1,
        shortcut: bool = False,
        n_groups: int = 1,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_channels = int(out_channels * expansion)
        self.in_conv = Conv(in_channels, 2 * self.hidden_channels, 1, 1)
        self.out_conv = Conv((2 + n_bottlenecks) * self.hidden_channels, out_channels, 1)
        self.m = nn.ModuleList(
            Bottleneck(
                self.hidden_channels,
                self.hidden_channels,
                shortcut,
                n_groups,
                kernel_sizes=((3, 3, 3), (3, 3, 3)),
                expansion=1.0,
            )
            for _ in range(n_bottlenecks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_layers = list(self.in_conv(x).chunk(2, 1))
        for m in self.m:
            hidden_layers.append(m(hidden_layers[-1]))
        return self.out_conv(torch.cat(hidden_layers, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        split_output = self.in_conv(x).split((self.hidden_channels, self.hidden_channels), 1)
        hidden_layers = [split_output[0], split_output[1]]
        for m in self.m:
            hidden_layers.append(m(hidden_layers[-1]))
        return self.out_conv(torch.cat(hidden_layers, 1))


class C3k2(C2f):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_bottlenecks: int = 1,
        c3k: bool = False,
        expansion: float = 0.5,
        n_groups: int = 1,
        shortcut: bool = True,
    ) -> None:
        super().__init__(in_channels, out_channels, n_bottlenecks, shortcut, n_groups, expansion)
        self.m = nn.ModuleList(
            C3k(self.hidden_channels, self.hidden_channels, 2, shortcut, n_groups)
            if c3k
            else Bottleneck(self.hidden_channels, self.hidden_channels, shortcut, n_groups)
            for _ in range(n_bottlenecks)
        )


class C3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_bottlenecks: int = 1,
        shortcut: bool = True,
        n_groups: int = 1,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(in_channels, hidden_channels, 1, 1)
        self.out_conv = Conv(2 * hidden_channels, out_channels, 1)
        self.hidden_layer = nn.Sequential(
            *(
                Bottleneck(
                    hidden_channels,
                    hidden_channels,
                    shortcut,
                    n_groups,
                    kernel_sizes=((1, 1, 1), (3, 3, 3)),
                    expansion=1.0,
                )
                for _ in range(n_bottlenecks)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_conv(torch.cat((self.hidden_layer(self.conv1(x)), self.conv2(x)), 1))


class C3k(C3):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_bottlenecks: int = 1,
        shortcut: bool = True,
        n_groups: int = 1,
        expansion: float = 0.5,
        kernel_size: int | tuple[int, int, int] = 3,
    ) -> None:
        super().__init__(in_channels, out_channels, n_bottlenecks, shortcut, n_groups, expansion)
        hidden_channels = int(out_channels * expansion)
        self.hidden_layer = nn.Sequential(
            *(
                Bottleneck(
                    hidden_channels,
                    hidden_channels,
                    shortcut,
                    n_groups,
                    kernel_sizes=(kernel_size, kernel_size),
                    expansion=1.0,
                )
                for _ in range(n_bottlenecks)
            )
        )


class SPPF(nn.Module):
    """Spatial Pyramid Pooling Fast."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        hidden_channels = in_channels // 2
        self.n_concat_layers = 4
        self.input_conv = Conv(in_channels, hidden_channels, 1, 1)
        self.output_conv = Conv(hidden_channels * self.n_concat_layers, out_channels, 1, 1)
        self.hidden_layer = nn.MaxPool3d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        concat_layers = [self.input_conv(x)]
        for _ in range(self.n_concat_layers - 1):
            concat_layers.append(self.hidden_layer(concat_layers[-1]))
        return self.output_conv(torch.cat(concat_layers, 1))


class Attention(nn.Module):
    def __init__(self, in_dim: int, num_heads: int = 8, attention_ratio: float = 0.5) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.key_dim = int(self.head_dim * attention_ratio)
        self.scale: float = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = in_dim + nh_kd * 2
        self.query_key_value = Conv(in_dim, h, 1, activation=False)
        self.projection = Conv(in_dim, in_dim, 1, activation=False)
        self.positional_encoding = Conv(in_dim, in_dim, 3, 1, n_groups=in_dim, activation=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channel, depth, height, width = x.shape
        n_pixels = depth * height * width
        query_key_value: torch.Tensor = self.query_key_value(x)
        query, key, value = query_key_value.view(
            batch, self.num_heads, self.key_dim * 2 + self.head_dim, n_pixels
        ).split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        if not isinstance(query, torch.Tensor):
            raise TypeError
        if not isinstance(key, torch.Tensor):
            raise TypeError
        if not isinstance(value, torch.Tensor):
            raise TypeError

        attention = (query.transpose(-2, -1) @ key) * self.scale
        attention = attention.softmax(dim=-1)

        x = (value @ attention.transpose(-2, -1)).view(
            batch, channel, depth, height, width
        ) + self.positional_encoding(value.reshape(batch, channel, depth, height, width))
        return self.projection(x)


class PSABlock(nn.Module):
    """PSABlock implements a Position-Sensitive Attention block.

    Attributes:
        feed_forward_network (nn.Sequential): Feed-forward neural network module.
        attention (Attention): Multi-head attention module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: The forward pass.
    """

    def __init__(
        self, channels: int, attention_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True
    ) -> None:
        """Initializes the PSABlock.

        Args:
            channels (int): the number of channels to be used throughout the whole block.
            attention_ratio (float): The attention ratio used for the `Attention` block.
                For further information see `Attention`-Block.
            num_heads (int): the number of attention heads used for the `Attention` block.
                For further information see `Attention`-Block.
            shortcut (bool): Whether to add shortcut connections or not.
        """
        super().__init__()

        self.attention = Attention(channels, attention_ratio=attention_ratio, num_heads=num_heads)
        self.feed_forward_network = nn.Sequential(
            Conv(channels, channels * 2, 1), Conv(channels * 2, channels, 1, activation=False)
        )
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x) if self.add else self.attention(x)
        x = x + self.feed_forward_network(x) if self.add else self.feed_forward_network(x)
        return x  # noqa: RET504


class C2PSA(nn.Module):
    """Cross Stage Partial Self-Attention.

    This module implements a convolutional block with a configurable number of
    Partial Self-Attention (PSA) blocks.

    Attributes:
        hidden_channels (int): Number of hidden channels.
        in_conv (Conv): Initial 1x1 convolution layer to adjust the number of input channels
            to 2*`hidden_channels`.
        out_conv (Conv): Final 1x1 convolution layer to adjust the number of output channels
            to `hidden_channels`.
        m (nn.Sequential): Sequential container of `n_hidden_layers` times `PSABlock` modules.
    """

    def __init__(
        self, in_channels: int, out_channels: int, n_hidden_layers: int = 1, expansion: float = 0.5
    ) -> None:
        """Initializes the C2PSA module.

        Args:
            in_channels (int): number of input channels for initial 1x1 convolution layer.
            out_channels(int): number of output channels for final 1x1 convolution layer.
            n_hidden_layers (int): number of PSABlock layers.
            expansion (float): expansion factor for `in_channels` to determine number of hidden channels.
        """
        super().__init__()
        if in_channels != out_channels:
            msg = (
                f"Input channels have to be equals to output channels in C2PSA, "
                f"but were {in_channels} and {out_channels}."
            )
            raise ValueError(msg)
        self.hidden_channels = int(in_channels * expansion)
        self.in_conv = Conv(in_channels, 2 * self.hidden_channels, 1, 1)
        self.out_conv = Conv(2 * self.hidden_channels, in_channels, 1)

        self.m = nn.Sequential(
            *(
                PSABlock(
                    self.hidden_channels, attention_ratio=0.5, num_heads=self.hidden_channels // 64
                )
                for _ in range(n_hidden_layers)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.in_conv(x).split((self.hidden_channels, self.hidden_channels), dim=1)
        b = self.m(b)
        return self.out_conv(torch.cat((a, b), 1))


class Concat(nn.Module):
    def __init__(self, dimension: int = 1) -> None:
        super().__init__()
        self.d: int = dimension

    def forward(self, x: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, self.d)


class Detect(nn.Module):
    export = False
    max_det = 300
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(
        self,
        num_classes: int = 80,
        ch: Sequence[int] = (),
        *,
        dfl_reg_max: int = 16,
        separate_objectness_head: bool = False,
        bbox_parametrization: BoxParametrization = "ltrb",
        center_offset_limit: float = 1.2,
    ) -> None:
        super().__init__()
        self.nc = num_classes  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = dfl_reg_max  # Number of non-negative DFL bins.
        self.use_dfl = self.reg_max > 1
        self.bbox_parametrization = bbox_parametrization
        self.center_offset_limit = center_offset_limit
        if self.use_dfl and self.bbox_parametrization != "ltrb":
            msg = "DFL is only supported with bbox_parametrization='ltrb'"
            raise ValueError(msg)
        self.dfl_min = -1.0 if self.use_dfl else 0.0
        self.dfl_bins = self.reg_max + int(abs(self.dfl_min)) if self.use_dfl else 1
        self.reg_channels = self.dfl_bins * 6 if self.use_dfl else 6
        self.separate_objectness_head = separate_objectness_head
        self.no = (
            num_classes + 1 if self.separate_objectness_head else num_classes
        ) + self.reg_channels  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = (
            max((16, ch[0] // 4, self.reg_channels)),
            max(ch[0], min(self.nc, 100)),
        )  # channels
        self.cv_boxes = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv3d(c2, self.reg_channels, 1),
            )
            for x in ch
        )
        self.cv_classes = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv3d(c3, self.nc, 1),
            )
            for x in ch
        )
        if self.separate_objectness_head:
            self.cv_obj = nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv3d(c3, 1, 1),
                )
                for x in ch
            )
        self.dfl = DFL(self.dfl_bins, min_value=self.dfl_min) if self.use_dfl else nn.Identity()

    def forward(
        self, pred_raw: list[torch.Tensor]
    ) -> list[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        for i in range(self.nl):
            reg = self.cv_boxes[i](pred_raw[i])
            cls = self.cv_classes[i](pred_raw[i])
            if self.separate_objectness_head:
                obj = self.cv_obj[i](pred_raw[i])
                pred_raw[i] = torch.cat((reg, cls, obj), 1)
            else:
                pred_raw[i] = torch.cat((reg, cls), 1)
        if self.training:
            return pred_raw
        pred_with_final_layers = self._inference(pred_raw)
        if self.export:
            return pred_with_final_layers
        return (pred_with_final_layers, pred_raw)

    def _inference(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        shape = x[0].shape
        if self.shape != shape:
            self.anchors, self.strides = (
                x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.separate_objectness_head:
            box, cls, obj = x_cat.split((self.reg_channels, self.nc, 1), 1)
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
            cls_prob = cls.softmax(1)
            obj_prob = obj.sigmoid()
            return (dbox, cls_prob, obj_prob)
        box, cls = x_cat.split((self.reg_channels, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        cls_prob = cls.sigmoid()
        return (dbox, cls_prob)

    def bias_init(self) -> None:
        for idx, (box_convolutions, class_convolutions, stride) in enumerate(
            zip(self.cv_boxes, self.cv_classes, self.stride, strict=True)
        ):
            if not isinstance(box_convolutions, nn.Sequential):
                raise TypeError
            box_last = box_convolutions[-1]
            if not isinstance(box_last, nn.Conv3d):
                raise TypeError
            box_bias = box_last.bias
            if box_bias is None:
                raise ValueError
            if self.bbox_parametrization == "center_extents":
                box_bias.data[:3] = 0.0
                box_bias.data[3:6] = 4.0
            else:
                box_bias.data[:] = 2.0
            if not self.separate_objectness_head:
                if not isinstance(class_convolutions, nn.Sequential):
                    raise TypeError
                class_last = class_convolutions[-1]
                if not isinstance(class_last, nn.Conv3d):
                    raise TypeError
                class_bias = class_last.bias
                if class_bias is None:
                    raise ValueError
                class_bias.data[: self.nc] = math.log(5 / self.nc / (256 / stride) ** 3)
            else:
                obj_seq = self.cv_obj[idx]
                if not isinstance(obj_seq, nn.Sequential):
                    raise TypeError
                obj_last = obj_seq[-1]
                if not isinstance(obj_last, nn.Conv3d):
                    raise TypeError
                obj_bias = obj_last.bias
                if obj_bias is None:
                    raise ValueError
                obj_bias.data[:] = math.log(5 / (256 / stride) ** 3)

    def decode_bboxes(
        self, bboxes: torch.Tensor, anchors: torch.Tensor, xyzwhd: bool = True
    ) -> torch.Tensor:
        return decode_bboxes(
            bboxes,
            anchors,
            bbox_parametrization=self.bbox_parametrization,
            xyzwhd=xyzwhd,
            dim=1,
            center_offset_limit=self.center_offset_limit,
        )


class PrototypeMasks(nn.Module):
    """PrototypeMasks module for segmentation models."""

    def __init__(
        self,
        input_channels: int,
        intermediate_channels: int = 256,
        prototypes: int = 32,
    ) -> None:
        """Initialize the PrototypeMasks module with specified number of prototypes.

        Args:
            input_channels (int): Input channels.
            intermediate_channels (int): Intermediate channels.
            prototypes (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(input_channels, intermediate_channels, kernel_size=3)
        self.upsample = nn.ConvTranspose3d(
            intermediate_channels, intermediate_channels, 2, 2, 0, bias=True
        )
        self.cv2 = Conv(intermediate_channels, intermediate_channels, kernel_size=3)
        self.cv3 = Conv(intermediate_channels, prototypes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class PrototypeMasksHighResolution(nn.Module):
    """PrototypeMasks module for segmentation models."""

    def __init__(
        self,
        input_channels: int,
        input_channels_skip_connection_low: int,
        input_channels_skip_connection_high: int,
        prototypes: int = 32,
    ) -> None:
        """Initialize the PrototypeMasks module with specified number of prototypes.

        Args:
            input_channels (int): Input channels.
            input_channels_skip_connection_low (int): Input channels of low resolution skip connection,
            input_channels_skip_connection_high (int): Input channels of high resolution skip connection,
            prototypes (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(input_channels, prototypes * 4, kernel_size=3)
        self.upsample1 = nn.ConvTranspose3d(prototypes * 4, prototypes * 4, 2, 2, 0, bias=True)

        self.cv_skip_low_1 = Conv(input_channels_skip_connection_low, prototypes * 4, kernel_size=3)
        self.cv_skip_low_2 = Conv(prototypes * 4, prototypes * 4, kernel_size=3)

        self.cv2 = Conv(prototypes * 4, prototypes * 2, kernel_size=3)
        self.cv3 = Conv(prototypes * 2, prototypes * 2, kernel_size=3)
        self.upsample2 = nn.ConvTranspose3d(prototypes * 2, prototypes * 2, 2, 2, 0, bias=True)

        self.cv_skip_high_1 = Conv(
            input_channels_skip_connection_high, prototypes * 2, kernel_size=3
        )
        self.cv_skip_high_2 = Conv(prototypes * 2, prototypes * 2, kernel_size=3)

        self.cv_out_1 = Conv(prototypes * 2, prototypes * 2, kernel_size=3)
        self.cv_out_2 = Conv(prototypes * 2, prototypes)

    def forward(
        self,
        x: torch.Tensor,
        skip_connection_low_res: torch.Tensor,
        skip_connection_high_res: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        x = self.cv1(x)
        x = self.upsample1(x)

        skip_low = self.cv_skip_low_1(skip_connection_low_res)
        skip_low = self.cv_skip_low_2(skip_low)
        x = x + skip_low

        x = self.cv2(x)
        x = self.cv3(x)
        x = self.upsample2(x)

        skip_high = self.cv_skip_high_1(skip_connection_high_res)
        skip_high = self.cv_skip_high_2(skip_high)

        out = self.cv_out_1(x + skip_high)
        return self.cv_out_2(out)


class PrototypeMasksHighResolutionOneSkip(nn.Module):
    """PrototypeMasks module for segmentation models."""

    def __init__(
        self,
        input_channels: int,
        input_channels_skip_connection: int,
        prototypes: int = 32,
    ) -> None:
        """Initialize the PrototypeMasks module with specified number of prototypes.

        Args:
            input_channels (int): Input channels.
            input_channels_skip_connection (int): Input channels of skip connection,
            prototypes (int): Output channels (number of protos).
        """
        super().__init__()

        self.cv1 = Conv(input_channels, prototypes * 4, kernel_size=3)
        self.cv2 = Conv(prototypes * 4, prototypes * 4, kernel_size=3)
        self.upsample2 = nn.ConvTranspose3d(prototypes * 4, prototypes * 2, 2, 2, 0, bias=True)

        self.cv_skip_1 = Conv(input_channels_skip_connection, prototypes * 2, kernel_size=3)
        self.cv_skip_2 = Conv(prototypes * 2, prototypes * 2, kernel_size=3)

        self.cv_out_1 = Conv(prototypes * 2, prototypes * 2, kernel_size=3)
        self.cv_out_2 = Conv(prototypes * 2, prototypes)

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        x = self.cv1(x)
        x = self.cv2(x)
        x = self.upsample2(x)

        skip_high = self.cv_skip_1(skip_connection)
        skip_high = self.cv_skip_2(skip_high)

        out = self.cv_out_1(x + skip_high)
        return self.cv_out_2(out)
