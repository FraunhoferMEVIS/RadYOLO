from __future__ import annotations

import math
from typing import TYPE_CHECKING, assert_never, cast, overload

import torch
from torch import nn

from yolo11_3d.ultralytics.utils.tal import dist2bbox, make_anchors, make_anchors_3d

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
        kernel_size: int | tuple[int, int] = 1,
        stride: int = 1,
        padding: int | str | None = None,
        n_groups: int = 1,
        dilation: int = 1,
        activation: bool | nn.Module = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            autopad(kernel_size, padding, dilation),
            groups=n_groups,
            dilation=dilation,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
        self.activation = (
            self.default_activation
            if activation is True
            else activation
            if isinstance(activation, nn.Module)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.batch_norm(self.conv(x)))

    def forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Conv_3D(nn.Module):
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

    def forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DWConv(Conv):
    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 1,
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


class DWConv_3D(Conv_3D):
    def __init__(  # noqa: PLR0913
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

    def __init__(self, in_channels: int = 16) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(in_channels, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, in_channels, 1, 1))
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.in_channels, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class DFL_3D(nn.Module):
    """Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, in_channels: int = 16) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(in_channels, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, in_channels, 1, 1))
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape
        return self.conv(x.view(b, 6, self.in_channels, a).transpose(2, 1).softmax(1)).view(b, 6, a)


class Bottleneck(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        n_groups: int = 1,
        kernel_sizes: tuple[int | tuple[int, int], int | tuple[int, int]] = (3, 3),
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        c_ = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, c_, kernel_sizes[0], 1)
        self.conv2 = Conv(c_, out_channels, kernel_sizes[1], 1, n_groups=n_groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class Bottleneck_3D(nn.Module):
    def __init__(  # noqa: PLR0913
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
        self.conv1 = Conv_3D(in_channels, c_, kernel_sizes[0], 1)
        self.conv2 = Conv_3D(c_, out_channels, kernel_sizes[1], 1, n_groups=n_groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class C2f(nn.Module):
    def __init__(  # noqa: PLR0913
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
                kernel_sizes=((3, 3), (3, 3)),
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


class C2f_3D(nn.Module):
    def __init__(  # noqa: PLR0913
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
        self.in_conv = Conv_3D(in_channels, 2 * self.hidden_channels, 1, 1)
        self.out_conv = Conv_3D((2 + n_bottlenecks) * self.hidden_channels, out_channels, 1)
        self.m = nn.ModuleList(
            Bottleneck_3D(
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
    def __init__(  # noqa: PLR0913
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


class C3k2_3D(C2f_3D):
    def __init__(  # noqa: PLR0913
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
            C3k_3D(self.hidden_channels, self.hidden_channels, 2, shortcut, n_groups)
            if c3k
            else Bottleneck_3D(self.hidden_channels, self.hidden_channels, shortcut, n_groups)
            for _ in range(n_bottlenecks)
        )


class C3(nn.Module):
    def __init__(  # noqa: PLR0913
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
                    kernel_sizes=((1, 1), (3, 3)),
                    expansion=1.0,
                )
                for _ in range(n_bottlenecks)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_conv(torch.cat((self.hidden_layer(self.conv1(x)), self.conv2(x)), 1))


class C3_3D(nn.Module):
    def __init__(  # noqa: PLR0913
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
        self.conv1 = Conv_3D(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv_3D(in_channels, hidden_channels, 1, 1)
        self.out_conv = Conv_3D(2 * hidden_channels, out_channels, 1)
        self.hidden_layer = nn.Sequential(
            *(
                Bottleneck_3D(
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
    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        n_bottlenecks: int = 1,
        shortcut: bool = True,
        n_groups: int = 1,
        expansion: float = 0.5,
        kernel_size: int | tuple[int, int] = 3,
    ) -> None:
        super().__init__(in_channels, out_channels, n_bottlenecks, shortcut, n_groups, expansion)
        hidden_channels = int(out_channels * 3)
        self.m = nn.Sequential(
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


class C3k_3D(C3_3D):
    def __init__(  # noqa: PLR0913
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
        hidden_channels = int(out_channels * 3)
        self.m = nn.Sequential(
            *(
                Bottleneck_3D(
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
        self.hidden_layer = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        concat_layers = [self.input_conv(x)]
        for _ in range(self.n_concat_layers - 1):
            concat_layers.append(self.hidden_layer(concat_layers[-1]))
        return self.output_conv(torch.cat(concat_layers, 1))


class SPPF_3D(nn.Module):
    """Spatial Pyramid Pooling Fast."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        hidden_channels = in_channels // 2
        self.n_concat_layers = 4
        self.input_conv = Conv_3D(in_channels, hidden_channels, 1, 1)
        self.output_conv = Conv_3D(hidden_channels * self.n_concat_layers, out_channels, 1, 1)
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
        batch, channel, height, width = x.shape
        n_pixels = height * width
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
            batch, channel, height, width
        ) + self.positional_encoding(value.reshape(batch, channel, height, width))
        return self.projection(x)


class Attention_3D(nn.Module):
    def __init__(self, in_dim: int, num_heads: int = 8, attention_ratio: float = 0.5) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.key_dim = int(self.head_dim * attention_ratio)
        self.scale: float = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = in_dim + nh_kd * 2
        self.query_key_value = Conv_3D(in_dim, h, 1, activation=False)
        self.projection = Conv_3D(in_dim, in_dim, 1, activation=False)
        self.positional_encoding = Conv_3D(in_dim, in_dim, 3, 1, n_groups=in_dim, activation=False)

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


class PSABlock_3D(nn.Module):
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

        self.attention = Attention_3D(
            channels, attention_ratio=attention_ratio, num_heads=num_heads
        )
        self.feed_forward_network = nn.Sequential(
            Conv_3D(channels, channels * 2, 1), Conv_3D(channels * 2, channels, 1, activation=False)
        )
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x) if self.add else self.attention(x)
        x = x + self.feed_forward_network(x) if self.add else self.feed_forward_network(x)
        return x  # noqa: RET504


class C2PSA(nn.Module):
    """Cross Stage Partial Self-Attention.

    This module implements a convolutional block with a configurable number of Partial Self-Attention (PSA) blocks.

    Attributes:
        hidden_channels (int): Number of hidden channels.
        in_conv (Conv): The initial 1x1 convolution layer to adjust the number of input channels to 2*`hidden_channels`.
        out_conv (Conv): The final 1x1 convolution layer to adjust the number of output channels to `hidden_channels`.
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
            raise ValueError  # TODO: add error message
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


class C2PSA_3D(nn.Module):
    """Cross Stage Partial Self-Attention.

    This module implements a convolutional block with a configurable number of Partial Self-Attention (PSA) blocks.

    Attributes:
        hidden_channels (int): Number of hidden channels.
        in_conv (Conv): The initial 1x1 convolution layer to adjust the number of input channels to 2*`hidden_channels`.
        out_conv (Conv): The final 1x1 convolution layer to adjust the number of output channels to `hidden_channels`.
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
            raise ValueError  # TODO: add error message
        self.hidden_channels = int(in_channels * expansion)
        self.in_conv = Conv_3D(in_channels, 2 * self.hidden_channels, 1, 1)
        self.out_conv = Conv_3D(2 * self.hidden_channels, in_channels, 1)

        self.m = nn.Sequential(
            *(
                PSABlock_3D(
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

    def __init__(self, num_classes: int = 80, ch: Sequence[int] = ()) -> None:
        super().__init__()
        self.nc = num_classes  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = num_classes + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = (
            max((16, ch[0] // 4, self.reg_max * 4)),
            max(ch[0], min(self.nc, 100)),
        )  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * self.reg_max, 1),
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(
        self, x: list[torch.Tensor]
    ) -> list[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def _inference(self, x: list[torch.Tensor]) -> torch.Tensor:
        shape = x[0].shape
        if self.shape != shape:
            self.anchors, self.strides = (
                x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self) -> None:
        for a, b, s in zip(self.cv2, self.cv3, self.stride, strict=True):
            if not isinstance(a, nn.Sequential):
                raise TypeError
            if not isinstance(b, nn.Sequential):
                raise TypeError

            a_last = a[-1]
            if not isinstance(a_last, nn.Conv2d):
                raise TypeError

            b_last = b[-1]
            if not isinstance(b_last, nn.Conv2d):
                raise TypeError

            a_bias = a_last.bias
            if a_bias is None:
                raise ValueError
            a_bias.data[:] = 1.0

            b_bias = b_last.bias
            if b_bias is None:
                raise ValueError
            b_bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)

    def decode_bboxes(
        self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True
    ) -> torch.Tensor:
        return dist2bbox(bboxes, anchors, xywh=xywh, dim=1)


class Detect_3D(nn.Module):
    export = False
    max_det = 300
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, num_classes: int = 80, ch: Sequence[int] = ()) -> None:
        super().__init__()
        self.nc = num_classes  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = num_classes + self.reg_max * 6  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = (
            max((16, ch[0] // 4, self.reg_max * 6)),
            max(ch[0], min(self.nc, 100)),
        )  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv_3D(x, c2, 3),
                Conv_3D(c2, c2, 3),
                nn.Conv3d(c2, 6 * self.reg_max, 1),
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv_3D(x, x, 3), Conv_3D(x, c3, 1)),
                nn.Sequential(DWConv_3D(c3, c3, 3), Conv_3D(c3, c3, 1)),
                nn.Conv3d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL_3D(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(
        self, x: list[torch.Tensor]
    ) -> list[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def _inference(self, x: list[torch.Tensor]) -> torch.Tensor:
        shape = x[0].shape
        if self.shape != shape:
            self.anchors, self.strides = (
                x.transpose(0, 1) for x in make_anchors_3d(x, self.stride, 0.5)
            )
            self.shape = shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        box, cls = x_cat.split((self.reg_max * 6, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self) -> None:
        for a, b, s in zip(self.cv2, self.cv3, self.stride, strict=True):
            if not isinstance(a, nn.Sequential):
                raise TypeError
            if not isinstance(b, nn.Sequential):
                raise TypeError

            a_last = a[-1]
            if not isinstance(a_last, nn.Conv3d):
                raise TypeError

            b_last = b[-1]
            if not isinstance(b_last, nn.Conv3d):
                raise TypeError

            a_bias = a_last.bias
            if a_bias is None:
                raise ValueError
            a_bias.data[:] = 1.0

            b_bias = b_last.bias
            if b_bias is None:
                raise ValueError
            # TODO(muellergroh): check for validity in 3D
            b_bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)

    def decode_bboxes(
        self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True
    ) -> torch.Tensor:
        return dist2bbox(bboxes, anchors, xywh=xywh, dim=1)
