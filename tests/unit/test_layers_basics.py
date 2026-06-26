from __future__ import annotations

import pytest
import torch
from torch import nn

from radyolo.models.layers import (
    C2PSA,
    C3,
    SPPF,
    Attention,
    Bottleneck,
    C2f,
    C3k,
    C3k2,
    Concat,
    Conv,
    ConvTransposed,
    Detect,
    DWConv,
    PrototypeMasksFullResolution,
    PrototypeMasksFullResolutionV2,
    PrototypeMasksFullResolutionV3,
    PrototypeMasksHalfResolution,
    PrototypeMasksHalfResolutionV2,
    PrototypeMasksHighResolution,
    PSABlock,
    autopad,
    build_detection_head_layers,
    resolve_head_hidden_channels,
)


def test_autopad_supports_int_tuple_and_dilation() -> None:
    assert autopad(3) == 1
    assert autopad((3, 5, 7)) == (1, 2, 3)
    assert autopad(3, dilation=2) == 2
    assert autopad((3, 3, 3), dilation=3) == (3, 3, 3)
    assert autopad(3, padding="same") == "same"


def test_conv_variants_keep_expected_shapes() -> None:
    x = torch.randn(1, 4, 8, 8, 8)

    conv = Conv(4, 6, kernel_size=3)
    y = conv(x)
    assert y.shape == (1, 6, 8, 8, 8)

    conv_identity = Conv(4, 4, kernel_size=1, activation=False)
    y_identity = conv_identity(x)
    assert y_identity.shape == (1, 4, 8, 8, 8)
    assert isinstance(conv_identity.activation, nn.Identity)

    conv_relu = Conv(4, 4, kernel_size=1, activation=nn.ReLU())
    y_relu = conv_relu(x)
    assert y_relu.shape == (1, 4, 8, 8, 8)


def test_conv_transposed_and_dwconv_shapes_and_groups() -> None:
    x = torch.randn(1, 6, 4, 4, 4)

    up = ConvTransposed(6, 4, kernel_size=2, stride=2)
    y_up = up(x)
    assert y_up.shape == (1, 4, 8, 8, 8)

    dw = DWConv(6, 9, kernel_size=3)
    y_dw = dw(x)
    assert y_dw.shape == (1, 9, 4, 4, 4)
    assert dw.conv.groups == 3


def test_bottleneck_with_and_without_shortcut() -> None:
    x_same = torch.randn(1, 8, 6, 6, 6)
    with_shortcut = Bottleneck(8, 8, shortcut=True)
    out_shortcut = with_shortcut(x_same)
    assert out_shortcut.shape == x_same.shape

    x_diff = torch.randn(1, 8, 6, 6, 6)
    no_shortcut = Bottleneck(8, 12, shortcut=True)
    out_no_shortcut = no_shortcut(x_diff)
    assert out_no_shortcut.shape == (1, 12, 6, 6, 6)


def test_csp_blocks_forward_shapes() -> None:
    x = torch.randn(1, 16, 6, 6, 6)

    c2f = C2f(16, 16, n_bottlenecks=2, shortcut=True)
    out_c2f = c2f(x)
    out_c2f_split = c2f.forward_split(x)
    assert out_c2f.shape == x.shape
    assert out_c2f_split.shape == x.shape

    c3 = C3(16, 16, n_bottlenecks=2, shortcut=True)
    out_c3 = c3(x)
    assert out_c3.shape == x.shape

    c3k = C3k(16, 16, n_bottlenecks=1, kernel_size=(3, 3, 3))
    out_c3k = c3k(x)
    assert out_c3k.shape == x.shape

    c3k2_plain = C3k2(16, 16, n_bottlenecks=1, c3k=False)
    out_c3k2_plain = c3k2_plain(x)
    assert out_c3k2_plain.shape == x.shape

    c3k2_nested = C3k2(16, 16, n_bottlenecks=1, c3k=True)
    out_c3k2_nested = c3k2_nested(x)
    assert out_c3k2_nested.shape == x.shape


def test_sppf_attention_psa_and_concat_forward_shapes() -> None:
    x = torch.randn(1, 64, 6, 6, 6)

    sppf = SPPF(64, 64)
    out_sppf = sppf(x)
    assert out_sppf.shape == x.shape

    attention = Attention(64, num_heads=4, attention_ratio=0.5)
    out_attention = attention(x)
    assert out_attention.shape == x.shape

    psa_shortcut = PSABlock(64, num_heads=4, shortcut=True)
    out_psa_shortcut = psa_shortcut(x)
    assert out_psa_shortcut.shape == x.shape

    psa_no_shortcut = PSABlock(64, num_heads=4, shortcut=False)
    out_psa_no_shortcut = psa_no_shortcut(x)
    assert out_psa_no_shortcut.shape == x.shape

    concat = Concat(dimension=1)
    merged = concat([x[:, :16], x[:, 16:32]])
    assert merged.shape == (1, 32, 6, 6, 6)


def test_c2psa_validates_channels_and_runs_forward() -> None:
    with pytest.raises(ValueError, match="Input channels"):
        C2PSA(64, 32)

    x = torch.randn(1, 128, 4, 4, 4)
    c2psa = C2PSA(128, 128, n_hidden_layers=1)
    out = c2psa(x)
    assert out.shape == x.shape


def test_head_builder_and_hidden_channel_resolution() -> None:
    regular = build_detection_head_layers(16, 8, 6, "regular")
    separable = build_detection_head_layers(16, 8, 6, "separable")

    x = torch.randn(1, 16, 5, 5, 5)
    out_regular = regular(x)
    out_separable = separable(x)
    assert out_regular.shape == (1, 6, 5, 5, 5)
    assert out_separable.shape == (1, 6, 5, 5, 5)

    assert resolve_head_hidden_channels(16, 0.5, "head") == 8
    with pytest.raises(ValueError, match="invalid hidden width"):
        resolve_head_hidden_channels(16, 0.0, "head")


def test_detect_forward_training_and_inference_paths() -> None:
    detect_train = Detect(num_classes=3, ch=(16, 32), dfl_reg_max=1, separate_objectness_head=False)
    detect_train.train()
    train_inputs = [
        torch.randn(1, 16, 4, 4, 4),
        torch.randn(1, 32, 2, 2, 2),
    ]
    train_output = detect_train(train_inputs)
    assert train_output.raw_predictions is not None
    assert len(train_output.raw_predictions) == 2

    detect_eval = Detect(num_classes=3, ch=(16, 32), dfl_reg_max=1, separate_objectness_head=True)
    detect_eval.eval()
    infer_inputs = [
        torch.randn(1, 16, 4, 4, 4),
        torch.randn(1, 32, 2, 2, 2),
    ]
    infer_output = detect_eval(infer_inputs)

    assert infer_output.decoded_boxes is not None
    assert infer_output.class_probabilities is not None
    assert infer_output.objectness_probabilities is not None
    assert infer_output.decoded_boxes.shape[1] == 6
    assert infer_output.class_probabilities.shape[1] == 3
    assert infer_output.objectness_probabilities.shape[1] == 1


def test_prototype_mask_decoders_shapes() -> None:
    x_full = torch.randn(1, 4, 16, 16, 16)
    x_half = torch.randn(1, 8, 8, 8, 8)
    x_quarter = torch.randn(1, 12, 4, 4, 4)
    x_eight = torch.randn(1, 16, 2, 2, 2)
    x_sixteenth = torch.randn(1, 20, 1, 1, 1)

    full_v2 = PrototypeMasksFullResolutionV2(4, 8, 12, 16, 20, prototypes=2)
    out_full_v2 = full_v2(x_full, x_half, x_quarter, x_eight, x_sixteenth)
    assert out_full_v2.shape == (1, 2, 16, 16, 16)

    full = PrototypeMasksFullResolution(4, 8, 12, 16, 20, prototypes=2)
    out_full = full(x_full, x_half, x_quarter, x_eight, x_sixteenth)
    assert out_full.shape == (1, 2, 16, 16, 16)

    full_v3 = PrototypeMasksFullResolutionV3(4, 8, 12, 16, 20, prototypes=2)
    out_full_v3 = full_v3(x_full, x_half, x_quarter, x_eight, x_sixteenth)
    assert out_full_v3.shape == (1, 2, 16, 16, 16)

    half = PrototypeMasksHalfResolution(8, 12, 16, 20, prototypes=2)
    out_half = half(x_half, x_quarter, x_eight, x_sixteenth)
    assert out_half.shape == (1, 2, 8, 8, 8)

    half_v2 = PrototypeMasksHalfResolutionV2(8, 12, 16, 20, prototypes=2)
    out_half_v2 = half_v2(x_half, x_quarter, x_eight, x_sixteenth)
    assert out_half_v2.shape == (1, 2, 8, 8, 8)

    high = PrototypeMasksHighResolution(
        channels_half_resolution=8,
        channels_quater_resolution=12,
        channels_eigth_resolution=16,
        prototypes=4,
    )
    out_high = high(
        input_half_resolution=x_half,
        input_quater_resolution=x_quarter,
        input_eigth_resolution=x_eight,
    )
    assert out_high.shape == (1, 4, 8, 8, 8)
