from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from radyolo.config.config import get_cfg
from radyolo.models.layers import Conv, DWConv
from radyolo.models.model import build_model_from_config


def _head_hidden_channels(head: nn.Sequential) -> int:
    first_stage = head[0]
    if isinstance(first_stage, Conv):
        return first_stage.conv.out_channels
    if isinstance(first_stage, nn.Sequential):
        projection = first_stage[1]
        if isinstance(projection, Conv):
            return projection.conv.out_channels
    msg = f"Unexpected head layout: {type(first_stage)!r}"
    raise TypeError(msg)


def _head_input_channels(head: nn.Sequential) -> int:
    first_stage = head[0]
    if isinstance(first_stage, Conv):
        return first_stage.conv.in_channels
    if isinstance(first_stage, nn.Sequential):
        depthwise = first_stage[0]
        if isinstance(depthwise, DWConv):
            return depthwise.conv.in_channels
    msg = f"Unexpected head layout: {type(first_stage)!r}"
    raise TypeError(msg)


def _assert_regular_head(head: nn.Sequential) -> None:
    assert isinstance(head[0], Conv)
    assert isinstance(head[1], Conv)
    assert isinstance(head[2], nn.Conv3d)


def _assert_separable_head(head: nn.Sequential) -> None:
    assert isinstance(head[0], nn.Sequential)
    assert isinstance(head[0][0], DWConv)
    assert isinstance(head[0][1], Conv)
    assert isinstance(head[1], nn.Sequential)
    assert isinstance(head[1][0], DWConv)
    assert isinstance(head[1][1], Conv)
    assert isinstance(head[2], nn.Conv3d)


def test_default_head_configuration_matches_existing_layout(
    minimal_train_config: Path,
) -> None:
    config = get_cfg(minimal_train_config, {"separate_objectness_head": True})

    assert config.classification_head_convolutions == "separable"
    assert config.objectness_head_convolutions == "separable"
    assert config.box_head_convolutions == "regular"
    assert config.segmentation_head_convolutions == "regular"
    assert config.classification_head_channel_factor == 1.0
    assert config.objectness_head_channel_factor == 1.0
    assert config.box_head_channel_factor == 0.25
    assert config.segmentation_head_channel_factor == 0.25

    model = build_model_from_config(config)

    _assert_regular_head(model.l23.cv_boxes[0])
    _assert_separable_head(model.l23.cv_classes[0])
    _assert_separable_head(model.l23.cv_obj[0])
    _assert_regular_head(model.mask_coefficient_layer[0])

    first_detect_channels = _head_input_channels(model.l23.cv_boxes[0])
    assert _head_hidden_channels(model.l23.cv_boxes[0]) == int(first_detect_channels * 0.25)
    assert _head_hidden_channels(model.l23.cv_classes[0]) == int(first_detect_channels * 1.0)
    assert _head_hidden_channels(model.l23.cv_obj[0]) == int(first_detect_channels * 1.0)
    assert _head_hidden_channels(model.mask_coefficient_layer[0]) == int(first_detect_channels * 0.25)


def test_custom_head_configuration_is_applied(minimal_train_config: Path) -> None:
    config = get_cfg(
        minimal_train_config,
        {
            "separate_objectness_head": True,
            "classification_head_convolutions": "regular",
            "objectness_head_convolutions": "regular",
            "box_head_convolutions": "separable",
            "segmentation_head_convolutions": "separable",
            "classification_head_channel_factor": 0.5,
            "objectness_head_channel_factor": 0.75,
            "box_head_channel_factor": 0.5,
            "segmentation_head_channel_factor": 0.5,
        },
    )

    model = build_model_from_config(config)

    _assert_separable_head(model.l23.cv_boxes[0])
    _assert_regular_head(model.l23.cv_classes[0])
    _assert_regular_head(model.l23.cv_obj[0])
    _assert_separable_head(model.mask_coefficient_layer[0])

    first_detect_channels = _head_input_channels(model.l23.cv_boxes[0])
    assert _head_hidden_channels(model.l23.cv_boxes[0]) == int(first_detect_channels * 0.5)
    assert _head_hidden_channels(model.l23.cv_classes[0]) == int(first_detect_channels * 0.5)
    assert _head_hidden_channels(model.l23.cv_obj[0]) == int(first_detect_channels * 0.75)
    assert _head_hidden_channels(model.mask_coefficient_layer[0]) == int(first_detect_channels * 0.5)