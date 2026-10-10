from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest
import torch

from radyolo.config.config import get_training_config
from radyolo.config.spatial import PatchSize
from radyolo.data.augmentations import resample_to_shape, sample_downsize_scales
from radyolo.data.data import apply_configured_augmentations_to_batch, pad_batch_to_common_shape
from radyolo.data.preprocessing import (
    INPUT_SIZE_DIVISOR,
    fit_shape_within,
    pad_to_multiple,
    preprocess_image,
    resolve_target_shape,
    round_up_to_multiple,
)
from radyolo.models.model import HardcodedModel
from radyolo.training.tal import make_anchors

if TYPE_CHECKING:
    from pathlib import Path

    from radyolo.config.config import DefaultConfig


def _build_model(*, separate_objectness_head: bool = False) -> HardcodedModel:
    model = HardcodedModel(
        input_channels=1,
        num_classes=2,
        scale="n",
        dfl_reg_max=1,
        bbox_parametrization="center_extents",
        bbox_center_offset_limit=4.0,
        enable_output_masks=True,
        prototype_layer="half_resolution_v2",
        separate_objectness_head=separate_objectness_head,
    )
    return model.eval()


def test_size_divisor_matches_coarsest_stride() -> None:
    model = _build_model()

    assert model.size_divisor == INPUT_SIZE_DIVISOR
    assert model.size_divisor == int(model.stride.max().item())


@pytest.mark.parametrize("separate_objectness_head", [False, True])
def test_model_accepts_different_input_sizes(separate_objectness_head: bool) -> None:
    model = _build_model(separate_objectness_head=separate_objectness_head)

    for shape in [(32, 32, 32), (64, 32, 96), (32, 96, 64)]:
        with torch.no_grad():
            output = model(torch.zeros(2, 1, *shape))

        expected_anchors = sum(
            (shape[0] // s) * (shape[1] // s) * (shape[2] // s) for s in (8, 16, 32)
        )
        assert output.decoded_boxes.shape == (2, 6, expected_anchors)
        assert output.class_probabilities.shape == (2, 2, expected_anchors)
        assert output.mask_coefficients.shape[-1] == expected_anchors
        assert output.prototype_masks.shape[-3:] == tuple(dim // 2 for dim in shape)


def test_decoded_boxes_follow_the_input_grid() -> None:
    """Anchors are rebuilt for each input size, so boxes stay inside the matching volume."""
    model = _build_model()
    with torch.no_grad():
        small = model(torch.zeros(1, 1, 32, 32, 32)).decoded_boxes
        large = model(torch.zeros(1, 1, 32, 32, 96)).decoded_boxes
        small_again = model(torch.zeros(1, 1, 32, 32, 32)).decoded_boxes

    # The first stride-8 anchors (lowest z, y, x) are shared, so their predictions agree
    # up to the receptive-field differences at the border; just check the grid extents.
    assert large[0, 0].max() > small[0, 0].max()  # x centers extend further for the wider input
    assert torch.equal(small, small_again)


def test_model_rejects_unsupported_input_sizes() -> None:
    model = _build_model()

    with pytest.raises(ValueError, match="multiple of 32"):
        model(torch.zeros(1, 1, 32, 48, 32))


def test_make_anchors_accepts_tensor_and_float_strides() -> None:
    feats = [torch.zeros(1, 4, 2, 3, 4), torch.zeros(1, 4, 1, 2, 2)]

    anchors_tensor, strides_tensor = make_anchors(feats, torch.tensor([8.0, 16.0]), 0.5)
    anchors_floats, strides_floats = make_anchors(feats, (8.0, 16.0), 0.5)

    assert anchors_tensor.shape == (2 * 3 * 4 + 1 * 2 * 2, 3)
    assert torch.equal(anchors_tensor, anchors_floats)
    assert torch.equal(strides_tensor, strides_floats)
    assert strides_tensor[:24].eq(8.0).all()
    assert strides_tensor[24:].eq(16.0).all()
    # x runs fastest, then y, then z
    assert anchors_tensor[0].tolist() == [0.5, 0.5, 0.5]
    assert anchors_tensor[1].tolist() == [1.5, 0.5, 0.5]
    assert anchors_tensor[4].tolist() == [0.5, 1.5, 0.5]
    assert anchors_tensor[12].tolist() == [0.5, 0.5, 1.5]


def test_round_up_and_pad_to_multiple() -> None:
    assert round_up_to_multiple((1, 32, 33, 64)) == (32, 32, 64, 64)

    tensor = torch.ones(2, 5, 40, 33)
    padded = pad_to_multiple(tensor.unsqueeze(0), value=-1.0)

    assert padded.shape == (1, 2, 32, 64, 64)
    assert padded[0, :, :5, :40, :33].eq(1.0).all()
    assert padded.sum() == tensor.sum() - (padded.numel() - tensor.numel())


def test_pad_to_multiple_rejects_shrinking() -> None:
    with pytest.raises(ValueError, match="smaller"):
        pad_to_multiple(torch.zeros(1, 40, 40, 40), target_shape=(32, 64, 64))


def test_fit_shape_within_only_downscales_oversized_shapes() -> None:
    assert fit_shape_within((20, 30, 40), (64, 64, 64)) == (20, 30, 40)
    # aspect ratio is kept: the x axis (200 -> 100) determines the scale
    assert fit_shape_within((50, 100, 200), (100, 100, 100)) == (25, 50, 100)


def test_resolve_target_shape_with_max_shape() -> None:
    source = (50, 100, 200)
    max_shape = PatchSize(x=100, y=100, z=100)

    assert resolve_target_shape(source, max_shape=max_shape) == (25, 50, 100)
    assert resolve_target_shape((10, 20, 30), max_shape=max_shape) == (10, 20, 30)
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_target_shape(source, target_shape=max_shape, max_shape=max_shape)


def test_preprocess_image_keeps_native_shape_within_max_shape() -> None:
    image = torch.rand(1, 20, 30, 40)

    kept, source_shape, resized_shape = preprocess_image(
        image, max_shape=PatchSize(x=64, y=64, z=64), apply_normalization=False
    )
    shrunk, _, shrunk_shape = preprocess_image(
        image, max_shape=PatchSize(x=20, y=20, z=20), apply_normalization=False
    )

    assert source_shape == resized_shape == (20, 30, 40)
    assert kept.shape == (1, 20, 30, 40)
    assert shrunk_shape == (10, 15, 20)
    assert shrunk.shape == (1, 10, 15, 20)


def test_pad_batch_to_common_shape_preserves_box_voxel_coordinates() -> None:
    small = torch.zeros(1, 32, 32, 32)
    large = torch.zeros(1, 40, 64, 96)
    small_masks = torch.ones(1, 32, 32, 32)
    # normalized xyzwhd boxes: centered box covering half of every axis
    batch = {
        "img": (small, large),
        "cls": torch.zeros(2, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.25, 0.5, 0.75, 0.1, 0.2, 0.3]]),
        "batch_idx": torch.tensor([0, 1]),
        "masks": (small_masks, None),
        "ratio_pad": (None, None),
        "shape": ((32, 32, 32), (40, 64, 96)),
    }

    padded = pad_batch_to_common_shape(batch)

    assert padded["img"].shape == (2, 1, 64, 64, 96)
    assert padded["masks"][0].shape == (1, 64, 64, 96)
    assert padded["masks"][0][:, :32, :32, :32].eq(1).all()
    assert padded["masks"][0].sum() == small_masks.sum()
    assert padded["masks"][1] is None

    common_xyz = torch.tensor([96.0, 64.0, 64.0])
    # sample 0: center at voxel 16 on each axis, size 16 voxels
    first = padded["bboxes"][0]
    assert torch.allclose(first[:3] * common_xyz, torch.tensor([16.0, 16.0, 16.0]))
    assert torch.allclose(first[3:] * common_xyz, torch.tensor([16.0, 16.0, 16.0]))
    # sample 1 (x=96, y=64, z=40): voxel centers/sizes are unchanged by padding
    second = padded["bboxes"][1]
    assert torch.allclose(second[:3] * common_xyz, torch.tensor([24.0, 32.0, 30.0]))
    assert torch.allclose(second[3:] * common_xyz, torch.tensor([9.6, 12.8, 12.0]))


def test_pad_batch_to_common_shape_is_noop_for_stacked_aligned_batch() -> None:
    images = torch.zeros(2, 1, 32, 64, 32)
    bboxes = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
    batch = {
        "img": images,
        "bboxes": bboxes,
        "batch_idx": torch.tensor([0]),
        "masks": (None, None),
    }

    padded = pad_batch_to_common_shape(batch)

    assert padded["img"] is images
    assert torch.equal(padded["bboxes"], bboxes)


def test_variable_input_size_requires_wholeimage_training(minimal_train_config: Path) -> None:
    with pytest.raises(ValueError, match="only supported with training_mode='wholeimage'"):
        get_training_config(
            minimal_train_config,
            {"variable_input_size": True, "training_mode": "patchwise"},
        )

    config = get_training_config(minimal_train_config, {"variable_input_size": True})

    assert config.variable_input_size is True
    assert config.training_mode == "wholeimage"


def test_sample_downsize_scales_picks_valid_sizes() -> None:
    base = (96, 64, 160)
    seen_shapes = set()
    for seed in range(200):
        random.seed(seed)
        scales = sample_downsize_scales(base, (40, 32, 100), INPUT_SIZE_DIVISOR, p=1.0)
        if scales is None:
            continue
        shape = tuple(round(b * s) for b, s in zip(base, scales, strict=True))
        seen_shapes.add(shape)
        assert all(new % INPUT_SIZE_DIVISOR == 0 for new in shape)
        assert all(new <= b for new, b in zip(shape, base, strict=True))
        # minimum (40, 32, 100) rounded up to multiples of 32
        assert all(new >= m for new, m in zip(shape, (64, 32, 128), strict=True))
    assert len(seen_shapes) > 5


def test_sample_downsize_scales_probability_and_noop() -> None:
    assert sample_downsize_scales((64, 64, 64), (32, 32, 32), INPUT_SIZE_DIVISOR, p=0.0) is None
    # minimum equals the base size: nothing to sample
    assert sample_downsize_scales((64, 64, 64), (64, 64, 64), INPUT_SIZE_DIVISOR, p=1.0) is None
    # a minimum larger than the base is clipped to the base
    assert sample_downsize_scales((32, 32, 32), (500, 500, 500), INPUT_SIZE_DIVISOR, p=1.0) is None


def test_resample_to_shape_keeps_content_and_mask_values() -> None:
    image = torch.arange(64.0).reshape(1, 4, 4, 4).repeat(1, 8, 8, 8)  # (1, 32, 32, 32)
    masks = torch.zeros(2, 32, 32, 32, dtype=torch.uint8)
    masks[0, :16] = 1
    masks[1, 16:] = 1

    small_image, small_masks = resample_to_shape(image, masks, (16, 8, 32))

    assert small_image.shape == (1, 16, 8, 32)
    assert small_masks.shape == (2, 16, 8, 32)
    assert small_masks.dtype == torch.uint8
    assert set(small_masks.unique().tolist()) == {0, 1}
    assert small_masks[0, :8].all()
    assert not small_masks[0, 8:].any()
    assert resample_to_shape(image, None, (16, 16, 16))[1] is None


def _downsize_config(minimal_train_config: Path, **overrides: object) -> DefaultConfig:
    return get_training_config(
        minimal_train_config,
        {
            "prob_mirroring": 0.0,
            "prob_zoom": 0.0,
            "prob_rotation": 0.0,
            "prob_cutout": 0.0,
            "do_batchgenerators_intensity_augmentation": False,
            **overrides,
        },
    )


def test_batch_downsize_resamples_images_masks_and_keeps_labels(minimal_train_config: Path) -> None:
    config = _downsize_config(
        minimal_train_config,
        prob_downsize=1.0,
        downsize_min_patch_size={"x": 32, "y": 32, "z": 64},
        patch_size={"x": 32, "y": 96, "z": 64},
    )
    bboxes = torch.tensor([[0.5, 0.5, 0.5, 0.25, 0.25, 0.25], [0.2, 0.3, 0.4, 0.1, 0.1, 0.1]])
    shapes_seen = set()
    for seed in range(20):
        random.seed(seed)
        batch = {
            "img": torch.rand(2, 1, 64, 96, 32),
            "cls": torch.zeros(2, 1),
            "bboxes": bboxes.clone(),
            "batch_idx": torch.tensor([0, 1]),
            "masks": (torch.ones(1, 64, 96, 32), None),
            "ratio_pad": (None, None),
            "shape": ((64, 96, 32), (64, 96, 32)),
        }

        out = apply_configured_augmentations_to_batch(batch, config, config.patch_size, "gpu")

        new_shape = tuple(out["img"].shape[2:])
        shapes_seen.add(new_shape)
        assert out["img"].shape[:2] == (2, 1)
        assert all(dim % INPUT_SIZE_DIVISOR == 0 for dim in new_shape)
        assert out["masks"][0].shape == (1, *new_shape)
        assert out["shape"] == (new_shape, new_shape)
        assert torch.allclose(out["bboxes"], bboxes, atol=1e-6)  # normalized labels are unchanged
    assert len(shapes_seen) > 1


def test_batch_downsize_disabled_leaves_batch_untouched(minimal_train_config: Path) -> None:
    config = _downsize_config(minimal_train_config, patch_size={"x": 64, "y": 64, "z": 64})
    assert config.prob_downsize == 0.0
    batch = {
        "img": torch.rand(1, 1, 64, 64, 64),
        "cls": torch.zeros(1, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.2, 0.2, 0.2]]),
        "batch_idx": torch.tensor([0]),
        "masks": (None,),
        "ratio_pad": (None,),
        "shape": ((64, 64, 64),),
    }

    out = apply_configured_augmentations_to_batch(batch, config, config.patch_size, "gpu")

    assert out["img"].shape == (1, 1, 64, 64, 64)


def test_downsize_rejects_too_small_minimum_size(minimal_train_config: Path) -> None:
    with pytest.raises(ValueError, match="at least 64"):
        _downsize_config(
            minimal_train_config,
            prob_downsize=0.5,
            downsize_min_patch_size={"x": 32, "y": 32, "z": 32},
        )
    config = _downsize_config(
        minimal_train_config,
        prob_downsize=0.5,
        downsize_min_patch_size={"x": 32, "y": 32, "z": 64},
    )
    assert config.downsize_min_patch_size.get_xyz() == (32, 32, 64)


def test_downsize_requires_gpu_stage_flag(minimal_train_config: Path) -> None:
    config = _downsize_config(
        minimal_train_config,
        prob_downsize=0.5,
        zoom_augmentation_device="cpu",
        rotation_augmentation_device="cpu",
        cutout_augmentation_device="cpu",
        mirroring_augmentation_device="cpu",
        intensity_noise_augmentation_device="cpu",
        intensity_blur_augmentation_device="cpu",
        intensity_mult_brightness_augmentation_device="cpu",
        intensity_add_brightness_augmentation_device="cpu",
        intensity_contrast_augmentation_device="cpu",
        intensity_lowres_augmentation_device="cpu",
        intensity_gamma_augmentation_device="cpu",
        gin_augmentation_device="cpu",
        channel_dropout_augmentation_device="cpu",
    )

    assert config.uses_gpu_augmentations()  # the batch stage must run so downsize gets applied
