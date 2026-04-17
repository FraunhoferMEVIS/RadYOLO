from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch

from radyolo.data import augmentations as augmentation_module

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


def test_tensor_cutout_removes_heavily_occluded_labels(monkeypatch: MonkeyPatch) -> None:
    image = torch.linspace(0.0, 1.0, steps=64, dtype=torch.float32).reshape(1, 4, 4, 4)
    labels = torch.tensor([[0.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0]], dtype=torch.float32)
    masks = torch.ones((1, 4, 4, 4), dtype=torch.uint8)
    randint_values = iter([2, 2, 2, 2, 2, 2])

    monkeypatch.setattr(augmentation_module.random, "random", lambda: 0.0)
    monkeypatch.setattr(
        augmentation_module.random,
        "randint",
        lambda a, b: next(randint_values),
    )
    monkeypatch.setattr(augmentation_module.random, "uniform", lambda a, b: 0.25)

    image_cut, labels_cut, masks_cut = augmentation_module.tensor_cutout(
        image.clone(),
        labels.clone(),
        masks.clone(),
        cutout_params=[(0.5, 1)],
        p=1.0,
    )

    assert torch.all(image_cut[:, 1:3, 1:3, 1:3] == 0.25)
    assert labels_cut.shape == (0, 7)
    assert masks_cut is not None
    assert masks_cut.shape == (0, 4, 4, 4)


def test_random_zoom_zoom_out_with_constant_padding_updates_boxes(monkeypatch: MonkeyPatch) -> None:
    image = torch.ones((1, 4, 4, 4), dtype=torch.float32)
    labels = torch.tensor([[0.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0]], dtype=torch.float32)
    randint_values = iter([1, 1, 1])

    monkeypatch.setattr(augmentation_module.random, "random", lambda: 0.0)
    monkeypatch.setattr(augmentation_module.random, "uniform", lambda a, b: 0.5)
    monkeypatch.setattr(augmentation_module.random, "randint", lambda a, b: next(randint_values))

    zoomed_image, zoomed_labels, zoomed_masks = augmentation_module.random_zoom(
        image.clone(),
        labels.clone(),
        masks=None,
        min_zoom=0.5,
        max_zoom=0.5,
        p=1.0,
        out_zoom_padding="constant",
        padding_fill_value=-1.0,
        uniform_zoom=True,
    )

    assert zoomed_image.shape == image.shape
    assert zoomed_masks is None
    assert torch.equal(zoomed_image[:, 0], torch.full((1, 4, 4), -1.0))
    assert torch.allclose(
        zoomed_labels,
        torch.tensor([[0.0, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5]], dtype=torch.float32),
    )


def test_random_rotation_zero_angles_preserves_image_boxes_and_masks(
    monkeypatch: MonkeyPatch,
) -> None:
    image = torch.arange(64, dtype=torch.float32).reshape(1, 4, 4, 4)
    labels = torch.tensor([[0.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0]], dtype=torch.float32)
    masks = torch.zeros((1, 4, 4, 4), dtype=torch.uint8)
    masks[:, 1:3, 1:3, 1:3] = 1

    monkeypatch.setattr(augmentation_module.random, "random", lambda: 0.0)
    monkeypatch.setattr(augmentation_module.random, "uniform", lambda a, b: 0.0)

    rotated_image, rotated_labels, rotated_masks = augmentation_module.random_rotation(
        image.clone(),
        labels.clone(),
        masks.clone(),
        angle_ranges=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        p=1.0,
        keep_box_volume=True,
    )

    assert torch.allclose(rotated_image, image)
    assert torch.allclose(rotated_labels, labels)
    assert rotated_masks is not None
    assert torch.equal(rotated_masks, masks)


def test_random_mirroring_flips_selected_axes(monkeypatch: MonkeyPatch) -> None:
    image = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    labels = torch.tensor([[0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]], dtype=torch.float32)
    masks = torch.arange(8, dtype=torch.uint8).reshape(1, 2, 2, 2)
    random_values = iter([0.0, 0.0, 1.0, 0.0])

    monkeypatch.setattr(augmentation_module.random, "random", lambda: next(random_values))

    mirrored_image, mirrored_labels, mirrored_masks = augmentation_module.random_mirroring(
        image.clone(),
        labels.clone(),
        masks.clone(),
        mirror_axes=("x", "y", "z"),
        p=1.0,
    )

    expected_image = torch.flip(image, dims=[3, 1])
    assert torch.equal(mirrored_image, expected_image)
    assert torch.allclose(
        mirrored_labels,
        torch.tensor([[0.0, 0.8, 0.3, 0.6, 0.5, 0.6, 0.7]], dtype=torch.float32),
    )
    assert mirrored_masks is not None
    assert torch.equal(mirrored_masks, torch.flip(masks, dims=[3, 1]))


def test_gaussian_noise_augmentation_synchronizes_noise_across_channels(
    monkeypatch: MonkeyPatch,
) -> None:
    image = torch.zeros((2, 2, 2, 2), dtype=torch.float32)

    monkeypatch.setattr(augmentation_module.random, "random", lambda: 0.0)
    monkeypatch.setattr(augmentation_module, "_sample_scalar", lambda value_range: 0.5)
    monkeypatch.setattr(
        augmentation_module.torch,
        "randn",
        lambda shape, device=None, dtype=None: torch.ones(shape, device=device, dtype=dtype),
    )

    noised = augmentation_module.gaussian_noise_augmentation(
        image.clone(),
        variance_range=(0.5, 0.5),
        p=1.0,
        synchronize_channels=True,
    )

    assert torch.equal(noised[0], noised[1])
    assert torch.all(noised == 0.5)


def test_gaussian_blur_augmentation_blurs_each_channel_independently(
    monkeypatch: MonkeyPatch,
) -> None:
    image = torch.zeros((2, 5, 5, 5), dtype=torch.float32)
    image[0, 2, 2, 2] = 1.0
    image[1, 1, 1, 1] = 1.0

    monkeypatch.setattr(augmentation_module.random, "random", lambda: 0.0)
    monkeypatch.setattr(augmentation_module, "_sample_scalar", lambda value_range: 0.75)

    blurred = augmentation_module.gaussian_blur_augmentation(
        image.clone(),
        sigma_range=(0.75, 0.75),
        p=1.0,
        synchronize_channels=False,
    )

    assert blurred.shape == image.shape
    assert blurred[0, 2, 2, 2] < 1.0
    assert blurred[0, 2, 2, 2] > 0.0
    assert blurred[1, 1, 1, 1] < 1.0
    assert blurred[1, 1, 1, 1] > 0.0
    assert not torch.equal(blurred[0], blurred[1])


@pytest.mark.parametrize(
    ("fn_name", "kwargs", "expected_transform"),
    [
        (
            "multiplicative_brightness_augmentation",
            {"multiplier_range": (0.75, 1.25), "p": 0.2, "synchronize_channels": True},
            "MultiplicativeBrightnessTransform",
        ),
        (
            "contrast_augmentation",
            {"contrast_range": (0.8, 1.2), "p": 0.4, "synchronize_channels": False},
            "ContrastTransform",
        ),
        (
            "simulate_low_resolution_augmentation",
            {"scale_range": (0.5, 1.0), "p": 0.5, "synchronize_channels": True},
            "SimulateLowResolutionTransform",
        ),
        (
            "gamma_augmentation",
            {
                "gamma_range": (0.7, 1.5),
                "p": 0.6,
                "synchronize_channels": False,
                "p_invert_image": 0.25,
                "p_retain_stats": 1.0,
            },
            "GammaTransform",
        ),
    ],
)
def test_batchgenerators_wrappers_delegate_with_expected_transform(
    fn_name: str,
    kwargs: dict[str, Any],
    expected_transform: str,
    monkeypatch: MonkeyPatch,
) -> None:
    image = torch.zeros((1, 3, 3, 3), dtype=torch.float32)
    captured: dict[str, Any] = {}

    def _capture(image_in: torch.Tensor, transform: Any, apply_probability: float) -> torch.Tensor:
        captured["image"] = image_in
        captured["transform"] = transform.__class__.__name__
        captured["probability"] = apply_probability
        return image_in + 1.0

    monkeypatch.setattr(augmentation_module, "_apply_batchgenerators_transform", _capture)

    wrapped = getattr(augmentation_module, fn_name)(image.clone(), **kwargs)

    assert captured["transform"] == expected_transform
    assert captured["probability"] == kwargs["p"]
    assert torch.equal(captured["image"], image)
    assert torch.equal(wrapped, image + 1.0)


def test_additive_brightness_augmentation_synchronizes_shift_across_channels(
    monkeypatch: MonkeyPatch,
) -> None:
    image = torch.ones((2, 2, 2, 2), dtype=torch.float32)

    monkeypatch.setattr(augmentation_module.random, "random", lambda: 0.0)
    monkeypatch.setattr(
        augmentation_module.torch,
        "randn",
        lambda shape, device=None, dtype=None: torch.full(shape, 2.0, device=device, dtype=dtype),
    )

    transformed = augmentation_module.additive_brightness_augmentation(
        image,
        mu=1.0,
        sigma=0.5,
        p=1.0,
        synchronize_channels=True,
    )

    assert torch.equal(transformed, torch.full_like(image, 3.0))
    assert torch.equal(transformed[0], transformed[1])


def test_additive_brightness_augmentation_preserves_low_precision_dtype(
    monkeypatch: MonkeyPatch,
) -> None:
    image = torch.ones((2, 2, 2, 2), dtype=torch.bfloat16)

    monkeypatch.setattr(augmentation_module.random, "random", lambda: 0.0)
    monkeypatch.setattr(
        augmentation_module.torch,
        "randn",
        lambda shape, device=None, dtype=None: torch.ones(shape, device=device, dtype=dtype),
    )

    transformed = augmentation_module.additive_brightness_augmentation(
        image,
        mu=0.5,
        sigma=0.5,
        p=1.0,
        synchronize_channels=False,
    )

    assert transformed.dtype == torch.bfloat16
    assert torch.equal(transformed.float(), torch.full_like(image.float(), 2.0))


def test_ensure_non_zero_box_extents_clamps_small_extents() -> None:
    labels = torch.tensor(
        [[0.0, 0.5, 0.5, 0.5, 0.0, 0.2, 0.9], [1.0, 0.2, 0.2, 0.2, 2.0, 3.0, 4.0]],
        dtype=torch.float32,
    )

    adjusted = augmentation_module.ensure_non_zero_box_extents(labels.clone())

    assert torch.equal(
        adjusted[:, 4:],
        torch.tensor([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]], dtype=torch.float32),
    )


def test_random_channel_dropout_drops_single_channel_with_mean_fill(
    monkeypatch: MonkeyPatch,
) -> None:
    image = torch.tensor(
        [
            [[[1.0, 1.0], [1.0, 1.0]]],
            [[[2.0, 4.0], [6.0, 8.0]]],
        ],
        dtype=torch.float32,
    )

    monkeypatch.setattr(
        augmentation_module.torch,
        "bernoulli",
        lambda probs: torch.tensor([1.0, 1.0], dtype=probs.dtype),
    )
    monkeypatch.setattr(
        augmentation_module.torch,
        "multinomial",
        lambda probs, num_samples: torch.tensor([1], dtype=torch.int64),
    )

    dropped = augmentation_module.random_channel_dropout(
        image.clone(),
        drop_probabilities=(1.0, 1.0),
        fill_value=None,
    )

    assert torch.equal(dropped[0], image[0])
    assert torch.all(dropped[1] == 5.0)


def test_gin_augmentation_uses_gin_model_when_probability_hits(monkeypatch: MonkeyPatch) -> None:
    image = torch.ones((2, 3, 3, 3), dtype=torch.float32)

    class FakeGINNet:
        def __init__(
            self, hidden_layers: int, hidden_channels: int, p_kernel_size_1: float
        ) -> None:
            self.hidden_layers = hidden_layers
            self.hidden_channels = hidden_channels
            self.p_kernel_size_1 = p_kernel_size_1

        def to(self, device: torch.device, dtype: torch.dtype) -> FakeGINNet:
            return self

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            return input_tensor + 2.0

    monkeypatch.setattr(augmentation_module.random, "random", lambda: 0.0)
    monkeypatch.setattr(augmentation_module, "GINNet", FakeGINNet)

    augmented = augmentation_module.gin_augmentation(
        image.clone(),
        hidden_layers=3,
        hidden_channels=4,
        p_kernel_size_1=0.25,
        p=1.0,
    )

    assert torch.equal(augmented, image + 2.0)
