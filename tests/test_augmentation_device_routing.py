from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from radyolo.config.config import DefaultConfig
from radyolo.data import augmentations as augmentation_module
from radyolo.data import data as data_module

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch


def _make_config(tmp_path: Path, **overrides: Any) -> DefaultConfig:
    base_config: dict[str, Any] = {
        "input_channels": 1,
        "num_classes": 1,
        "model_scale": "n",
        "data": tmp_path,
        "fold": 0,
        "batch_size": 1,
        "imgsz": [8, 8, 8],
        "name": "test",
    }
    base_config.update(overrides)
    return DefaultConfig(**base_config)


def _record_image_only_call(
    call_order: list[str],
    name: str,
) -> Callable[[torch.Tensor, Any], torch.Tensor]:
    def _recorder(image: torch.Tensor, *_: Any, **__: Any) -> torch.Tensor:
        call_order.append(name)
        return image

    return _recorder


def _make_batch(image: torch.Tensor) -> data_module.Batch:
    return {
        "img": image.unsqueeze(0),
        "ori_shape": ((4, 4, 4),),
        "ratio_pad": (None,),
        "shape": ((4, 4, 4),),
        "cls": torch.empty((0, 1), dtype=torch.float32),
        "bboxes": torch.empty((0, 6), dtype=torch.float32),
        "batch_idx": torch.empty((0,), dtype=torch.int64),
        "masks": (None,),
        "case_names": ("test-case",),
    }


def test_intensity_augmentations_follow_configured_order_per_device(
    monkeypatch: MonkeyPatch,
) -> None:
    call_order: list[str] = []
    monkeypatch.setattr(
        augmentation_module,
        "gaussian_noise_augmentation",
        _record_image_only_call(call_order, "noise"),
    )
    monkeypatch.setattr(
        augmentation_module,
        "gaussian_blur_augmentation",
        _record_image_only_call(call_order, "blur"),
    )
    monkeypatch.setattr(
        augmentation_module,
        "multiplicative_brightness_augmentation",
        _record_image_only_call(call_order, "mult_brightness"),
    )
    monkeypatch.setattr(
        augmentation_module,
        "additive_brightness_augmentation",
        _record_image_only_call(call_order, "add_brightness"),
    )
    monkeypatch.setattr(
        augmentation_module,
        "contrast_augmentation",
        _record_image_only_call(call_order, "contrast"),
    )
    monkeypatch.setattr(
        augmentation_module,
        "simulate_low_resolution_augmentation",
        _record_image_only_call(call_order, "lowres"),
    )
    monkeypatch.setattr(
        augmentation_module,
        "gamma_augmentation",
        _record_image_only_call(call_order, "gamma"),
    )

    image = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
    augmentation_module.batch_generators_intensity_augmentations(
        image,
        do_intensity_augmentations_per_channel=False,
        noise_prob=1.0,
        noise_augmentation_device="cpu",
        blur_prob=1.0,
        blur_augmentation_device="gpu",
        mult_brightness_prob=1.0,
        mult_brightness_augmentation_device="cpu",
        add_brightness_prob=1.0,
        add_brightness_augmentation_device="gpu",
        contrast_prob=1.0,
        contrast_augmentation_device="cpu",
        lowres_prob=1.0,
        lowres_augmentation_device="gpu",
        gamma_prob=1.0,
        gamma_augmentation_device="cpu",
        placement="cpu",
    )

    assert call_order == ["noise", "mult_brightness", "contrast", "gamma"]


def test_apply_configured_image_augmentations_preserves_order_on_same_device(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    config = _make_config(
        tmp_path,
        gin_augmentation_device="cpu",
        channel_dropout_augmentation_device="cpu",
        zoom_augmentation_device="gpu",
        rotation_augmentation_device="gpu",
        cutout_augmentation_device="gpu",
        mirroring_augmentation_device="gpu",
    )
    call_order: list[str] = []

    monkeypatch.setattr(
        data_module,
        "batch_generators_intensity_augmentations",
        _record_image_only_call(call_order, "intensity"),
    )
    monkeypatch.setattr(
        data_module,
        "gin_augmentation",
        _record_image_only_call(call_order, "gin"),
    )
    monkeypatch.setattr(
        data_module,
        "random_channel_dropout",
        _record_image_only_call(call_order, "dropout"),
    )

    image = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
    batch = _make_batch(image)

    data_module.apply_configured_augmentations_to_batch(
        batch,
        config,
        (4, 4, 4),
        placement="cpu",
    )

    assert call_order == ["intensity", "gin", "dropout"]


def test_apply_configured_augmentations_runs_spatial_before_image_augmentations(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    config = _make_config(
        tmp_path,
        gin_augmentation_device="cpu",
        channel_dropout_augmentation_device="cpu",
        zoom_augmentation_device="cpu",
        rotation_augmentation_device="gpu",
        cutout_augmentation_device="gpu",
        mirroring_augmentation_device="gpu",
    )
    call_order: list[str] = []

    monkeypatch.setattr(
        data_module,
        "batch_generators_intensity_augmentations",
        _record_image_only_call(call_order, "intensity"),
    )
    monkeypatch.setattr(
        data_module,
        "gin_augmentation",
        _record_image_only_call(call_order, "gin"),
    )
    monkeypatch.setattr(
        data_module,
        "random_channel_dropout",
        _record_image_only_call(call_order, "dropout"),
    )

    def _record_spatial(
        image: torch.Tensor,
        labels: torch.Tensor,
        masks: torch.Tensor | None,
        *_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        call_order.append("zoom")
        return image, labels, masks

    monkeypatch.setattr(
        data_module,
        "random_zoom",
        _record_spatial,
    )

    image = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
    batch = _make_batch(image)

    data_module.apply_configured_augmentations_to_batch(
        batch,
        config,
        (4, 4, 4),
        placement="cpu",
    )

    assert call_order == ["zoom", "intensity", "gin", "dropout"]


def test_apply_configured_augmentations_routes_only_matching_device(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    config = _make_config(
        tmp_path,
        intensity_noise_augmentation_device="gpu",
        intensity_blur_augmentation_device="gpu",
        intensity_mult_brightness_augmentation_device="gpu",
        intensity_add_brightness_augmentation_device="gpu",
        intensity_contrast_augmentation_device="gpu",
        intensity_lowres_augmentation_device="gpu",
        intensity_gamma_augmentation_device="gpu",
        gin_augmentation_device="cpu",
        channel_dropout_augmentation_device="gpu",
        zoom_augmentation_device="cpu",
        rotation_augmentation_device="gpu",
        cutout_augmentation_device="gpu",
        mirroring_augmentation_device="gpu",
    )
    call_order: list[str] = []

    monkeypatch.setattr(
        data_module,
        "batch_generators_intensity_augmentations",
        _record_image_only_call(call_order, "intensity"),
    )
    monkeypatch.setattr(
        data_module,
        "gin_augmentation",
        _record_image_only_call(call_order, "gin"),
    )
    monkeypatch.setattr(
        data_module,
        "random_channel_dropout",
        _record_image_only_call(call_order, "dropout"),
    )

    def _record_spatial(
        image: torch.Tensor,
        labels: torch.Tensor,
        masks: torch.Tensor | None,
        *_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        call_order.append("zoom")
        return image, labels, masks

    monkeypatch.setattr(
        data_module,
        "random_zoom",
        _record_spatial,
    )

    image = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
    batch = _make_batch(image)

    data_module.apply_configured_augmentations_to_batch(
        batch,
        config,
        (4, 4, 4),
        placement="cpu",
    )

    assert call_order == ["zoom", "gin"]


def test_gaussian_noise_and_blur_preserve_tensor_device_on_cpu() -> None:
    device = torch.device("cpu")
    image = torch.ones((2, 8, 8, 8), device=device, dtype=torch.float32)

    noised = augmentation_module.gaussian_noise_augmentation(
        image.clone(),
        variance_range=(0.1, 0.1),
        p=1.0,
        synchronize_channels=False,
    )
    blurred = augmentation_module.gaussian_blur_augmentation(
        image.clone(),
        sigma_range=(0.75, 0.75),
        p=1.0,
        synchronize_channels=False,
    )

    assert noised.device == image.device
    assert blurred.device == image.device
    assert noised.shape == image.shape
    assert blurred.shape == image.shape
    assert torch.isfinite(noised).all()
    assert torch.isfinite(blurred).all()
