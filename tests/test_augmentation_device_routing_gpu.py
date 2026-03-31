from __future__ import annotations

import pytest
import torch

from radyolo.data import augmentations as augmentation_module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gaussian_noise_and_blur_preserve_tensor_device_on_gpu() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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