from __future__ import annotations

import pytest
import torch

from radyolo.data.normalization import normalize_image


def test_normalize_image_none_leaves_image_unchanged() -> None:
    image = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]]]],
        dtype=torch.float32,
    )

    normalized = normalize_image(image.clone(), {"type": "none"})

    assert torch.equal(normalized, image)


def test_normalize_image_zscore_normalizes_global_mean_and_std() -> None:
    image = torch.arange(1, 9, dtype=torch.float32).reshape(1, 2, 2, 2)

    normalized = normalize_image(image.clone(), {"type": "zscore"})

    assert normalized.mean().item() == pytest.approx(0.0, abs=1e-6)
    assert normalized.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-6)


def test_normalize_image_255_to_01_scales_and_clamps() -> None:
    image = torch.tensor(
        [[[[0.0, 127.5], [255.0, 300.0]]]],
        dtype=torch.float32,
    )

    normalized = normalize_image(image.clone(), {"type": "255_to_01"})

    expected = torch.tensor(
        [[[[0.0, 0.5], [1.0, 1.0]]]],
        dtype=torch.float32,
    )
    assert torch.allclose(normalized, expected)


def test_normalize_image_ct_clips_and_standardizes() -> None:
    image = torch.tensor(
        [[[[-2000.0, -1024.0], [100.0, 4000.0]]]],
        dtype=torch.float32,
    )

    normalized = normalize_image(
        image.clone(),
        {
            "type": "ct",
            "clip_min": -1024.0,
            "clip_max": 3071.0,
            "mean": 100.0,
            "std": 200.0,
        },
    )

    expected = torch.tensor(
        [
            [
                [
                    [(-1024.0 - 100.0) / 200.0, (-1024.0 - 100.0) / 200.0],
                    [0.0, (3071.0 - 100.0) / 200.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(normalized, expected)


def test_normalize_image_applies_per_channel_configs() -> None:
    image = torch.tensor(
        [
            [[[0.0, 255.0], [127.5, 300.0]]],
            [[[1.0, 2.0], [3.0, 4.0]]],
        ],
        dtype=torch.float32,
    )

    normalized = normalize_image(
        image.clone(),
        [
            {"type": "255_to_01"},
            {"type": "zscore"},
        ],
    )

    expected_first_channel = torch.tensor(
        [[[0.0, 1.0], [0.5, 1.0]]],
        dtype=torch.float32,
    )
    expected_second_channel = (image[1] - image[1].mean()) / image[1].std(unbiased=False)

    assert torch.allclose(normalized[0], expected_first_channel)
    assert torch.allclose(normalized[1], expected_second_channel)


def test_normalize_image_rejects_per_channel_length_mismatch() -> None:
    image = torch.zeros((2, 1, 2, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="does not match number of channels"):
        normalize_image(image, [{"type": "none"}])


def test_normalize_image_rejects_invalid_per_channel_entry() -> None:
    image = torch.zeros((1, 1, 2, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="expected dict with 'type'"):
        normalize_image(image, [{}])  # type: ignore[list-item]
