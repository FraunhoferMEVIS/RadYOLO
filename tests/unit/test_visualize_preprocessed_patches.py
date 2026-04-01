from __future__ import annotations

import torch

from scripts.visualize_preprocessed_patches import _expand_masks_to_instances, _select_slice_indices


def test_expand_masks_to_instances_splits_label_map() -> None:
    mask = torch.tensor(
        [
            [
                [0, 1, 1],
                [0, 0, 0],
                [2, 2, 0],
            ],
            [
                [0, 1, 0],
                [0, 0, 0],
                [2, 2, 0],
            ],
        ],
        dtype=torch.uint8,
    )

    expanded = _expand_masks_to_instances(mask)

    assert expanded is not None
    assert expanded.shape == (2, 2, 3, 3)
    assert torch.equal(expanded[0], (mask == 1).to(torch.uint8))
    assert torch.equal(expanded[1], (mask == 2).to(torch.uint8))


def test_select_slice_indices_prefers_annotated_slices() -> None:
    boxes = torch.tensor(
        [
            [1.0, 1.0, 4.0, 5.0, 5.0, 8.0],
            [0.0, 0.0, 6.0, 3.0, 3.0, 7.0],
        ],
        dtype=torch.float32,
    )
    masks = torch.zeros((1, 10, 8, 8), dtype=torch.uint8)
    masks[0, 6:8, 2:6, 2:6] = 1

    indices = _select_slice_indices("xy", 10, boxes, masks, count=3)

    assert len(indices) == 3
    assert 6 in indices or 7 in indices
    assert all(0 <= index < 10 for index in indices)