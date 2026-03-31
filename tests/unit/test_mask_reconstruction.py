from __future__ import annotations

import torch

from radyolo.utils import ops


def test_reconstruct_mask_probabilities_promotes_to_float32() -> None:
    coefficients = torch.tensor([1000.0, -1000.0], dtype=torch.float16)
    prototypes = torch.tensor(
        [
            [[[0.001, -0.001], [0.5, -0.5]]],
            [[[-0.001, 0.001], [-0.5, 0.5]]],
        ],
        dtype=torch.float16,
    )

    probabilities = ops.reconstruct_mask_probabilities(coefficients, prototypes)

    assert probabilities.dtype == torch.float32
    assert probabilities.shape == (1, 2, 2)
    assert torch.isfinite(probabilities).all()
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))
