from __future__ import annotations

from pathlib import Path

import pytest
import torch

from radyolo.validation.plotting import plot_validation_case


@pytest.mark.filterwarnings(
    "ignore:constrained_layout not applied because axes sizes collapsed to zero.*:UserWarning"
)
def test_plot_validation_case_saves_with_out_of_bounds_boxes(tmp_path: Path) -> None:
    image = torch.zeros((1, 16, 16, 16), dtype=torch.float32)
    reference_boxes = torch.tensor([[2, 2, 2, 10, 10, 10]], dtype=torch.float32)
    predicted_boxes = torch.tensor(
        [[-1_000_000.0, -1_000_000.0, -1_000_000.0, 1_000_000.0, 1_000_000.0, 1_000_000.0]],
        dtype=torch.float32,
    )
    output_path = tmp_path / "validation_case.jpg"

    plot_validation_case(
        image=image,
        reference_boxes=reference_boxes,
        reference_classes=torch.tensor([0], dtype=torch.int64),
        predicted_boxes=predicted_boxes,
        predicted_classes=torch.tensor([0], dtype=torch.int64),
        prediction_confs=torch.tensor([0.95], dtype=torch.float32),
        fname=output_path,
        names={0: "lesion"},
        case_name="case-001",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
