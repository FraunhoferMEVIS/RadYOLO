from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from matplotlib.figure import Figure

from radyolo.validation import plotting
from radyolo.validation.plotting import plot_validation_case


def test_plotting_uses_non_interactive_backend() -> None:
    assert "agg" in plotting.plt.get_backend().lower()


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


def test_plot_results_warns_when_results_png_is_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    results_csv = tmp_path / "results.csv"
    columns = [
        "epoch",
        "train/box_loss",
        "train/cls_obj_loss",
        "train/dfl_loss",
        "train/mask_loss",
        "train/cls_sep_loss",
        "val/box_loss",
        "val/cls_obj_loss",
        "val/dfl_loss",
        "val/mask_loss",
        "val/cls_sep_loss",
        "metrics/precision",
        "metrics/mAP10Agn",
        "metrics/mAP10",
        "metrics/mAP10-95",
        "metrics/mAP10-95Agn",
        "metrics/dice",
        "metrics/fitness",
    ]
    pd.DataFrame([[0, *range(1, 18)], [1, *range(2, 19)]], columns=columns).to_csv(
        results_csv,
        index=False,
    )

    busy_target = tmp_path / "loss_curves.png"

    original_savefig = Figure.savefig

    def non_interactive_subplots(*args: object, **kwargs: object) -> tuple[Figure, object]:
        fig = Figure(figsize=kwargs.get("figsize"), tight_layout=kwargs.get("tight_layout", False))
        axes = fig.subplots(
            *args,
            sharex=kwargs.get("sharex", False),
            sharey=kwargs.get("sharey", False),
            squeeze=kwargs.get("squeeze", True),
        )
        return fig, axes

    def savefig_with_busy(self: Figure, fname: str | Path, *args: object, **kwargs: object) -> None:
        if Path(fname) == busy_target:
            msg = "Device or resource busy"
            raise OSError(msg)
        original_savefig(self, fname, *args, **kwargs)

    monkeypatch.setattr(plotting.plt, "subplots", non_interactive_subplots)
    monkeypatch.setattr(Figure, "savefig", savefig_with_busy)

    caplog.set_level("WARNING", logger=plotting.LOGGER.name)
    plotting.plot_results(results_csv)

    assert not busy_target.exists()
    assert "Could not save plot" in caplog.text


def test_plot_results_saves_image_label_curves_separately(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    columns = [
        "epoch",
        "train/box_loss",
        "val/box_loss",
        "metrics/mAP10-95",
        "metrics/dice",
        "train/img_cce",
        "train/img_cst",
        "train/img_cox",
        "val/img_cce",
        "val/img_cst",
        "val/img_cox",
        "metrics/image_label_balanced_accuracy",
        "metrics/image_label_multiclass_roc_auc",
        "metrics/image_label_c_index",
    ]
    pd.DataFrame(
        [
            [0, 1.0, 1.1, 0.2, 0.3, 0.8, 0.7, 0.6, 0.75, 0.65, 0.55, 0.4, 0.5, 0.6],
            [1, 0.9, 1.0, 0.25, 0.35, 0.7, 0.6, 0.5, 0.65, 0.55, 0.45, 0.45, 0.55, 0.65],
        ],
        columns=columns,
    ).to_csv(results_csv, index=False)

    plotting.plot_results(results_csv)

    assert (tmp_path / "loss_curves.png").exists()
    assert (tmp_path / "loss_curves_image_labels.png").exists()
