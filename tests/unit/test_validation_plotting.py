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


def test_plot_results_marks_best_fitness_epoch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results_csv = tmp_path / "results.csv"
    columns = ["epoch", *plotting.RESULTS_PLOT_COLUMNS]
    pd.DataFrame(
        [
            [0, 1.0, 0.9, 0.8, 0.7, 0.6, 1.1, 1.0, 0.9, 0.8, 0.7, 0.2, 0.1, 0.15, 0.12, 0.05],
            [1, 0.9, 0.8, 0.7, 0.6, 0.5, 1.0, 0.9, 0.8, 0.7, 0.6, 0.6, 0.2, 0.25, 0.22, 0.15],
            [2, 0.8, 0.7, 0.6, 0.5, 0.4, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.35, 0.32, 0.25],
        ],
        columns=columns,
    ).to_csv(results_csv, index=False)

    captured_figures: list[Figure] = []
    original_savefig = Figure.savefig

    def capture_savefig(self: Figure, fname: str | Path, *args: object, **kwargs: object) -> None:
        captured_figures.append(self)
        original_savefig(self, fname, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_savefig)

    plotting.plot_results(results_csv)

    loss_curves_figure = captured_figures[0]
    assert loss_curves_figure.axes[0].get_legend() is None

    fitness_axis = loss_curves_figure.axes[10]
    marker_lines = [
        line
        for line in fitness_axis.lines
        if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]
    ]

    assert marker_lines
    assert marker_lines[0].get_xdata()[0] == pytest.approx(1.0)
    legend = fitness_axis.get_legend()
    assert legend is not None
    assert "Best epoch" in [text.get_text() for text in legend.get_texts()]


def test_palette_overflow_generates_distinct_followup_variant() -> None:
    base_color = plotting.palette_rgb(0)
    overflow_color = plotting.palette_rgb(len(plotting.BASE_PLOT_HEX_COLORS))

    assert base_color == plotting.Colors.hex2rgb(plotting.MAIN_CURVE_COLOR)
    assert overflow_color != base_color
    assert all(0 <= channel <= 255 for channel in overflow_color)


def test_plot_results_uses_shared_palette_for_series(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results_csv = tmp_path / "results.csv"
    columns = ["epoch", "train/box_loss", "metrics/fitness"]
    pd.DataFrame(
        [
            [0, 1.0, 0.2],
            [1, 0.9, 0.4],
        ],
        columns=columns,
    ).to_csv(results_csv, index=False)

    captured_figures: list[Figure] = []
    original_savefig = Figure.savefig

    def capture_savefig(self: Figure, fname: str | Path, *args: object, **kwargs: object) -> None:
        captured_figures.append(self)
        original_savefig(self, fname, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_savefig)

    plotting.plot_results(results_csv)

    loss_curves_figure = captured_figures[0]
    expected_color = plotting.palette_mpl(0)
    assert loss_curves_figure.axes[0].lines[0].get_color() == expected_color
