from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from radyolo.validation.plotting import (
    _annotate_projected_boxes,
    _concat_or_empty,
    _get_case_plot_volume,
    _label_text_color,
    _plot_results_grid,
    _project_boxes_for_plane,
    _project_case_volume,
    _softmax,
    confusion_matrix_cmap,
    named_plot_color,
    palette_mpl,
    palette_rgb,
    plot_results,
    plot_validation_case,
    save_figure_with_warning,
)


def test_save_figure_with_warning_logs_on_oserror(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fig = plt.figure()

    def _raise_oserror(*_args: object, **_kwargs: object) -> None:
        msg = "busy"
        raise OSError(msg)

    monkeypatch.setattr(fig, "savefig", _raise_oserror)

    save_figure_with_warning(fig, "out.png", dpi=120)

    assert "Could not save plot" in caplog.text
    plt.close(fig)


def test_palette_and_named_colors_and_confusion_colormap() -> None:
    base_color = palette_rgb(0)
    overflow_color = palette_rgb(30)

    assert isinstance(base_color, tuple)
    assert len(base_color) == 3
    assert isinstance(overflow_color, tuple)
    assert len(overflow_color) == 3
    assert all(0 <= channel <= 255 for channel in overflow_color)
    assert palette_mpl(1)[0] <= 1.0

    assert named_plot_color("mean")
    assert named_plot_color("background")
    assert named_plot_color("operating")
    assert named_plot_color("best")

    cmap = confusion_matrix_cmap()
    assert cmap.name == "validation_confusion_matrix"


def test_label_text_color_uses_luminance_thresholds() -> None:
    assert _label_text_color((255, 255, 255)) == (104, 31, 17)
    assert _label_text_color((0, 0, 0)) == (255, 255, 255)
    assert _label_text_color((120, 120, 120), (1, 2, 3)) == (1, 2, 3)


def test_get_case_plot_volume_variants_and_shape_errors(caplog: pytest.LogCaptureFixture) -> None:
    image = torch.arange(3 * 2 * 2 * 2, dtype=torch.float32).reshape(3, 2, 2, 2)

    default_volume = _get_case_plot_volume(image)
    assert torch.allclose(default_volume, image[1] - image[0])

    diff_volume = _get_case_plot_volume(
        image,
        image_mode="channel_difference",
        difference_channels=(2, 0),
    )
    assert torch.allclose(diff_volume, image[2] - image[0])

    fallback_volume = _get_case_plot_volume(
        image,
        image_mode="channel_difference",
        difference_channels=(4, 0),
    )
    assert torch.allclose(fallback_volume, image[1] - image[0])
    assert "Falling back to the default plot image" in caplog.text

    with pytest.raises(ValueError, match="Expected case image with shape"):
        _get_case_plot_volume(torch.zeros(2, 2, 2))


def test_project_case_volume_handles_planes_and_nonfinite_inputs() -> None:
    image = torch.tensor(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            [
                [[0.0, 1.0], [1.0, 0.0]],
                [[2.0, 0.0], [0.0, 2.0]],
            ],
        ],
        dtype=torch.float32,
    )

    xy = _project_case_volume(image, "xy", "max")
    xz = _project_case_volume(image, "xz", "mean")
    yz = _project_case_volume(image, "yz", "max")

    assert xy.shape == (2, 2)
    assert xz.shape == (2, 2)
    assert yz.shape == (2, 2)
    assert np.all((xy >= 0) & (xy <= 1))

    nonfinite = torch.full((1, 2, 2, 2), float("nan"), dtype=torch.float32)
    projected_nonfinite = _project_case_volume(nonfinite, "xy", "max")
    assert np.allclose(projected_nonfinite, 0.0)


def test_project_boxes_for_plane_handles_tensor_and_empty() -> None:
    boxes = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.float32)
    assert _project_boxes_for_plane(boxes, "xy").shape == (1, 4)
    assert _project_boxes_for_plane(boxes, "xz").shape == (1, 4)
    assert _project_boxes_for_plane(boxes, "yz").shape == (1, 4)

    empty = torch.zeros((0, 6), dtype=torch.float32)
    assert _project_boxes_for_plane(empty, "xy").shape == (0, 4)


def test_annotate_projected_boxes_filters_by_confidence_and_finite_values() -> None:
    fig, axis = plt.subplots(1, 1)
    boxes = np.array([[0, 0, 0, 4, 4, 4], [1, 1, 1, np.nan, 3, 3]], dtype=np.float32)
    classes = np.array([0, 1], dtype=np.int64)
    confs = np.array([0.9, 0.95], dtype=np.float32)

    _annotate_projected_boxes(
        axis,
        boxes,
        classes,
        "xy",
        names={0: "lesion", 1: "other"},
        confs=confs,
        conf_thres=0.5,
    )

    assert len(axis.patches) == 1
    plt.close(fig)


def test_plot_validation_case_calls_saver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    saved: dict[str, Path] = {}

    def _capture_save(figure: plt.Figure, output_path: Path | str, *, dpi: int) -> None:
        _ = figure
        _ = dpi
        saved["path"] = Path(output_path)

    monkeypatch.setattr("radyolo.validation.plotting.save_figure_with_warning", _capture_save)

    image = torch.rand((2, 6, 6, 6), dtype=torch.float32)
    boxes = torch.tensor([[1, 1, 1, 4, 4, 4]], dtype=torch.float32)
    classes = torch.tensor([0], dtype=torch.int64)

    out_file = tmp_path / "case_plot.jpg"
    plot_validation_case(
        image=image,
        reference_boxes=boxes,
        reference_classes=classes,
        predicted_boxes=boxes,
        predicted_classes=classes,
        prediction_confs=torch.tensor([0.8], dtype=torch.float32),
        fname=out_file,
        names={0: "lesion"},
        case_name="case_1",
        image_mode="channel_difference",
    )

    assert saved["path"] == out_file


def test_plot_results_grid_saves_for_existing_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "results.csv"
    pd.DataFrame(
        {
            "epoch": [0, 1, 2],
            "train/box_loss": [2.0, 1.5, 1.0],
            "metrics/fitness": [0.1, 0.3, 0.2],
        }
    ).to_csv(csv_path, index=False)

    saved: dict[str, Path] = {}

    def _capture_save(figure: plt.Figure, output_path: Path | str, *, dpi: int) -> None:
        _ = figure
        _ = dpi
        saved["path"] = Path(output_path)

    monkeypatch.setattr("radyolo.validation.plotting.save_figure_with_warning", _capture_save)

    _plot_results_grid(
        files=[csv_path],
        columns=("train/box_loss", "metrics/fitness", "missing/column"),
        save_path=tmp_path / "loss_curves.png",
        shape=(1, 3),
        figsize=(10, 3),
    )

    assert saved["path"].name == "loss_curves.png"


def test_plot_results_raises_when_no_results_csv(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match=r"No results\.csv files found"):
        plot_results(file=None, directory=tmp_path)


def test_concat_or_empty_and_softmax() -> None:
    empty = _concat_or_empty([], dtype=np.float32, shape=(0, 3))
    assert empty.shape == (0, 3)

    joined = _concat_or_empty(
        [np.array([[1.0, 2.0]], dtype=np.float32), np.array([[3.0, 4.0]], dtype=np.float32)],
        dtype=np.float32,
        shape=(0, 2),
    )
    assert joined.shape == (2, 2)

    probs = _softmax(np.array([[1.0, 2.0, 3.0]], dtype=np.float64))
    assert probs.shape == (1, 3)
    assert np.isclose(probs.sum(), 1.0)
