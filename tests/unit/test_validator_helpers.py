from __future__ import annotations

# ruff: noqa: SLF001
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest  # noqa: TC002
import torch

from radyolo.validation.validate import Validator


def _make_validator(tmp_path: Path, *, plots: bool = False) -> Validator:
    config = SimpleNamespace(
        confidence_threshold_validation=0.2,
        confidence_threshold_plotting=0.1,
        nms_iou_threshold=0.2,
        agnostic_nms=False,
        max_det=50,
        clip_boxes_to_image=True,
        plots=plots,
        normalization={"type": "none"},
        patch_size=SimpleNamespace(get_zyx=lambda: (4, 4, 4)),
        apply_preprocessing=False,
        preprocessing_device="cpu",
        validation_plot_projection="max",
        validation_plot_image_mode="default",
        validation_plot_difference_channels=(2, 0),
    )
    validator = Validator(
        dataloader=[],
        save_dir=tmp_path,
        config=config,
        names={0: "lesion"},
        save_predictions=False,
    )
    validator.device = torch.device("cpu")
    return validator


def test_get_dice_average_returns_zero_for_non_finite_values(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    validator = _make_validator(tmp_path)
    validator.dice_sum = float("nan")
    validator.dice_count = 1

    assert validator._get_dice_average() == 0.0
    assert "Encountered non-finite Dice aggregate" in caplog.text


def test_get_matches_handles_empty_and_class_filtering(tmp_path: Path) -> None:
    validator = _make_validator(tmp_path)

    m0, m1 = validator._get_matches(
        detections=torch.zeros((0, 8), dtype=torch.float32),
        gt_bboxes=torch.zeros((0, 6), dtype=torch.float32),
        iou_thres=0.1,
        gt_classes=torch.zeros((0,), dtype=torch.float32),
    )
    assert m0.size == 0
    assert m1.size == 0

    detections = torch.tensor([[0, 0, 0, 2, 2, 2, 0.9, 0]], dtype=torch.float32)
    gt_boxes = torch.tensor([[0, 0, 0, 2, 2, 2]], dtype=torch.float32)
    gt_classes_wrong = torch.tensor([1], dtype=torch.float32)
    m0_wrong, m1_wrong = validator._get_matches(detections, gt_boxes, 0.1, gt_classes_wrong)
    assert m0_wrong.size == 0
    assert m1_wrong.size == 0

    gt_classes_ok = torch.tensor([0], dtype=torch.float32)
    m0_ok, m1_ok = validator._get_matches(detections, gt_boxes, 0.1, gt_classes_ok)
    assert m0_ok.tolist() == [0]
    assert m1_ok.tolist() == [0]


def test_update_dice_for_image_skips_non_finite_reconstructed_masks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    validator = _make_validator(tmp_path)

    def _nan_reconstruction(coeff: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        _ = coeff
        _ = prototypes
        return torch.full((2, 2, 2), float("nan"), dtype=torch.float32)

    monkeypatch.setattr(
        "radyolo.validation.validate.ops.reconstruct_mask_probabilities",
        _nan_reconstruction,
    )

    predn = torch.tensor([[0, 0, 0, 2, 2, 2, 0.9, 0]], dtype=torch.float32)
    pred_with_coeffs = torch.tensor([[0, 0, 0, 2, 2, 2, 0.9, 0, 1.0]], dtype=torch.float32)
    gt_boxes = torch.tensor([[0, 0, 0, 2, 2, 2]], dtype=torch.float32)
    gt_cls = torch.tensor([0], dtype=torch.float32)
    prototypes = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    gt_masks = [torch.ones((2, 2, 2), dtype=torch.bool)]

    validator._update_dice_for_image(
        predn,
        pred_with_coeffs,
        gt_boxes,
        gt_cls,
        prototypes,
        gt_masks,
    )

    assert validator.dice_count == 0
    assert "reconstructed mask probabilites contain non-finite values" in caplog.text


def test_update_dice_for_image_accumulates_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _make_validator(tmp_path)

    def _ones_reconstruction(coeff: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        _ = coeff
        _ = prototypes
        return torch.ones((2, 2, 2), dtype=torch.float32)

    monkeypatch.setattr(
        "radyolo.validation.validate.ops.reconstruct_mask_probabilities",
        _ones_reconstruction,
    )

    predn = torch.tensor([[0, 0, 0, 2, 2, 2, 0.9, 0]], dtype=torch.float32)
    pred_with_coeffs = torch.tensor([[0, 0, 0, 2, 2, 2, 0.9, 0, 1.0]], dtype=torch.float32)
    gt_boxes = torch.tensor([[0, 0, 0, 2, 2, 2]], dtype=torch.float32)
    gt_cls = torch.tensor([0], dtype=torch.float32)
    prototypes = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    gt_masks = [torch.ones((2, 2, 2), dtype=torch.bool)]

    validator._update_dice_for_image(
        predn,
        pred_with_coeffs,
        gt_boxes,
        gt_cls,
        prototypes,
        gt_masks,
    )

    assert validator.dice_count == 1
    assert np.isclose(validator.dice_sum, 1.0, atol=1e-5)


def test_match_predictions_with_missing_class_and_class_agnostic(tmp_path: Path) -> None:
    validator = _make_validator(tmp_path)

    pred_classes = torch.tensor([0], dtype=torch.float32)
    true_classes = torch.tensor([-1], dtype=torch.float32)
    iou = torch.tensor([[1.0]], dtype=torch.float32)

    tp, fp = validator.match_predictions(pred_classes, true_classes, iou, class_agnostic=False)
    assert not tp.any()
    assert not fp.any()

    tp_agn, fp_agn = validator.match_predictions(
        pred_classes,
        true_classes,
        iou,
        class_agnostic=True,
    )
    assert tp_agn.any()
    assert not fp_agn.any()


def test_get_stats_skips_metric_process_when_no_true_positives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _make_validator(tmp_path)

    validator.stats = {
        "tp": [torch.zeros((1, 18), dtype=torch.bool)],
        "fp": [torch.ones((1, 18), dtype=torch.bool)],
        "tp_agnostic": [torch.zeros((1, 18), dtype=torch.bool)],
        "fp_agnostic": [torch.ones((1, 18), dtype=torch.bool)],
        "conf": [torch.tensor([0.2], dtype=torch.float32)],
        "pred_cls": [torch.tensor([0], dtype=torch.float32)],
        "target_cls": [torch.tensor([-1], dtype=torch.float32)],
        "target_img": [torch.tensor([-1], dtype=torch.float32)],
    }
    called = {"value": False}

    def _mark_called(**kwargs: object) -> None:
        _ = kwargs
        called["value"] = True

    monkeypatch.setattr(validator.metrics, "process", _mark_called)

    stats = validator.get_stats()

    assert called["value"] is False
    assert stats["metrics/dice"] == 0.0
    assert validator.nt_per_class is not None
    assert validator.nt_per_image is not None


def test_plot_validation_cases_clips_predictions_and_sanitizes_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _make_validator(tmp_path, plots=True)
    captured: dict[str, object] = {}

    def _capture_plot_validation_case(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        "radyolo.validation.validate.plot_validation_case",
        _capture_plot_validation_case,
    )

    batch = {
        "batch_idx": torch.tensor([0], dtype=torch.int64),
        "cls": torch.tensor([[0.0]], dtype=torch.float32),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.4, 0.4, 0.4]], dtype=torch.float32),
        "img": torch.zeros((1, 1, 4, 4, 4), dtype=torch.float32),
        "case_names": ["case:name/with spaces"],
    }
    preds = [torch.tensor([[-1.0, -1.0, -1.0, 6.0, 6.0, 6.0, 0.9, 0.0]], dtype=torch.float32)]

    validator.plot_validation_cases(batch, preds)

    assert "predicted_boxes" in captured
    predicted_boxes = captured["predicted_boxes"]
    assert isinstance(predicted_boxes, torch.Tensor)
    assert torch.min(predicted_boxes).item() >= 0.0

    fname = captured["fname"]
    assert isinstance(fname, Path)
    assert fname.name == "case_name_with_spaces.jpg"

    assert Validator._sanitize_case_name("abc:123") == "abc_123"
