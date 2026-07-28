from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from radyolo.inference.predict_ensemble_tta import (
    _aggregate_detection_clusters,
    _LoadedEnsembleModel,
    _PassPrediction,
    _unflip_detections,
    predict_loaded_image_ensemble_tta,
)
from radyolo.inference.predict_utils import LoadedImageTask, WriteTask


def test_aggregate_detection_clusters_weights_confidence_by_pass_support() -> None:
    pass_predictions = [
        _PassPrediction(
            detections=torch.tensor([[0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 0.8, 1.0]]),
            masks=[torch.ones((2, 2, 2), dtype=torch.float32)],
            original_shape=(20, 20, 20),
            pass_index=0,
        ),
        _PassPrediction(
            detections=torch.tensor([[1.0, 1.0, 1.0, 11.0, 11.0, 11.0, 0.6, 1.0]]),
            masks=[torch.full((2, 2, 2), 0.5, dtype=torch.float32)],
            original_shape=(20, 20, 20),
            pass_index=1,
        ),
    ]

    detections, masks = _aggregate_detection_clusters(
        pass_predictions,
        iou_threshold=0.5,
        confidence_threshold=0.1,
        class_agnostic=False,
        total_passes=4,
        max_det=10,
    )

    assert detections.shape == (1, 8)
    assert torch.allclose(detections[0, :6], torch.tensor([0.5, 0.5, 0.5, 10.5, 10.5, 10.5]))
    assert detections[0, 6].item() == pytest.approx(0.35)
    assert detections[0, 7].item() == 1.0
    assert len(masks) == 1
    assert np.allclose(masks[0], np.full((2, 2, 2), 0.75, dtype=np.float32))


def test_aggregate_detection_clusters_keeps_classes_separate_by_default() -> None:
    pass_predictions = [
        _PassPrediction(
            detections=torch.tensor([[0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 0.8, 1.0]]),
            masks=[None],
            original_shape=(20, 20, 20),
            pass_index=0,
        ),
        _PassPrediction(
            detections=torch.tensor([[0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 0.6, 2.0]]),
            masks=[None],
            original_shape=(20, 20, 20),
            pass_index=1,
        ),
    ]

    class_aware, _ = _aggregate_detection_clusters(
        pass_predictions,
        iou_threshold=0.5,
        confidence_threshold=0.1,
        class_agnostic=False,
        total_passes=2,
        max_det=10,
    )
    class_agnostic, _ = _aggregate_detection_clusters(
        pass_predictions,
        iou_threshold=0.5,
        confidence_threshold=0.1,
        class_agnostic=True,
        total_passes=2,
        max_det=10,
    )

    assert class_aware.shape[0] == 2
    assert class_agnostic.shape[0] == 1


def test_aggregate_detection_clusters_selects_class_by_summed_confidence() -> None:
    pass_predictions = [
        _PassPrediction(
            detections=torch.tensor([[0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 0.9, 1.0]]),
            masks=[None],
            original_shape=(20, 20, 20),
            pass_index=0,
        ),
        _PassPrediction(
            detections=torch.tensor([[0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 0.6, 2.0]]),
            masks=[None],
            original_shape=(20, 20, 20),
            pass_index=1,
        ),
        _PassPrediction(
            detections=torch.tensor([[0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 0.5, 2.0]]),
            masks=[None],
            original_shape=(20, 20, 20),
            pass_index=2,
        ),
    ]

    detections, _ = _aggregate_detection_clusters(
        pass_predictions,
        iou_threshold=0.5,
        confidence_threshold=0.1,
        class_agnostic=True,
        total_passes=3,
        max_det=10,
    )

    assert detections.shape == (1, 8)
    assert detections[0, 7].item() == 2.0


def test_aggregate_detection_clusters_filters_low_weighted_confidence() -> None:
    pass_predictions = [
        _PassPrediction(
            detections=torch.tensor([[0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 0.2, 1.0]]),
            masks=[torch.ones((2, 2, 2), dtype=torch.float32)],
            original_shape=(20, 20, 20),
            pass_index=0,
        ),
        _PassPrediction(
            detections=torch.tensor([[20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 0.9, 1.0]]),
            masks=[torch.zeros((2, 2, 2), dtype=torch.float32)],
            original_shape=(20, 20, 20),
            pass_index=1,
        ),
    ]

    detections, masks = _aggregate_detection_clusters(
        pass_predictions,
        iou_threshold=0.5,
        confidence_threshold=0.2,
        class_agnostic=False,
        total_passes=8,
        max_det=10,
    )

    assert detections.shape == (0, 8)
    assert masks == []


def test_unflip_detections_restores_box_coordinates_for_all_axes() -> None:
    detections = torch.tensor([[2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 0.7, 0.0]])

    unflipped = _unflip_detections(detections, (0, 1, 2), (20, 30, 40))

    assert torch.equal(
        unflipped,
        torch.tensor([[32.0, 21.0, 10.0, 38.0, 27.0, 16.0, 0.7, 0.0]]),
    )


def test_predict_loaded_image_ensemble_tta_uses_pass_original_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction_config = SimpleNamespace(
        tta_flipping=False,
        ensembling_iou=0.5,
        ensembling_confidence_threshold=0.1,
        ensembling_class_agnostic=False,
        save_masks=False,
    )
    loaded_model = _LoadedEnsembleModel(
        model=object(),
        train_config=SimpleNamespace(max_det=10),
        names={0: "class_0"},
        prediction_config=prediction_config,
    )
    loaded_image = LoadedImageTask(
        case_name="case_a",
        affine=np.eye(4, dtype=np.float32),
        image=np.zeros((1, 5, 6, 7), dtype=np.float32),
    )

    def fake_predict_loaded_image(*_args: object, **_kwargs: object) -> WriteTask:
        return WriteTask(
            case_name="case_a",
            image_results=[
                {
                    "class": 0,
                    "class_name": "class_0",
                    "confidence": 0.8,
                    "box_image_xyzxyz": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    "box_world_xyzxyz": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    "box_relative_xyzxyz": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                }
            ],
            affine=np.eye(4, dtype=np.float32),
            original_shape=(5, 6, 7),
            prototype_masks=None,
            mask_detections=[],
        )

    monkeypatch.setattr(
        "radyolo.inference.predict._predict_loaded_image",
        fake_predict_loaded_image,
    )

    write_task = predict_loaded_image_ensemble_tta(
        loaded_image,
        loaded_models=[loaded_model],
        prediction_config=prediction_config,
    )

    assert write_task.original_shape == (5, 6, 7)
    assert write_task.image_results[0]["box_relative_xyzxyz"] == [
        0.142857,
        0.333333,
        0.6,
        0.571429,
        0.833333,
        1.2,
    ]
