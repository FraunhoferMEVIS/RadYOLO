from __future__ import annotations

from types import SimpleNamespace

from radyolo.inference import predict as predict_module
from radyolo.inference.predict import PredictionConfig, _load_model


def _prediction_config(tmp_path):
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"checkpoint")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return PredictionConfig(
        model_path=model_path,
        input_dir=input_dir,
        output_dir=output_dir,
        confidence_threshold=0.1,
        nms_iou_threshold=0.1,
        device="cpu",
        clip_boxes_to_image=True,
    )


def test_load_model_uses_checkpoint_dataset_names_dict(monkeypatch, tmp_path) -> None:
    cfg = _prediction_config(tmp_path)

    fake_loaded_model = SimpleNamespace(
        model=object(),
        train_config=SimpleNamespace(num_classes=3),
        checkpoint={"model_metadata": {"dataset_yaml": {"names": {"0": "liver", 1: "kidney"}}}},
    )
    monkeypatch.setattr(predict_module, "load_checkpoint_model", lambda *_args, **_kwargs: fake_loaded_model)

    _model, _train_config, names = _load_model(cfg)

    assert names == {0: "liver", 1: "kidney", 2: "class_2"}


def test_load_model_falls_back_when_checkpoint_dataset_names_is_list(monkeypatch, tmp_path) -> None:
    cfg = _prediction_config(tmp_path)

    fake_loaded_model = SimpleNamespace(
        model=object(),
        train_config=SimpleNamespace(num_classes=2),
        checkpoint={"model_metadata": {"dataset_yaml": {"names": ["lesion", "organ"]}}},
    )
    monkeypatch.setattr(predict_module, "load_checkpoint_model", lambda *_args, **_kwargs: fake_loaded_model)

    _model, _train_config, names = _load_model(cfg)

    assert names == {0: "class_0", 1: "class_1"}


def test_load_model_falls_back_to_default_class_names(monkeypatch, tmp_path) -> None:
    cfg = _prediction_config(tmp_path)

    fake_loaded_model = SimpleNamespace(
        model=object(),
        train_config=SimpleNamespace(num_classes=2),
        checkpoint={"model_metadata": {"dataset_yaml": {}}},
    )
    monkeypatch.setattr(predict_module, "load_checkpoint_model", lambda *_args, **_kwargs: fake_loaded_model)

    _model, _train_config, names = _load_model(cfg)

    assert names == {0: "class_0", 1: "class_1"}
