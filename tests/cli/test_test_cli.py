from __future__ import annotations

import json

from click.testing import CliRunner

from radyolo import main


def test_test_cli(monkeypatch, trained_run, synthetic_dataset) -> None:
    class FakeValidator:
        def __init__(self, dataloader, save_dir, config, names, save_predictions) -> None:
            del dataloader, config, names, save_predictions
            self.save_dir = save_dir

        def __call__(self, trainer, model) -> None:
            del trainer, model
            self.save_dir.mkdir(parents=True, exist_ok=True)
            (self.save_dir / "predictions.json").write_text("[]", encoding="utf-8")

    monkeypatch.setattr("radyolo.Validator", FakeValidator)

    model_path = trained_run / "weights_best.pt"
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "test",
            str(model_path),
            str(synthetic_dataset),
            "--fold",
            "0",
            "--group",
            "val",
        ],
    )

    assert result.exit_code == 0
    val_dir = trained_run / "validation" / "val"
    assert val_dir.exists(), "Validation output directory missing"
    predictions = val_dir / "predictions.json"
    assert predictions.exists(), "predictions.json missing"
    assert json.loads(predictions.read_text(encoding="utf-8")) == []
