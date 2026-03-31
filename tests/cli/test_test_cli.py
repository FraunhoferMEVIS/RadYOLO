from __future__ import annotations

from click.testing import CliRunner

from radyolo import main


def test_test_cli(trained_run, synthetic_dataset) -> None:
    model_path = trained_run / "weights" / "best.pt"
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
