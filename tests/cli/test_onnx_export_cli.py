from __future__ import annotations

from click.testing import CliRunner

from radyolo import main


def test_onnx_export_cli(trained_run) -> None:
    model_path = trained_run / "weights" / "last.pt"
    runner = CliRunner()

    result = runner.invoke(main, ["onnx-export", str(model_path)])

    assert result.exit_code == 0
    assert model_path.with_suffix(".onnx").exists(), "ONNX export not created"
