from __future__ import annotations

from click.testing import CliRunner

from radyolo import main


def test_analyze_dataset_cli(synthetic_dataset) -> None:
    runner = CliRunner()

    result = runner.invoke(
        main,
        ["analyze-dataset", str(synthetic_dataset), "--workers", "0"],
    )

    assert result.exit_code == 0
    fingerprint = synthetic_dataset / "dataset_fingerprint.json"
    assert fingerprint.exists(), "dataset_fingerprint.json not created"
