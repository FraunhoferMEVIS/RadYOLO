from __future__ import annotations

import json

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


def test_analyze_dataset_cli_custom_output_file(synthetic_dataset, tmp_path) -> None:
    runner = CliRunner()
    output_file = tmp_path / "fingerprints" / "custom_fingerprint.json"

    result = runner.invoke(
        main,
        [
            "analyze-dataset",
            str(synthetic_dataset),
            "--workers",
            "0",
            "--output-file",
            str(output_file),
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists(), "custom fingerprint output file not created"
    fingerprint = json.loads(output_file.read_text(encoding="utf-8"))
    assert "foreground_channel_stats" in fingerprint
    assert fingerprint["n_images_with_objects"] == 5
