from __future__ import annotations

from click.testing import CliRunner

from radyolo import main


def test_create_data_splits_cli(synthetic_dataset) -> None:
    splits_file = synthetic_dataset / "data_splits.json"
    if splits_file.exists():
        splits_file.unlink()

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "create-data-splits",
            str(synthetic_dataset),
            "--k",
            "5",
            "--seed",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert splits_file.exists(), "data_splits.json not created"
