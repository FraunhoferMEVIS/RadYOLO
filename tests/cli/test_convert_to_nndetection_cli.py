from __future__ import annotations

from click.testing import CliRunner

from radyolo import main


def test_convert_to_nndetection_cli(synthetic_dataset, tmp_path) -> None:
    out_root = tmp_path / "nndet"
    out_root.mkdir()
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "convert-to-nndetection",
            str(synthetic_dataset),
            "--out-root",
            str(out_root),
            "--task-id",
            "123",
            "--task-name",
            "Tiny",
            "--fold",
            "0",
            "--workers",
            "0",
        ],
    )

    assert result.exit_code == 0
    expected_dir = out_root / "Task123_Tiny"
    assert expected_dir.exists(), "Task folder not created"
    dataset_yaml = expected_dir / "dataset.yaml"
    assert dataset_yaml.exists(), "nnDetection dataset.yaml missing"
