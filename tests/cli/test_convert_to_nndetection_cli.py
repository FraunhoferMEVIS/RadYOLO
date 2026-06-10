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


def test_convert_to_nndetection_cli_custom_split_file_name(synthetic_dataset, tmp_path) -> None:
    out_root = tmp_path / "nndet_custom_split"
    out_root.mkdir()
    runner = CliRunner()

    source_split = synthetic_dataset / "data_splits.json"
    custom_split = synthetic_dataset / "my_custom_splits.json"
    custom_split.write_text(source_split.read_text(encoding="utf-8"), encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "convert-to-nndetection",
            str(synthetic_dataset),
            "--out-root",
            str(out_root),
            "--task-id",
            "124",
            "--task-name",
            "TinyCustomSplit",
            "--fold",
            "0",
            "--workers",
            "0",
            "--split-file-name",
            "my_custom_splits.json",
        ],
    )

    assert result.exit_code == 0
    expected_dir = out_root / "Task124_TinyCustomSplit"
    assert expected_dir.exists(), "Task folder not created"
    splits_pkl = expected_dir / "splits_final.pkl"
    assert splits_pkl.exists(), "splits_final.pkl missing"
