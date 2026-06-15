from __future__ import annotations

import json

from click.testing import CliRunner

from radyolo import main


def test_convert_to_nnunet_cli(synthetic_dataset, tmp_path) -> None:
    out_root = tmp_path / "nnunet"
    out_root.mkdir()
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "convert-to-nnunet",
            str(synthetic_dataset),
            "--out-root",
            str(out_root),
            "--dataset-id",
            "123",
            "--dataset-name",
            "Tiny",
            "--fold",
            "0",
            "--workers",
            "0",
        ],
    )

    assert result.exit_code == 0
    expected_dir = out_root / "Dataset123_Tiny"
    assert expected_dir.exists(), "Dataset folder not created"
    dataset_json = expected_dir / "dataset.json"
    assert dataset_json.exists(), "nnU-Net dataset.json missing"
    payload = json.loads(dataset_json.read_text(encoding="utf-8"))
    assert payload["labels"] == {"background": 0, "lesion": 1}
    assert payload["numTraining"] == 4
    assert (expected_dir / "splits_final.json").exists(), "splits_final.json missing"
    assert (expected_dir / "labelsTs" / "case5.nii.gz").exists(), "nnU-Net test label mask missing"


def test_convert_to_nnunet_cli_custom_split_file_names(synthetic_dataset, tmp_path) -> None:
    out_root = tmp_path / "nnunet_custom_split"
    out_root.mkdir()
    runner = CliRunner()

    source_split = synthetic_dataset / "data_splits.json"
    custom_split = synthetic_dataset / "my_custom_splits.json"
    custom_split.write_text(source_split.read_text(encoding="utf-8"), encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "convert-to-nnunet",
            str(synthetic_dataset),
            "--out-root",
            str(out_root),
            "--dataset-id",
            "124",
            "--dataset-name",
            "TinyCustomSplit",
            "--fold",
            "0",
            "--workers",
            "0",
            "--split-file-name",
            "my_custom_splits.json",
            "--export-split-file-name",
            "my_exported_splits.json",
        ],
    )

    assert result.exit_code == 0
    expected_dir = out_root / "Dataset124_TinyCustomSplit"
    assert expected_dir.exists(), "Dataset folder not created"
    assert (expected_dir / "my_exported_splits.json").exists(), "custom split export missing"