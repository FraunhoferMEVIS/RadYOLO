from __future__ import annotations

import json

from click.testing import CliRunner

from radyolo import main


def test_create_k_fold_split_cli(synthetic_dataset) -> None:
    splits_file = synthetic_dataset / "data_splits.json"
    if splits_file.exists():
        splits_file.unlink()

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "create-k-fold-split",
            str(synthetic_dataset),
            "--k",
            "5",
            "--seed",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert splits_file.exists(), "data_splits.json not created"


def test_create_k_fold_split_cli_without_test_group_allows_k2(synthetic_dataset) -> None:
    splits_file = synthetic_dataset / "data_splits.json"
    if splits_file.exists():
        splits_file.unlink()

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "create-k-fold-split",
            str(synthetic_dataset),
            "--k",
            "2",
            "--no-create-test-group",
            "--seed",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert splits_file.exists(), "data_splits.json not created"

    splits = json.loads(splits_file.read_text(encoding="utf-8"))
    assert len(splits) == 2
    assert all(fold["test"] == [] for fold in splits)
    assert sorted(splits[0]["train"] + splits[0]["val"]) == [
        "case1",
        "case2",
        "case3",
        "case4",
        "case5",
    ]


def test_create_train_val_test_split_cli(synthetic_dataset) -> None:
    splits_file = synthetic_dataset / "data_splits.json"
    if splits_file.exists():
        splits_file.unlink()

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "create-train-val-test-split",
            str(synthetic_dataset),
            "--train-percentage",
            "60",
            "--val-percentage",
            "20",
            "--test-percentage",
            "20",
            "--seed",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert splits_file.exists(), "data_splits.json not created"

    splits = json.loads(splits_file.read_text(encoding="utf-8"))
    assert len(splits) == 1
    assert len(splits[0]["train"]) == 3
    assert len(splits[0]["val"]) == 1
    assert len(splits[0]["test"]) == 1


def test_create_train_val_test_split_cli_allows_empty_groups(synthetic_dataset) -> None:
    splits_file = synthetic_dataset / "data_splits.json"
    if splits_file.exists():
        splits_file.unlink()

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "create-train-val-test-split",
            str(synthetic_dataset),
            "--train-percentage",
            "100",
            "--val-percentage",
            "0",
            "--test-percentage",
            "0",
            "--seed",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert splits_file.exists(), "data_splits.json not created"

    splits = json.loads(splits_file.read_text(encoding="utf-8"))
    assert len(splits) == 1
    assert len(splits[0]["train"]) == 5
    assert splits[0]["val"] == []
    assert splits[0]["test"] == []
