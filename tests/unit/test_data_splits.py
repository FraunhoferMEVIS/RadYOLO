from __future__ import annotations

import json
from pathlib import Path

from radyolo.data.data_splits import k_fold_data_splits, train_val_test_data_split


def _create_case_files(dataset_path: Path, case_names: list[str]) -> None:
    images_dir = dataset_path / "images"
    images_dir.mkdir(parents=True)
    for case_name in case_names:
        (images_dir / f"{case_name}.nii.gz").touch()


def _load_first_fold(split_file_path: Path) -> dict[str, list[str]]:
    splits = json.loads(split_file_path.read_text(encoding="utf-8"))
    return splits[0]


def _split_for_case(case_name: str, fold: dict[str, list[str]]) -> str:
    if case_name in fold["train"]:
        return "train"
    if case_name in fold["val"]:
        return "val"
    if case_name in fold["test"]:
        return "test"
    msg = f"Case not found in fold: {case_name}"
    raise AssertionError(msg)


def test_train_val_test_split_group_by_prefix_keeps_groups_together(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    case_names = [
        "patient1_scan1",
        "patient1_scan2",
        "patient2_scan1",
        "patient2_scan2",
        "patient3_scan1",
        "patient3_scan2",
    ]
    _create_case_files(dataset_path, case_names)

    split_file_path = train_val_test_data_split(
        dataset_path=dataset_path,
        seed=123,
        train_percentage=34.0,
        val_percentage=33.0,
        test_percentage=33.0,
        group_by_prefix=True,
        prefix_sep="_",
    )

    fold = _load_first_fold(split_file_path)

    assert _split_for_case("patient1_scan1", fold) == _split_for_case("patient1_scan2", fold)
    assert _split_for_case("patient2_scan1", fold) == _split_for_case("patient2_scan2", fold)
    assert _split_for_case("patient3_scan1", fold) == _split_for_case("patient3_scan2", fold)


def test_k_fold_split_group_by_prefix_keeps_groups_together_per_fold(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    case_names = [
        "patient1_scan1",
        "patient1_scan2",
        "patient2_scan1",
        "patient2_scan2",
        "patient3_scan1",
        "patient3_scan2",
    ]
    _create_case_files(dataset_path, case_names)

    split_file_path = k_fold_data_splits(
        dataset_path=dataset_path,
        seed=123,
        k=3,
        group_by_prefix=True,
        prefix_sep="_",
    )

    splits = json.loads(split_file_path.read_text(encoding="utf-8"))
    for fold in splits:
        assert _split_for_case("patient1_scan1", fold) == _split_for_case("patient1_scan2", fold)
        assert _split_for_case("patient2_scan1", fold) == _split_for_case("patient2_scan2", fold)
        assert _split_for_case("patient3_scan1", fold) == _split_for_case("patient3_scan2", fold)
