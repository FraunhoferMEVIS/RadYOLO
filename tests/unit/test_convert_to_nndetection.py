from __future__ import annotations

import json
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import pytest
import yaml

from radyolo.data.convert_to_nndetection import (
    _compose_instance_seg_from_boxes,
    _compose_instance_seg_from_masks,
    _convert_one_case,
    _copy_image_to_nndet,
    _labels_for_case,
    _load_dataset_names,
    _read_splits,
    _try_load_masks,
    _write_label_files,
    convert_dataset_to_nndetection,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_nifti(path: Path, array: np.ndarray, *, dtype: np.dtype = np.float32) -> None:
    image = nib.Nifti1Image(array.astype(dtype), affine=np.eye(4))
    nib.save(image, path)


def test_load_dataset_names_variants(tmp_path: Path) -> None:
    assert _load_dataset_names(tmp_path / "missing.yaml") is None

    class_names_yaml = tmp_path / "class_names.yaml"
    class_names_yaml.write_text("class_names: [lesion, organ]\n", encoding="utf-8")
    assert _load_dataset_names(class_names_yaml) == {0: "lesion", 1: "organ"}

    names_yaml = tmp_path / "names.yaml"
    names_yaml.write_text("names: [tumor, vessel]\n", encoding="utf-8")
    assert _load_dataset_names(names_yaml) == {0: "tumor", 1: "vessel"}


def test_read_splits_fallback_and_invalid_fold(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "images").mkdir(parents=True)
    _write_nifti(dataset_dir / "images" / "case_a.nii.gz", np.zeros((4, 4, 4), dtype=np.float32))

    fallback = _read_splits(dataset_dir, fold=0, split_file_name="custom_splits.json")
    assert len(fallback.train) == 1
    assert fallback.val == []
    assert fallback.test == []

    (dataset_dir / "custom_splits.json").write_text(
        json.dumps([{"train": ["case_a"], "val": [], "test": []}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid fold"):
        _read_splits(dataset_dir, fold=2, split_file_name="custom_splits.json")


def test_labels_for_case_handles_empty_and_invalid_lines(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "case1.txt").write_text(
        "0 0.5 0.5 0.5 0.4 0.4 0.4\n\n1 0.2 0.2 0.2 0.1 0.1 0.1\n",
        encoding="utf-8",
    )

    labels = _labels_for_case(labels_dir, "case1")
    assert len(labels) == 2
    assert labels[0][0] == 0
    assert labels[1][0] == 1

    (labels_dir / "bad.txt").write_text("0 0.5 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid label line"):
        _labels_for_case(labels_dir, "bad")


def test_try_load_masks_and_mask_composition(tmp_path: Path) -> None:
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()

    assert _try_load_masks(masks_dir, "missing") is None

    (masks_dir / "broken.nii.gz").write_text("not-a-nifti", encoding="utf-8")
    assert _try_load_masks(masks_dir, "broken") is None

    valid_mask = np.zeros((2, 4, 4, 4), dtype=np.uint8)
    valid_mask[0, 0:2, 0:2, 0:2] = 1
    valid_mask[1, 2:4, 2:4, 2:4] = 1
    _write_nifti(masks_dir / "ok.nii.gz", valid_mask, dtype=np.uint8)
    loaded = _try_load_masks(masks_dir, "ok")
    assert loaded is not None
    assert loaded.shape == (2, 4, 4, 4)

    three_dim = np.zeros((4, 4, 4), dtype=np.uint8)
    three_dim[1:3, 1:3, 1:3] = 1
    single_instance = _compose_instance_seg_from_masks(three_dim, num_instances=1)
    assert single_instance is not None
    assert single_instance.max() == 1
    assert _compose_instance_seg_from_masks(three_dim, num_instances=2) is None
    assert (
        _compose_instance_seg_from_masks(np.zeros((4, 4), dtype=np.uint8), num_instances=1)
        is None
    )

    channels_last = np.zeros((4, 4, 4, 2), dtype=np.uint8)
    channels_last[0:2, 0:2, 0:2, 0] = 1
    channels_last[2:4, 2:4, 2:4, 1] = 1
    composed_last = _compose_instance_seg_from_masks(channels_last, num_instances=2)
    assert composed_last is not None
    assert composed_last[0, 0, 0] == 1
    assert composed_last[3, 3, 3] == 2

    channels_first = np.zeros((2, 4, 4, 4), dtype=np.uint8)
    channels_first[0, 0:2, 0:2, 0:2] = 1
    channels_first[1, 2:4, 2:4, 2:4] = 1
    composed_first = _compose_instance_seg_from_masks(channels_first, num_instances=2)
    assert composed_first is not None
    assert composed_first[0, 0, 0] == 1
    assert composed_first[3, 3, 3] == 2

    assert _compose_instance_seg_from_masks(channels_first, num_instances=3) is None


def test_compose_instance_seg_from_boxes_skips_degenerate_boxes() -> None:
    labelmap = _compose_instance_seg_from_boxes(
        (10, 10, 10),
        [
            (0.5, 0.5, 0.5, 0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5, 0.4, 0.4, 0.4),
        ],
    )
    assert labelmap.shape == (10, 10, 10)
    assert np.max(labelmap) == 2
    assert np.count_nonzero(labelmap == 2) > 0


def test_copy_image_and_write_label_files(tmp_path: Path) -> None:
    image_path = tmp_path / "src.nii.gz"
    data = np.ones((5, 5, 5), dtype=np.float32)
    _write_nifti(image_path, data)

    out_images_dir = tmp_path / "out_images"
    copied = _copy_image_to_nndet(image_path, out_images_dir, "caseX")
    assert copied.exists()
    assert copied.name == "caseX_0000.nii.gz"

    out_labels_dir = tmp_path / "out_labels"
    labelmap = np.zeros((5, 5, 5), dtype=np.uint16)
    labelmap[1:3, 1:3, 1:3] = 1
    ref_nii = nib.load(copied)
    _write_label_files(out_labels_dir, "caseX", labelmap, [7], ref_nii)

    label_path = out_labels_dir / "caseX.nii.gz"
    props_path = out_labels_dir / "caseX.json"
    assert label_path.exists()
    assert props_path.exists()
    properties = json.loads(props_path.read_text(encoding="utf-8"))
    assert properties == {"instances": {"1": 7}}


def test_convert_one_case_success_and_failure_paths(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    for subdir in ("images", "labels", "masks"):
        (dataset_dir / subdir).mkdir(parents=True)

    image = np.zeros((8, 8, 8), dtype=np.float32)
    image[2:6, 2:6, 2:6] = 1
    _write_nifti(dataset_dir / "images" / "good.case.nii.gz", image)
    (dataset_dir / "labels" / "good.case.txt").write_text(
        "1 0.5 0.5 0.5 0.5 0.5 0.5\n", encoding="utf-8"
    )

    out_img_tr = tmp_path / "raw_splitted" / "imagesTr"
    out_lbl_tr = tmp_path / "raw_splitted" / "labelsTr"
    result = _convert_one_case(
        str(dataset_dir),
        "good.case",
        str(out_img_tr),
        str(out_lbl_tr),
    )
    assert result[0] is True
    assert result[1] == "train"
    assert result[5] == "good_case"
    assert (out_img_tr / "good_case_0000.nii.gz").exists()
    assert (out_lbl_tr / "good_case.nii.gz").exists()

    missing = _convert_one_case(
        str(dataset_dir),
        "missing",
        str(out_img_tr),
        str(out_lbl_tr),
    )
    assert missing[0] is False
    assert missing[2] == "missing.nii.gz"

    _write_nifti(dataset_dir / "images" / "bad.nii.gz", image)
    (dataset_dir / "labels" / "bad.txt").write_text("bad line\n", encoding="utf-8")
    bad = _convert_one_case(
        str(dataset_dir),
        "bad",
        str(tmp_path / "raw_splitted" / "imagesTs"),
        str(tmp_path / "raw_splitted" / "labelsTs"),
    )
    assert bad[0] is False
    assert bad[1] == "test"


def test_convert_dataset_to_nndetection_infers_labels_when_names_missing(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    for subdir in ("images", "labels", "masks"):
        (dataset_dir / subdir).mkdir(parents=True)

    _write_nifti(dataset_dir / "images" / "case1.nii.gz", np.ones((6, 6, 6), dtype=np.float32))
    (dataset_dir / "labels" / "case1.txt").write_text(
        "2 0.5 0.5 0.5 0.6 0.6 0.6\n", encoding="utf-8"
    )
    (dataset_dir / "data_splits.json").write_text(
        json.dumps([{"train": ["case1"], "val": [], "test": []}]),
        encoding="utf-8",
    )

    task_dir = convert_dataset_to_nndetection(
        dataset_dir=dataset_dir,
        out_root=tmp_path / "out",
        workers=1,
        split_file_name="data_splits.json",
    )
    dataset_yaml = yaml.safe_load((task_dir / "dataset.yaml").read_text(encoding="utf-8"))
    assert dataset_yaml["labels"] == {"2": "class_2"}
