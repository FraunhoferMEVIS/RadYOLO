from __future__ import annotations

import json
import shutil

import torch

from radyolo.config.spatial import PatchSize
from radyolo.data.data import WholeImageDataset
from radyolo.data.preprocess_data import preprocess_dataset
from radyolo.data.tabular import load_case_tabular_inputs


def test_dataset_loads_preprocessed_blosc2_images_and_masks(
    synthetic_dataset,
    tmp_path,
) -> None:
    preprocessed_dataset = preprocess_dataset(
        dataset_path=synthetic_dataset,
        output_path=tmp_path / "preprocessed_blosc2_dataset",
        workers=0,
    )

    dataset = WholeImageDataset(
        src_path=preprocessed_dataset,
        patch_size=PatchSize(x=32, y=32, z=64),
        augment=False,
    )

    assert dataset.image_format == "blosc2"

    image = dataset.load_image("case1")
    _classes, _bboxes, masks = dataset.load_labels_and_masks("case1")

    assert image.dtype == torch.float32
    assert image.shape == (64, 32, 32)
    assert torch.isclose(image.max(), torch.tensor(1.0, dtype=torch.float32))
    assert masks is not None
    assert masks.dtype == torch.uint8
    assert masks.shape == (1, 64, 32, 32)


def test_whole_image_dataset_length_includes_foreground_oversamples(
    synthetic_dataset,
) -> None:
    dataset = WholeImageDataset(
        src_path=synthetic_dataset,
        patch_size=PatchSize(x=32, y=32, z=64),
        augment=False,
        foreground_oversampling=0.5,
    )

    assert len(dataset.cases) == 3
    assert dataset.foreground_oversample_count == 3
    assert len(dataset) == 6


def test_dataset_loads_optional_global_image_labels_and_preprocess_copies_them(
    synthetic_dataset,
    tmp_path,
) -> None:
    dataset_path = tmp_path / "dataset_with_image_labels"
    shutil.copytree(synthetic_dataset, dataset_path)
    (dataset_path / "labels_image").mkdir()
    (dataset_path / "inputs_tabular").mkdir()
    (dataset_path / "labels_image" / "case1.json").write_text(
        json.dumps({"tumor_stage": 2, "survival_days": 418.5, "relapse": False}),
        encoding="utf-8",
    )
    (dataset_path / "inputs_tabular" / "case1.json").write_text(
        json.dumps(
            {
                "Age": 73.0,
                "Gender": 1.0,
                "Treatment": 1.0,
                "HPV_Status": None,
            }
        ),
        encoding="utf-8",
    )
    (dataset_path / "dataset.yaml").write_text(
        "nc: 1\n"
        "names: {0: lesion}\n"
        "image_label_tasks:\n"
        "  tumor_stage:\n"
        "    type: multiclass\n"
        "    classes: {0: stage_i, 1: stage_ii, 2: stage_iii}\n"
        "  survival_days:\n"
        "    type: survival\n"
        "    masking: relapse\n",
        encoding="utf-8",
    )

    preprocessed_dataset = preprocess_dataset(
        dataset_path=dataset_path,
        output_path=tmp_path / "preprocessed_blosc2_dataset_with_image_labels",
        workers=0,
    )

    dataset = WholeImageDataset(
        src_path=preprocessed_dataset,
        patch_size=PatchSize(x=32, y=32, z=64),
        augment=False,
    )

    sample = dataset.load_image_label_and_mask(0)

    assert sample["image_labels"] == {
        "tumor_stage": 2,
        "survival_days": 418.5,
        "relapse": False,
    }
    assert dataset.collate_fn([sample])["image_labels"] == (
        {"tumor_stage": 2, "survival_days": 418.5, "relapse": False},
    )
    assert (preprocessed_dataset / "labels_image" / "case1.json").exists()
    assert (preprocessed_dataset / "inputs_tabular" / "case1.json").exists()

    tabular_values, tabular_mask = load_case_tabular_inputs(
        preprocessed_dataset,
        "case1",
        ("Age", "Gender", "Treatment", "HPV_Status"),
    )
    assert torch.equal(tabular_mask, torch.tensor([True, True, True, False]))
    assert torch.allclose(tabular_values, torch.tensor([73.0, 1.0, 1.0, 0.0]))


def test_dataset_loads_null_global_image_labels_as_none(
    synthetic_dataset,
    tmp_path,
) -> None:
    dataset_path = tmp_path / "dataset_with_missing_image_labels"
    shutil.copytree(synthetic_dataset, dataset_path)
    (dataset_path / "labels_image").mkdir()
    (dataset_path / "labels_image" / "case1.json").write_text(
        json.dumps({"tumor_stage": None, "survival_days": None, "relapse": None}),
        encoding="utf-8",
    )
    (dataset_path / "dataset.yaml").write_text(
        "nc: 1\n"
        "names: {0: lesion}\n"
        "image_label_tasks:\n"
        "  tumor_stage:\n"
        "    type: multiclass\n"
        "    classes: {0: stage_i, 1: stage_ii, 2: stage_iii}\n"
        "  survival_days:\n"
        "    type: survival\n"
        "    masking: relapse\n",
        encoding="utf-8",
    )

    dataset = WholeImageDataset(
        src_path=dataset_path,
        patch_size=PatchSize(x=32, y=32, z=64),
        augment=False,
    )

    sample = dataset.load_image_label_and_mask(0)

    assert sample["image_labels"] == {
        "tumor_stage": None,
        "survival_days": None,
        "relapse": None,
    }
