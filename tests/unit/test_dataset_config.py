from __future__ import annotations

import pytest

from radyolo.config.config import DatasetConfig


def test_dataset_config_derives_nc_from_names() -> None:
    config = DatasetConfig.model_validate({"names": {0: "lesion", 1: "organ"}})

    assert config.num_classes == 2


def test_dataset_config_rejects_legacy_nc_key() -> None:
    with pytest.raises(ValueError, match="must not define 'nc'"):
        DatasetConfig.model_validate({"nc": 1, "names": {0: "lesion"}})


def test_dataset_config_requires_contiguous_class_indices() -> None:
    with pytest.raises(KeyError, match="requires class indices 0-1"):
        DatasetConfig.model_validate({"names": {0: "lesion", 2: "organ"}})


def test_dataset_config_preserves_normalization_and_image_label_tasks() -> None:
    config = DatasetConfig.model_validate(
        {
            "names": {0: "lesion"},
            "normalization": {0: "CT"},
            "image_label_tasks": {
                "tumor_stage": {"type": "multiclass", "classes": {0: "i", 1: "ii"}}
            },
        }
    )

    assert config.normalization == {0: "CT"}
    assert config.image_label_tasks["tumor_stage"]["type"] == "multiclass"
