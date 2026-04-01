from __future__ import annotations

import torch

from radyolo.config.spatial import PatchSize
from radyolo.data.data import WholeImageDataset
from radyolo.data.preprocess_data import preprocess_dataset


def test_dataset_loads_preprocessed_blosc2_images_and_masks(
    synthetic_dataset,
    tmp_path,
) -> None:
    preprocessed_dataset = preprocess_dataset(
        dataset_path=synthetic_dataset,
        output_path=tmp_path / "preprocessed_blosc2_dataset",
        workers=0,
        image_format="blosc2",
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