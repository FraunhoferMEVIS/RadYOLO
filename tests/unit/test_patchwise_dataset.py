from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from radyolo.config.spatial import PatchSize
from radyolo.data import data as data_module
from radyolo.data.data import PatchWiseDataset
from radyolo.data.preprocess_data import preprocess_dataset


def _build_dataset_for_format(
    synthetic_dataset: Path,
    tmp_path: Path,
    image_format: str,
) -> tuple[Path, str]:
    if image_format == "nifti":
        return synthetic_dataset, "nifti"
    return (
        preprocess_dataset(
            dataset_path=synthetic_dataset,
            output_path=tmp_path / "preprocessed_blosc2_dataset",
            workers=0,
        ),
        "blosc2",
    )


def test_patchwise_dataset_extracts_centered_patch_for_all_supported_formats(
    synthetic_dataset: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    expected_classes = torch.tensor([[0.0]], dtype=torch.float32)
    expected_bboxes = torch.tensor([[0.5, 0.5, 0.5, 1.0, 1.0, 1.0]], dtype=torch.float32)
    expected_mask: torch.Tensor | None = None

    for image_format in ("nifti", "blosc2"):
        dataset_root, expected_format = _build_dataset_for_format(
            synthetic_dataset,
            tmp_path / image_format,
            image_format,
        )
        dataset = PatchWiseDataset(
            src_path=dataset_root,
            patch_size=PatchSize(x=16, y=16, z=32),
            augment=False,
        )
        monkeypatch.setattr(dataset, "_sample_patch_center", lambda _shape, _patch_shape: (16, 16, 32))

        sample = dataset[0]

        assert dataset.image_format == expected_format
        assert sample["image"].dtype == torch.float32
        assert sample["image"].shape == (1, 32, 16, 16)
        assert torch.all(sample["image"] == 1)
        assert sample["original_shape"] == (64, 32, 32)
        assert sample["ratio_pad"] == ((1.0, 1.0, 1.0), (-8, -8, -16))
        assert torch.allclose(sample["classes"], expected_classes)
        assert torch.allclose(sample["bboxes"], expected_bboxes)
        assert sample["masks"] is not None
        assert sample["masks"].shape == (1, 32, 16, 16)
        assert torch.all((sample["masks"] == 0) | (sample["masks"] == 1))
        if expected_mask is None:
            expected_mask = sample["masks"].clone()
        else:
            assert torch.equal(sample["masks"], expected_mask)


def test_patchwise_dataset_requests_only_patch_region_from_blosc2(
    synthetic_dataset: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeChunkedArray:
        def __init__(self, dtype: np.dtype) -> None:
            self.shape = (32, 32, 64)
            self.requests: list[tuple[slice, slice, slice]] = []
            self.dtype = dtype

        def __getitem__(self, item: tuple[slice, slice, slice]) -> np.ndarray:
            self.requests.append(item)
            return np.ones(
                tuple(slice_.stop - slice_.start for slice_ in item),
                dtype=self.dtype,
            )

    expected_request = (slice(8, 24), slice(8, 24), slice(16, 48))

    dataset_root, _ = _build_dataset_for_format(
        synthetic_dataset,
        tmp_path / "chunked_blosc2",
        "blosc2",
    )
    dataset = PatchWiseDataset(
        src_path=dataset_root,
        patch_size=PatchSize(x=16, y=16, z=32),
        augment=False,
    )
    fake_image_store = FakeChunkedArray(np.float32)
    fake_mask_store = FakeChunkedArray(np.uint8)
    monkeypatch.setattr(dataset, "_sample_patch_center", lambda _shape, _patch_shape: (16, 16, 32))

    def _open_store(path: str, *_args, **_kwargs):
        normalized_path = path.replace("\\", "/")
        if normalized_path.endswith("images/case1.b2nd"):
            return fake_image_store
        if normalized_path.endswith("masks/case1.b2nd"):
            return fake_mask_store
        raise AssertionError(f"Unexpected blosc2 path requested: {path}")

    monkeypatch.setattr(data_module.blosc2, "open", _open_store)

    sample = dataset[0]

    assert sample["image"].shape == (1, 32, 16, 16)
    assert fake_image_store.requests == [expected_request]
    assert fake_mask_store.requests == [expected_request]
    assert sample["masks"] is not None
    assert sample["masks"].shape == (1, 32, 16, 16)


def test_extract_patch_from_array_uses_array_min_for_image_padding() -> None:
    image = np.arange(1, 9, dtype=np.float32).reshape((2, 2, 2))

    patch = data_module._extract_patch_from_array(
        image,
        patch_origin_xyz=(-1, -1, -1),
        patch_shape_xyz=(3, 3, 3),
    )

    assert patch.shape == (3, 3, 3)
    assert patch.min() == image.min()
    assert patch[0, 0, 0] == image.min()


def test_extract_patch_from_array_uses_zero_for_mask_padding() -> None:
    mask = np.ones((2, 2, 2), dtype=np.uint8)

    patch = data_module._extract_patch_from_array(
        mask,
        patch_origin_xyz=(-1, -1, -1),
        patch_shape_xyz=(3, 3, 3),
        fill_value=0,
    )

    assert patch.shape == (3, 3, 3)
    assert patch[0, 0, 0] == 0
    assert patch[1:, 1:, 1:].min() == 1


def test_load_chunked_blosc2_patch_uses_loaded_patch_min_for_image_padding(monkeypatch) -> None:
    class FakeChunkedArray:
        def __init__(self, dtype: np.dtype) -> None:
            self.shape = (4, 4, 4)
            self.dtype = dtype

        def __getitem__(self, item: tuple[slice, slice, slice]) -> np.ndarray:
            return np.full(
                tuple(slice_.stop - slice_.start for slice_ in item),
                7,
                dtype=self.dtype,
            )

        def min(self) -> float:
            raise AssertionError("store.min() should not be called for image padding")

    monkeypatch.setattr(data_module.blosc2, "open", lambda *_args, **_kwargs: FakeChunkedArray(np.float32))

    patch = data_module._load_chunked_blosc2_patch(
        Path("fake.b2nd"),
        patch_origin_xyz=(-1, -1, -1),
        patch_shape_xyz=(3, 3, 3),
        dtype=np.float32,
    )

    assert patch.shape == (3, 3, 3)
    assert patch[0, 0, 0] == 7.0
    assert patch[1:, 1:, 1:].min() == 7.0


def test_load_chunked_blosc2_patch_uses_zero_for_mask_padding(monkeypatch) -> None:
    class FakeChunkedArray:
        def __init__(self, dtype: np.dtype) -> None:
            self.shape = (4, 4, 4)
            self.dtype = dtype

        def __getitem__(self, item: tuple[slice, slice, slice]) -> np.ndarray:
            return np.ones(
                tuple(slice_.stop - slice_.start for slice_ in item),
                dtype=self.dtype,
            )

        def min(self) -> int:
            return 9

    monkeypatch.setattr(data_module.blosc2, "open", lambda *_args, **_kwargs: FakeChunkedArray(np.uint8))

    patch = data_module._load_chunked_blosc2_patch(
        Path("fake_mask.b2nd"),
        patch_origin_xyz=(-1, -1, -1),
        patch_shape_xyz=(3, 3, 3),
        dtype=np.uint8,
        fill_value=0,
    )

    assert patch.shape == (3, 3, 3)
    assert patch[0, 0, 0] == 0
    assert patch[1:, 1:, 1:].min() == 1


def test_build_dataloader_skips_multiprocessing_context_without_workers(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_dataloader(*_args, **kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(data_module, "DataLoader", fake_dataloader)
    monkeypatch.setattr(data_module.os, "name", "posix", raising=False)
    monkeypatch.setattr(data_module.os, "cpu_count", lambda: 8)

    data_module.build_dataloader(
        dataset=[{"image": torch.zeros(1)}],
        batch_size=1,
        workers=0,
        device="cpu",
    )

    assert captured["num_workers"] == 0
    assert captured["persistent_workers"] is False
    assert captured["multiprocessing_context"] is None


def test_build_dataloader_uses_spawn_on_posix_with_workers(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_dataloader(*_args, **kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(data_module, "DataLoader", fake_dataloader)
    monkeypatch.setattr(data_module.os, "name", "posix", raising=False)
    monkeypatch.setattr(data_module.os, "cpu_count", lambda: 8)

    data_module.build_dataloader(
        dataset=[{"image": torch.zeros(1)}],
        batch_size=1,
        workers=2,
        device="cpu",
    )

    assert captured["num_workers"] == 2
    assert captured["persistent_workers"] is True
    assert captured["multiprocessing_context"] == "spawn"


def test_sample_patch_center_keeps_patch_inside_image_when_patch_fits() -> None:
    patch_shape_xyz = (16, 16, 32)
    image_shape_xyz = (32, 32, 64)

    for _ in range(200):
        center_xyz = data_module._sample_patch_center(image_shape_xyz, patch_shape_xyz)
        origin_xyz = data_module._compute_patch_origin(center_xyz, patch_shape_xyz)

        assert origin_xyz[0] >= 0
        assert origin_xyz[1] >= 0
        assert origin_xyz[2] >= 0
        assert origin_xyz[0] + patch_shape_xyz[0] <= image_shape_xyz[0]
        assert origin_xyz[1] + patch_shape_xyz[1] <= image_shape_xyz[1]
        assert origin_xyz[2] + patch_shape_xyz[2] <= image_shape_xyz[2]


def test_sample_patch_center_allows_overhang_when_requested() -> None:
    crop_shape_xyz = (16, 16, 32)
    patch_shape_xyz = (20, 20, 39)
    lower_overhang_xyz = tuple(
        (patch_dim - crop_dim) // 2
        for patch_dim, crop_dim in zip(patch_shape_xyz, crop_shape_xyz, strict=True)
    )
    upper_overhang_xyz = tuple(
        patch_dim - crop_dim - lower_overhang
        for patch_dim, crop_dim, lower_overhang in zip(
            patch_shape_xyz,
            crop_shape_xyz,
            lower_overhang_xyz,
            strict=True,
        )
    )
    image_shape_xyz = (32, 32, 64)

    saw_negative_origin = False
    saw_upper_overhang = False

    for _ in range(400):
        center_xyz = data_module._sample_patch_center(
            image_shape_xyz,
            patch_shape_xyz,
            crop_shape_xyz=crop_shape_xyz,
            allow_overhang=True,
        )
        origin_xyz = data_module._compute_patch_origin(center_xyz, patch_shape_xyz)

        assert origin_xyz[0] >= -lower_overhang_xyz[0]
        assert origin_xyz[1] >= -lower_overhang_xyz[1]
        assert origin_xyz[2] >= -lower_overhang_xyz[2]
        assert origin_xyz[0] + patch_shape_xyz[0] <= image_shape_xyz[0] + upper_overhang_xyz[0]
        assert origin_xyz[1] + patch_shape_xyz[1] <= image_shape_xyz[1] + upper_overhang_xyz[1]
        assert origin_xyz[2] + patch_shape_xyz[2] <= image_shape_xyz[2] + upper_overhang_xyz[2]

        saw_negative_origin |= any(origin < 0 for origin in origin_xyz)
        saw_upper_overhang |= any(
            origin + patch_dim > image_dim
            for origin, patch_dim, image_dim in zip(
                origin_xyz,
                patch_shape_xyz,
                image_shape_xyz,
                strict=True,
            )
        )

    assert saw_negative_origin
    assert saw_upper_overhang


def test_patchwise_training_load_uses_enlarged_extraction_patch(
    synthetic_dataset: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root, _ = _build_dataset_for_format(
        synthetic_dataset,
        tmp_path / "augmented_extract_blosc2",
        "blosc2",
    )
    dataset = PatchWiseDataset(
        src_path=dataset_root,
        patch_size=PatchSize(x=16, y=16, z=32),
        augment=True,
    )
    monkeypatch.setattr(dataset, "_sample_patch_center", lambda _shape, _patch_shape: (16, 16, 32))

    sample = dataset.load_image_label_and_mask(0)

    assert sample["image"].shape == (1, 39, 20, 20)
    assert sample["shape"] == (39, 20, 20)
    assert sample["ratio_pad"] == ((1.0, 1.0, 1.0), (-6, -6, -13))


def test_patchwise_dataset_uses_larger_extraction_size_only_for_augmented_training(
    synthetic_dataset: Path,
) -> None:
    train_dataset = PatchWiseDataset(
        src_path=synthetic_dataset,
        patch_size=PatchSize(x=16, y=16, z=32),
        augment=True,
    )
    val_dataset = PatchWiseDataset(
        src_path=synthetic_dataset,
        patch_size=PatchSize(x=16, y=16, z=32),
        augment=False,
        group="val",
    )

    assert train_dataset.extraction_patch_size == PatchSize(x=20, y=20, z=39)
    assert val_dataset.extraction_patch_size == PatchSize(x=16, y=16, z=32)


def test_patchwise_dataset_centers_oversized_patch_dimensions(
    synthetic_dataset: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root, _ = _build_dataset_for_format(
        synthetic_dataset,
        tmp_path / "oversized_patch_blosc2",
        "blosc2",
    )
    dataset = PatchWiseDataset(
        src_path=dataset_root,
        patch_size=PatchSize(x=48, y=16, z=32),
        augment=False,
    )
    monkeypatch.setattr(dataset, "_sample_patch_center", lambda _shape, _patch_shape: (16, 16, 32))

    sample = dataset[0]

    assert sample["image"].shape == (1, 32, 16, 48)
    assert sample["ratio_pad"] == ((1.0, 1.0, 1.0), (8, -8, -16))
    assert torch.all(sample["image"][..., :8] == 0)
    assert torch.all(sample["image"][..., 8:16] == 0)
    assert torch.all(sample["image"][..., 16:32] == 1)
    assert torch.all(sample["image"][..., 32:40] == 0)
    assert torch.all(sample["image"][..., 40:] == 0)


def test_patchwise_dataset_length_includes_foreground_oversamples(
    synthetic_dataset: Path,
) -> None:
    dataset = PatchWiseDataset(
        src_path=synthetic_dataset,
        patch_size=PatchSize(x=16, y=16, z=32),
        augment=False,
        foreground_oversampling=0.33,
    )

    assert len(dataset.cases) == 3
    assert dataset.foreground_oversample_count == 2
    assert len(dataset) == 5


def test_patchwise_dataset_oversampled_indices_sample_from_foreground_boxes(
    synthetic_dataset: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "foreground_only_case1"
    shutil.copytree(synthetic_dataset, dataset_root)
    (dataset_root / "data_splits.json").write_text(
        json.dumps([
            {"train": ["case1", "case2", "case3"], "val": ["case4"], "test": ["case5"]}
        ]),
        encoding="utf-8",
    )
    (dataset_root / "labels" / "case2.txt").write_text("", encoding="utf-8")
    (dataset_root / "labels" / "case3.txt").write_text("", encoding="utf-8")

    dataset = PatchWiseDataset(
        src_path=dataset_root,
        patch_size=PatchSize(x=16, y=16, z=32),
        augment=False,
        foreground_oversampling=0.33,
        min_box_crop_fraction=0.0,
    )
    expected_bounds = [(8, 23), (8, 23), (16, 47)]
    sampled_bounds: list[tuple[int, int]] = []

    def _unexpected_uniform_sampling(_image_shape_xyz, _patch_shape_xyz):
        raise AssertionError("Foreground oversampled indices must not use uniform patch sampling")

    def _recording_randint(lower: int, upper: int) -> int:
        sampled_bounds.append((lower, upper))
        return upper

    monkeypatch.setattr(dataset, "_sample_patch_center", _unexpected_uniform_sampling)
    monkeypatch.setattr(data_module.random, "randint", _recording_randint)

    sample = dataset[len(dataset.cases)]

    assert dataset.foreground_cases == ("case1",)
    assert sampled_bounds == expected_bounds
    assert sample["case_name"] == "case1"
    assert sample["classes"].shape == (1, 1)
    assert sample["bboxes"].shape == (1, 6)
    assert sample["masks"] is not None


def test_patchwise_validation_sampling_is_deterministic_per_index(
    synthetic_dataset: Path,
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "deterministic_val_dataset"
    shutil.copytree(synthetic_dataset, dataset_root)
    (dataset_root / "data_splits.json").write_text(
        json.dumps([
            {"train": ["case1"], "val": ["case2", "case3"], "test": ["case4"]}
        ]),
        encoding="utf-8",
    )

    dataset = PatchWiseDataset(
        src_path=dataset_root,
        patch_size=PatchSize(x=16, y=16, z=32),
        augment=False,
        group="val",
        foreground_oversampling=0.5,
        sampling_seed=17,
    )

    repeated_base_samples = [dataset[0], dataset[0]]
    repeated_oversampled_samples = [dataset[len(dataset.cases)], dataset[len(dataset.cases)]]

    for first, second in (
        repeated_base_samples,
        repeated_oversampled_samples,
    ):
        assert first["case_name"] == second["case_name"]
        assert first["ratio_pad"] == second["ratio_pad"]
        assert torch.equal(first["image"], second["image"])
        assert torch.equal(first["classes"], second["classes"])
        assert torch.equal(first["bboxes"], second["bboxes"])
        assert (first["masks"] is None) == (second["masks"] is None)
        if first["masks"] is not None:
            assert second["masks"] is not None
            assert torch.equal(first["masks"], second["masks"])


def test_patchwise_dataset_reads_custom_split_file_name(
    synthetic_dataset: Path,
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "custom_split_name_dataset"
    shutil.copytree(synthetic_dataset, dataset_root)
    custom_split_file = dataset_root / "custom_splits.json"
    (dataset_root / "data_splits.json").replace(custom_split_file)

    dataset = PatchWiseDataset(
        src_path=dataset_root,
        patch_size=PatchSize(x=16, y=16, z=32),
        augment=False,
        split_file_name="custom_splits.json",
    )

    assert dataset.cases == ["case1", "case2", "case3"]


def test_crop_boxes_to_patch_drops_boxes_below_minimum_containment() -> None:
    classes = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    bboxes = torch.tensor(
        [
            [0.25, 0.25, 0.25, 0.2, 0.2, 0.2],
            [0.45, 0.45, 0.45, 0.4, 0.4, 0.4],
        ],
        dtype=torch.float32,
    )

    kept_classes, kept_bboxes = data_module._crop_boxes_to_patch(
        classes,
        bboxes,
        image_shape_xyz=(100, 100, 100),
        patch_origin_xyz=(0, 0, 0),
        patch_shape_xyz=(50, 50, 50),
        min_box_crop_fraction=0.5,
    )

    assert torch.equal(kept_classes, classes[:1])
    assert torch.allclose(
        kept_bboxes,
        torch.tensor([[0.5, 0.5, 0.5, 0.4, 0.4, 0.4]], dtype=torch.float32),
    )


def test_crop_boxes_to_patch_keeps_boxes_above_minimum_containment() -> None:
    classes = torch.tensor([[1.0]], dtype=torch.float32)
    bboxes = torch.tensor([[0.45, 0.45, 0.45, 0.4, 0.4, 0.4]], dtype=torch.float32)

    kept_classes, kept_bboxes = data_module._crop_boxes_to_patch(
        classes,
        bboxes,
        image_shape_xyz=(100, 100, 100),
        patch_origin_xyz=(0, 0, 0),
        patch_shape_xyz=(50, 50, 50),
        min_box_crop_fraction=0.2,
    )

    assert torch.equal(kept_classes, classes)
    assert kept_bboxes.shape == (1, 6)


def test_filter_instance_masks_drops_masks_for_removed_boxes() -> None:
    classes = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    bboxes = torch.tensor(
        [
            [0.25, 0.25, 0.25, 0.2, 0.2, 0.2],
            [0.45, 0.45, 0.45, 0.4, 0.4, 0.4],
        ],
        dtype=torch.float32,
    )
    masks = torch.stack(
        [
            torch.ones((4, 4, 4), dtype=torch.uint8),
            torch.full((4, 4, 4), 2, dtype=torch.uint8),
        ]
    )

    kept_classes, kept_bboxes, keep = data_module._crop_boxes_to_patch_with_keep(
        classes,
        bboxes,
        image_shape_xyz=(100, 100, 100),
        patch_origin_xyz=(0, 0, 0),
        patch_shape_xyz=(50, 50, 50),
        min_box_crop_fraction=0.5,
    )
    kept_masks = data_module._filter_instance_masks(masks, keep)

    assert torch.equal(kept_classes, classes[:1])
    assert torch.allclose(
        kept_bboxes,
        torch.tensor([[0.5, 0.5, 0.5, 0.4, 0.4, 0.4]], dtype=torch.float32),
    )
    assert kept_masks is not None
    assert kept_masks.shape[0] == 1
    assert torch.equal(kept_masks[0], masks[0])


def test_filter_instance_masks_returns_none_when_all_masks_are_removed() -> None:
    keep = torch.tensor([False], dtype=torch.bool)
    masks = torch.ones((1, 4, 4, 4), dtype=torch.uint8)

    kept_masks = data_module._filter_instance_masks(masks, keep)

    assert kept_masks is None