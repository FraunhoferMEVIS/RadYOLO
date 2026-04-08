from __future__ import annotations

from pathlib import Path

from radyolo.config.config import DefaultConfig
from radyolo.config.spatial import PatchSize, VoxelSize
from radyolo.data.preprocessing import resolve_target_shape


def _make_minimal_config(tmp_path: Path, patch_size: object) -> dict[str, object]:
    return {
        "input_channels": 1,
        "num_classes": 1,
        "model_scale": "n",
        "data": str(tmp_path),
        "fold": 0,
        "batch_size": 1,
        "patch_size": patch_size,
        "experiment_name": "spatial-test",
    }


def test_patch_size_and_voxel_size_accessors_use_explicit_axis_orders() -> None:
    patch_size = PatchSize(x=12, y=14, z=16)
    voxel_size = VoxelSize(x=0.8, y=1.2, z=2.5)

    assert patch_size.get_xyz() == (12, 14, 16)
    assert patch_size.get_zyx() == (16, 14, 12)
    assert voxel_size.get_xyz() == (0.8, 1.2, 2.5)
    assert voxel_size.get_zyx() == (2.5, 1.2, 0.8)


def test_default_config_accepts_named_patch_size_objects(tmp_path: Path) -> None:
    cfg = DefaultConfig.model_validate(_make_minimal_config(tmp_path, {"x": 20, "y": 18, "z": 16}))

    assert cfg.patch_size.get_xyz() == (20, 18, 16)
    assert cfg.patch_size.get_zyx() == (16, 18, 20)


def test_default_config_requires_named_patch_size_objects(tmp_path: Path) -> None:
    cfg = DefaultConfig.model_validate(_make_minimal_config(tmp_path, PatchSize(x=16, y=18, z=20)))

    assert cfg.patch_size.get_xyz() == (16, 18, 20)
    assert cfg.patch_size.get_zyx() == (20, 18, 16)


def test_default_config_accepts_foreground_oversampling(tmp_path: Path) -> None:
    cfg = DefaultConfig.model_validate(
        {
            **_make_minimal_config(tmp_path, {"x": 20, "y": 18, "z": 16}),
            "foreground_oversampling": 0.25,
        }
    )

    assert cfg.foreground_oversampling == 0.25


def test_default_config_accepts_patchwise_training_mode(tmp_path: Path) -> None:
    cfg = DefaultConfig.model_validate(
        {
            **_make_minimal_config(tmp_path, {"x": 20, "y": 18, "z": 16}),
            "training_mode": "patchwise",
        }
    )

    assert cfg.training_mode == "patchwise"


def test_default_config_accepts_min_box_crop_fraction(tmp_path: Path) -> None:
    cfg = DefaultConfig.model_validate(
        {
            **_make_minimal_config(tmp_path, {"x": 20, "y": 18, "z": 16}),
            "min_box_crop_fraction": 0.75,
        }
    )

    assert cfg.min_box_crop_fraction == 0.75


def test_resolve_target_shape_uses_voxel_sizes_through_named_axes() -> None:
    target_shape = resolve_target_shape(
        (6, 5, 4),
        input_voxel_size=VoxelSize(x=2.0, y=3.0, z=4.0),
        target_voxel_size=VoxelSize(x=2.0, y=1.5, z=1.0),
    )

    assert target_shape == (24, 10, 4)
