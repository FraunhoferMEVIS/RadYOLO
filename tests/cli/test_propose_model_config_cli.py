from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml
from click.testing import CliRunner

from radyolo import main


def _write_nifti(path: Path, array: np.ndarray) -> None:
    image = nib.Nifti1Image(array.astype(np.float32), affine=np.eye(4))
    nib.save(image, path)


def test_propose_model_config_cli_autoruns_analysis_and_proposes_ct_config(
    synthetic_dataset,
    tmp_path,
) -> None:
    (synthetic_dataset / "dataset.yaml").write_text(
        "nc: 1\nnames: {0: lesion}\nnormalization: {0: CT}\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "proposed_ct.yaml"
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "propose-model-config",
            str(synthetic_dataset),
            "--model-scale",
            "s",
            "--output",
            str(output_path),
            "--workers",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert (synthetic_dataset / "dataset_fingerprint.json").exists()
    with output_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    assert config["input_channels"] == 1
    assert "num_classes" not in config
    assert config["model_scale"] == "s"
    assert config["segmentation_output"] is True
    assert config["normalization"]["type"] == "ct"
    assert config["training_mode"] == "wholeimage"
    assert config["voxel_size"] is None
    assert config["patch_size"]["x"] % 32 == 0
    assert config["patch_size"]["y"] % 32 == 0
    assert config["patch_size"]["z"] % 32 == 0
    assert "Selected training mode: wholeimage because the median image fits" in result.output


def test_propose_model_config_cli_uses_global_zscore_for_all_mrglobal(tmp_path) -> None:
    dataset_path = tmp_path / "mrglobal_dataset"
    (dataset_path / "images").mkdir(parents=True)
    (dataset_path / "labels").mkdir()

    image = np.zeros((24, 24, 32, 2), dtype=np.float32)
    image[..., 0] = 2.0
    image[..., 1] = 5.0
    image[4:20, 4:20, 8:24, :] += 1.0
    _write_nifti(dataset_path / "images" / "case1.nii.gz", image)
    (dataset_path / "labels" / "case1.txt").write_text(
        "0 0.5 0.5 0.5 0.5 0.5 0.5\n",
        encoding="utf-8",
    )
    (dataset_path / "dataset.yaml").write_text(
        "nc: 1\nnames: {0: lesion}\nnormalization: {0: MRGlobal, 1: MRGlobal}\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "proposed_mrglobal.yaml"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "propose-model-config",
            str(dataset_path),
            "--model-scale",
            "n",
            "--output",
            str(output_path),
            "--workers",
            "0",
        ],
    )

    assert result.exit_code == 0
    with output_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    assert config["input_channels"] == 2
    assert config["segmentation_output"] is False
    assert config["normalization"] == {"type": "zscore"}
    assert config["channel_dropout_probabilities"] == [0.0, 0.0]


def test_propose_model_config_cli_automatic_patchwise_sets_voxel_size_and_prints_message(
    tmp_path,
) -> None:
    dataset_path = tmp_path / "patchwise_dataset"
    dataset_path.mkdir()
    (dataset_path / "dataset.yaml").write_text(
        "nc: 1\nnames: {0: lesion}\nnormalization: {0: CT}\n",
        encoding="utf-8",
    )
    (dataset_path / "dataset_fingerprint.json").write_text(
        json.dumps(
            {
                "median_resampled_imagesize": [512, 320, 192],
                "median_voxel_size_mm": [1.5, 1.0, 2.0],
                "foreground_channel_stats": [
                    {"mean": 10.0, "std": 2.5, "p00_5": -100.0, "p99_5": 200.0}
                ],
                "n_images_with_masks": 0,
            }
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "proposed_patchwise.yaml"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "propose-model-config",
            str(dataset_path),
            "--model-scale",
            "s",
            "--output",
            str(output_path),
            "--training-mode",
            "automatic",
        ],
    )

    assert result.exit_code == 0
    with output_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    assert config["training_mode"] == "patchwise"
    assert config["voxel_size"] == {"x": 1.5, "y": 1.0, "z": 2.0}
    assert "Selected training mode: patchwise because the median image exceeds" in result.output
