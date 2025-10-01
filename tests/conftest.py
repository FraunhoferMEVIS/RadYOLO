from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import yaml

# Import package to ensure entry point is importable during tests
import yolo11_3d  # noqa: F401


def _write_nifti(path: Path, array: np.ndarray) -> None:
    img = nib.Nifti1Image(array.astype(np.float32), affine=np.eye(4))
    nib.save(img, str(path))


def _write_mask(path: Path, array: np.ndarray) -> None:
    img = nib.Nifti1Image(array.astype(np.uint8), affine=np.eye(4))
    nib.save(img, str(path))


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a small synthetic 3D dataset with a single class and predefined splits.

    Layout:
      dataset/
        dataset.yaml
        data_splits.json (single fold)
        images/*.nii.gz
        labels/*.txt
        masks/*.nii.gz
    """
    root = tmp_path_factory.mktemp("synthetic_dataset")
    for d in ("images", "labels", "masks"):
        (root / d).mkdir()

    vol_shape = (64, 64, 64)
    cases = ["case1", "case2", "case3", "case4", "case5"]
    for case in cases:
        vol = np.zeros(vol_shape, dtype=np.float32)
        vol[8:24, 8:24, 8:24] = 1.0
        _write_nifti(root / "images" / f"{case}.nii.gz", vol)
        (root / "labels" / f"{case}.txt").write_text(
            "0 0.5 0.5 0.5 0.5 0.5 0.5\n", encoding="utf-8"
        )
        _write_mask(root / "masks" / f"{case}.nii.gz", (vol > 0).astype(np.uint8))

    (root / "dataset.yaml").write_text(
        "nc: 1\nnames: {0: lesion}\n", encoding="utf-8"
    )
    splits = [{"train": cases[:3], "val": [cases[3]], "test": [cases[4]]}]
    (root / "data_splits.json").write_text(json.dumps(splits), encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def minimal_train_config(synthetic_dataset: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a minimal training config file for fast CPU test.

    Only essential fields are specified; all others rely on defaults.
    """
    cfg_dir = tmp_path_factory.mktemp("configs")
    cfg_path = cfg_dir / "train.yaml"
    project_root = tmp_path_factory.mktemp("runs_root") / "runs"
    project_root.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        (
            f"input_channels: 1\n"
            f"model_scale: n\n"
            f"data: {synthetic_dataset.as_posix()}\n"
            f"fold: 0\n"
            f"epochs: 2\n"
            f"nbs: 1\n"
            f"warmup_epochs: 0.5\n"
            f"patience: 1\n"
            f"batch_size: 1\n"
            f"imgsz: [64, 64, 64]\n"
            f"normalization: {{type: 'none'}}\n"
            f"device: cpu\n"
            f"workers: 0\n"
            f"project: {project_root.as_posix()}\n"
            f"name: e2e\n"
            f"exist_ok: True\n"
            f"optimizer: Adam\n"
            f"verbose: False\n"
            f"seed: 0\n"
            f"deterministic: True\n"
            f"resume: False\n"
            f"amp: False\n"
            f"use_cudnn_benchmark: False\n"
            f"val: True\n"
            f"plots: False\n"
            f"augment: False\n"
        ),
        encoding="utf-8",
    )
    return cfg_path


@pytest.fixture(scope="session")
def cli_executable() -> list[str]:
    # Use module invocation for consistency
    return [sys.executable, "-m", "yolo11_3d"]


def run_cli(
    args: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    # We intentionally do not use check=True to capture stdout/stderr for assertions.
    return subprocess.run(  # noqa: S603
        args,
        cwd=cwd,
        env=full_env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )

@pytest.fixture
def trained_run(minimal_train_config: Path, cli_executable: list[str]) -> Path:
    """Train a minimal model and return the run directory.

    Fails fast if training does not produce both last.pt and best.pt.
    """
    cfg = minimal_train_config
    command = [*cli_executable, "train", str(cfg)]
    proc = run_cli(command)
    if proc.returncode != 0:
        msg = (
            "Training failed.\n"
            f"STDERR:\n{proc.stderr}\n"
            f"STDOUT:\n{proc.stdout}"
        )
        raise RuntimeError(msg)
    with cfg.open() as f:
        data = yaml.safe_load(f)
    run_dir = Path(data["project"]) / data["name"]
    weights_dir = run_dir / "weights"
    if not (weights_dir / "last.pt").exists():
        msg_last = "last.pt missing after training"
        raise FileNotFoundError(msg_last)
    if not (weights_dir / "best.pt").exists():
        msg_best = "best.pt missing after training"
        raise FileNotFoundError(msg_best)
    return run_dir
