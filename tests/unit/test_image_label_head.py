from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F  # noqa: N812
import yaml

from radyolo.config.config import get_cfg
from radyolo.data.image_labels import ImageLabelTaskSpec
from radyolo.models.layers import ModelOutput
from radyolo.models.model import build_model_from_config
from radyolo.training.loss import (
    ImageLabelClassificationLoss,
    ImageLabelSurvivalLoss,
    build_detection_loss_from_config,
)


def _write_image_label_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    labels_image_dir = root / "labels_image"
    labels_image_dir.mkdir(exist_ok=True)
    (labels_image_dir / "case1.json").write_text(
        json.dumps(
            {
                "stage": 1,
                "rfs": 12.0,
                "relapse": True,
            },
        ),
        encoding="utf-8",
    )
    (labels_image_dir / "case2.json").write_text(
        json.dumps(
            {
                "stage": 2,
                "rfs": 24.0,
                "relapse": False,
            },
        ),
        encoding="utf-8",
    )
    (root / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "names": {0: "lesion"},
                "image_label_tasks": {
                    "stage": {
                        "type": "multiclass",
                        "classes": {0: "I", 1: "II", 2: "III"},
                    },
                    "rfs": {
                        "type": "survival",
                        "masking": "relapse",
                        "unit": "days",
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return root


def _write_train_config(
    config_path: Path,
    dataset_path: Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> Path:
    config = {
        "input_channels": 1,
        "model_scale": "n",
        "data": dataset_path.as_posix(),
        "fold": 0,
        "epochs": 1,
        "nominal_batch_size": 1,
        "warmup_epochs": 0.0,
        "patience": 1,
        "batch_size": 2,
        "patch_size": {"x": 32, "y": 32, "z": 32},
        "normalization": {"type": "none"},
        "device": "cpu",
        "workers": 0,
        "output_folder_base": (config_path.parent / "runs").as_posix(),
        "experiment_name": "image_label_head",
        "exist_ok": True,
        "optimizer": "Adam",
        "verbose": False,
        "seed": 0,
        "deterministic": True,
        "resume": False,
        "amp": False,
        "use_cudnn_benchmark": False,
        "val": False,
        "plots": False,
        "segmentation_output": False,
        "enable_image_label_head": True,
        "image_label_head_feature_maps": ["o10", "o16", "o19", "o22"],
        "image_label_head_projection_channels": 16,
        "image_label_head_mlp_hidden_layers": 2,
        "image_label_head_mlp_hidden_channels": 24,
        "image_label_head_dropout": 0.1,
        "image_label_survival_num_bins": 8,
    }
    if overrides is not None:
        config.update(overrides)
    config_path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def test_build_model_from_config_emits_image_label_outputs(tmp_path: Path) -> None:
    dataset_path = _write_image_label_dataset(tmp_path / "dataset")
    config_path = _write_train_config(tmp_path / "train.yaml", dataset_path)
    config = get_cfg(config_path, {})

    model = build_model_from_config(config)
    model.eval()

    with torch.no_grad():
        output = model(torch.zeros((2, 1, 32, 32, 32), dtype=torch.float32))

    assert output.image_label_outputs is not None
    assert set(output.image_label_outputs) == {"stage", "rfs"}
    assert output.image_label_outputs["stage"].shape == (2, 3)
    assert output.image_label_outputs["rfs"].shape == (2, 8)
    assert model.image_label_task_specs["rfs"].survival_bin_thresholds is not None
    assert torch.allclose(
        torch.tensor(model.image_label_task_specs["rfs"].survival_bin_thresholds),
        torch.linspace(0.0, 24.0, steps=8),
    )


def test_detection_loss_includes_image_label_loss_term(tmp_path: Path) -> None:
    dataset_path = _write_image_label_dataset(tmp_path / "dataset")
    config_path = _write_train_config(tmp_path / "train.yaml", dataset_path)
    config = get_cfg(config_path, {})

    model = build_model_from_config(config)
    loss_fn = build_detection_loss_from_config(config, model, torch.device("cpu"))
    images = torch.randn((2, 1, 32, 32, 32), dtype=torch.float32)
    model_output = model(images)
    batch = {
        "img": images,
        "ori_shape": ((32, 32, 32), (32, 32, 32)),
        "ratio_pad": (None, None),
        "shape": ((32, 32, 32), (32, 32, 32)),
        "cls": torch.zeros((0,), dtype=torch.float32),
        "bboxes": torch.zeros((0, 6), dtype=torch.float32),
        "batch_idx": torch.zeros((0,), dtype=torch.int64),
        "masks": (None, None),
        "case_names": ("case1", "case2"),
        "image_labels": (
            {"stage": 1, "rfs": 12.0, "relapse": True},
            {"stage": 2, "rfs": 24.0, "relapse": False},
        ),
    }

    total_loss, loss_items = loss_fn(model_output, batch, current_epoch=1)

    assert torch.isfinite(total_loss)
    assert torch.isfinite(loss_items).all()
    assert loss_fn.loss_names[-3:] == ["img_cce", "img_cst", "img_cox"]
    assert loss_items.shape == (8,)
    assert float(loss_items[-3]) > 0.0
    assert float(loss_items[-2]) > 0.0
    assert float(loss_items[-1]) > 0.0


def test_detection_loss_skips_zero_gain_image_label_losses(tmp_path: Path) -> None:
    dataset_path = _write_image_label_dataset(tmp_path / "dataset")
    config_path = _write_train_config(
        tmp_path / "train.yaml",
        dataset_path,
        overrides={
            "image_label_multiclass_cce_gain": 0.0,
            "image_label_survival_mtlr_gain": 0.0,
            "image_label_survival_cox_gain": 0.0,
        },
    )
    config = get_cfg(config_path, {})

    model = build_model_from_config(config)
    loss_fn = build_detection_loss_from_config(config, model, torch.device("cpu"))

    assert "img_cce" not in loss_fn.loss_names
    assert "img_cst" not in loss_fn.loss_names
    assert "img_cox" not in loss_fn.loss_names


def test_detection_loss_keeps_mask_branch_graph_connected_without_masks(
    minimal_train_config: Path,
) -> None:
    config = get_cfg(minimal_train_config, {})
    model = build_model_from_config(config)
    loss_fn = build_detection_loss_from_config(config, model, torch.device("cpu"))
    images = torch.randn((1, 1, 32, 32, 64), dtype=torch.float32)
    model_output = model(images)
    batch = {
        "img": images,
        "ori_shape": ((32, 32, 64),),
        "ratio_pad": (None,),
        "shape": ((32, 32, 64),),
        "cls": torch.zeros((0,), dtype=torch.float32),
        "bboxes": torch.zeros((0, 6), dtype=torch.float32),
        "batch_idx": torch.zeros((0,), dtype=torch.int64),
        "masks": (None,),
        "case_names": ("case1",),
    }

    total_loss, loss_items = loss_fn(model_output, batch, current_epoch=1)
    total_loss.backward()

    prototype_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "prototype_layer" in name and parameter.requires_grad
    ]
    coefficient_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "mask_coefficient_layer" in name and parameter.requires_grad
    ]

    assert torch.isfinite(total_loss)
    assert torch.isfinite(loss_items).all()
    assert prototype_grads
    assert coefficient_grads
    assert all(grad is not None for grad in prototype_grads)
    assert all(grad is not None for grad in coefficient_grads)
    assert all(torch.count_nonzero(grad) == 0 for grad in prototype_grads if grad is not None)
    assert all(torch.count_nonzero(grad) == 0 for grad in coefficient_grads if grad is not None)


def test_image_label_multiclass_loss_supports_class_weights() -> None:
    loss_fn = ImageLabelClassificationLoss(
        device=torch.device("cpu"),
        image_label_tasks={
            "stage": ImageLabelTaskSpec(
                type="multiclass",
                classes={0: "I", 1: "II", 2: "III"},
            )
        },
        image_label_multiclass_cce_class_weights={"stage": [1.0, 1.0, 4.0]},
    )
    logits = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [1.0, 0.5, -1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    batch = {
        "image_labels": (
            {"stage": 0},
            {"stage": 0},
            {"stage": 2},
        )
    }

    total_loss, loss_items = loss_fn(ModelOutput(image_label_outputs={"stage": logits}), batch)
    expected = F.cross_entropy(
        logits,
        torch.tensor([0, 0, 2], dtype=torch.long),
        weight=torch.tensor([1.0, 1.0, 4.0], dtype=torch.float32),
        reduction="mean",
    )

    assert loss_fn.loss_names == ["img_cce"]
    assert torch.isclose(total_loss, expected)
    assert torch.isclose(loss_items[0], expected)
def test_image_label_survival_losses_support_censoring_and_cox() -> None:
    loss_fn = ImageLabelSurvivalLoss(
        device=torch.device("cpu"),
        image_label_tasks={
            "rfs": ImageLabelTaskSpec(
                type="survival",
                masking="relapse",
                survival_bin_thresholds=(0.0, 1.0, 2.0),
            )
        },
        image_label_survival_mtlr_gain=1.0,
        image_label_survival_cox_gain=1.0,
    )
    logits = torch.tensor(
        [
            [2.0, 1.0, -1.0],
            [1.5, 0.5, -2.0],
            [0.5, 0.0, -0.5],
        ],
        dtype=torch.float32,
    )
    batch = {
        "image_labels": (
            {"rfs": 1.5, "relapse": True},
            {"rfs": 1.5, "relapse": False},
            {"rfs": 2.5, "relapse": True},
        )
    }

    total_loss, loss_items = loss_fn(ModelOutput(image_label_outputs={"rfs": logits}), batch)

    targets = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    supervision_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    expected_cst = (
        (F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * supervision_mask)
        .sum(dim=1)
        .div(supervision_mask.sum(dim=1))
        .mean()
    )
    risk_scores = -logits.sigmoid().sum(dim=1)
    survival_times = torch.tensor([1.5, 1.5, 2.5], dtype=torch.float32)
    events = torch.tensor([True, False, True])
    sorted_indices = torch.argsort(survival_times, descending=True)
    expected_cox = -(
        risk_scores[sorted_indices][events[sorted_indices]]
        - torch.logcumsumexp(risk_scores[sorted_indices], dim=0)[events[sorted_indices]]
    ).mean()

    assert loss_fn.loss_names == ["img_cst", "img_cox"]
    assert torch.isclose(loss_items[0], expected_cst * 0.5)
    assert torch.isclose(loss_items[1], expected_cox * 0.5)
    assert torch.isclose(total_loss, (expected_cst + expected_cox) * 0.5)