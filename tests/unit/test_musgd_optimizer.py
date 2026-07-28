from __future__ import annotations

import pytest
import torch

from radyolo.training.optimizer import (
    MuSGD,
    _is_muon_applicable_parameter,
    _to_moun_matrix,
    build_optimizer,
)


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="Torch Muon optimizer unavailable")
def test_musgd_dummy_training_step_updates_parameters() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    )
    optimizer = build_optimizer(
        model,
        name="MuSGD",
        lr=1e-2,
        momentum=0.9,
        decay=1e-4,
        muon_weight=0.3,
    )

    inputs = torch.randn(16, 4)
    targets = torch.randn(16, 2)
    criterion = torch.nn.MSELoss()

    first_param_before = next(model.parameters()).detach().clone()
    losses: list[float] = []
    for _ in range(3):
        optimizer.zero_grad()
        prediction = model(inputs)
        loss = criterion(prediction, targets)
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()

    first_param_after = next(model.parameters()).detach().clone()
    assert not torch.allclose(first_param_before, first_param_after)
    assert torch.isfinite(torch.tensor(losses)).all()


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="Torch Muon optimizer unavailable")
def test_musgd_state_dict_roundtrip_and_parent_fallback_load() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    optimizer = build_optimizer(model, name="MuSGD", muon_weight=0.5)

    inputs = torch.randn(8, 4)
    targets = torch.randn(8, 2)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()

    state = optimizer.state_dict()

    restored = build_optimizer(model, name="MuSGD", muon_weight=0.5)
    restored.load_state_dict(state)

    assert isinstance(state, dict)
    assert "parent" in state
    assert "sgd" in state
    assert "muon" in state

    # Exercise fallback branch where only the parent Optimizer state dict is provided.
    parent_only = state["parent"]
    assert isinstance(parent_only, dict)
    restored.load_state_dict(parent_only)


def test_musgd_constructor_validates_muon_weight() -> None:
    with pytest.raises(ValueError, match="muon_weight"):
        MuSGD(
            param_groups=[
                {
                    "params": [torch.nn.Parameter(torch.randn(2, 2))],
                    "lr": 1e-3,
                    "momentum": 0.9,
                    "weight_decay": 0.0,
                    "nesterov": True,
                }
            ],
            applicable_params=[],
            muon_weight=1.5,
        )


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="Torch Muon optimizer unavailable")
def test_musgd_step_with_zero_muon_weight_uses_sgd_path() -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = build_optimizer(model, name="MuSGD", muon_weight=0.0)

    x = torch.randn(10, 3)
    y = torch.randn(10, 2)
    weight_before = model.weight.detach().clone()

    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()

    assert not torch.allclose(weight_before, model.weight.detach())


def test_optimizer_helpers_and_builder_paths() -> None:
    matrix = torch.randn(3, 4)
    vector = torch.randn(2, 3, 4)

    assert _to_moun_matrix(matrix).shape == (3, 4)
    assert _to_moun_matrix(vector).shape == (2, 12)

    param = torch.nn.Parameter(torch.randn(4, 4), requires_grad=True)
    assert _is_muon_applicable_parameter("l1.weight", param)
    assert not _is_muon_applicable_parameter(
        "prototype_layer.cv_out_2.weight",
        param,
    )
    assert not _is_muon_applicable_parameter(
        "mask_coefficient_layer.cv.2.weight",
        param,
    )
    assert not _is_muon_applicable_parameter(
        "l23.cv_boxes.conv.2.weight",
        param,
    )
    assert not _is_muon_applicable_parameter(
        "l1.bias",
        torch.nn.Parameter(torch.randn(4), requires_grad=True),
    )

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4), torch.nn.Linear(4, 2))
    assert isinstance(build_optimizer(model, name="SGD"), torch.optim.SGD)
    assert isinstance(build_optimizer(model, name="RMSProp"), torch.optim.RMSprop)
    assert isinstance(build_optimizer(model, name="AdamW"), torch.optim.AdamW)

    with pytest.raises(NotImplementedError, match="Optimizer"):
        build_optimizer(model, name="does-not-exist")
