from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import optim

if TYPE_CHECKING:
    from collections.abc import Callable


def _to_moun_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten non-matrix tensors to 2D while preserving the first dimension."""
    if tensor.ndim == 2:
        return tensor
    return tensor.reshape(tensor.shape[0], -1)


def _is_muon_applicable_parameter(fullname: str, param: torch.nn.Parameter) -> bool:
    """Exclude input and output layers from muon optimizer."""
    if param.ndim < 2 or not param.requires_grad:
        return False
    if fullname.startswith(
        (
            "l0.",
            "image_label_head.task_heads.",
            "prototype_layer.reshape_full.",
            "prototype_layer.reshape_full_1.",
            "prototype_layer.cv_final_2.",
            "prototype_layer.cv_final.",
            "prototype_layer.cv_out_2.",
        )
    ):
        return False

    parts = fullname.split(".")
    if len(parts) >= 3 and parts[0] == "mask_coefficient_layer" and parts[2] == "2":
        return False
    if (  # noqa: SIM103
        len(parts) >= 4
        and parts[0] == "l23"
        and parts[1] in {"cv_boxes", "cv_classes", "cv_obj"}
        and parts[3] == "2"
    ):
        return False

    return True


class MuSGD(optim.Optimizer):
    """Weighted mixture of Muon and SGD for selected parameters."""

    def __init__(
        self,
        param_groups: list[dict[str, Any]],
        applicable_params: list[torch.nn.Parameter],
        muon_weight: float,
    ) -> None:
        """Initialize mixed Muon/SGD optimizers over shared parameter groups."""
        if not 0.0 <= muon_weight <= 1.0:
            msg = f"muon_weight must be in [0, 1], got {muon_weight}."
            raise ValueError(msg)

        super().__init__(param_groups, {})
        self.muon_weight = muon_weight

        sgd_param_groups = [
            {
                "params": group["params"],
                "lr": group["lr"],
                "momentum": group["momentum"],
                "weight_decay": group["weight_decay"],
                "nesterov": group["nesterov"],
            }
            for group in self.param_groups
        ]
        self._sgd_optimizer = torch.optim.SGD(sgd_param_groups)

        self._applicable_by_id: dict[int, torch.nn.Parameter] = {
            id(param): param for param in applicable_params
        }
        self._flat_params_by_id: dict[int, torch.nn.Parameter] = {}
        self._muon_group_indices: list[int] = []

        muon_param_groups = []
        for index, group in enumerate(self.param_groups):
            muon_group_params = []
            for param in group["params"]:
                param_id = id(param)
                if param_id not in self._applicable_by_id:
                    continue
                flat_parameter = torch.nn.Parameter(_to_moun_matrix(param.detach()).clone())
                self._flat_params_by_id[param_id] = flat_parameter
                muon_group_params.append(flat_parameter)

            if not muon_group_params:
                continue

            self._muon_group_indices.append(index)
            muon_param_groups.append(
                {
                    "params": muon_group_params,
                    "lr": group["lr"],
                    "momentum": group["momentum"],
                    "weight_decay": group["weight_decay"],
                    "nesterov": group["nesterov"],
                }
            )

        self._muon_optimizer = torch.optim.Muon(muon_param_groups) if muon_param_groups else None

    def _sync_child_optimizers(self) -> None:
        """Propagate parent group hyperparameters to child optimizers."""
        for parent_group, sgd_group in zip(
            self.param_groups, self._sgd_optimizer.param_groups, strict=True
        ):
            for key in ("lr", "momentum", "weight_decay", "nesterov"):
                sgd_group[key] = parent_group[key]

        if self._muon_optimizer is None:
            return

        for parent_index, muon_group in zip(
            self._muon_group_indices, self._muon_optimizer.param_groups, strict=True
        ):
            parent_group = self.param_groups[parent_index]
            for key in ("lr", "momentum", "weight_decay", "nesterov"):
                muon_group[key] = parent_group[key]

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        """Run SGD and Muon updates, then blend applicable parameter deltas."""
        self._sync_child_optimizers()

        old_values = {
            param_id: param.detach().clone() for param_id, param in self._applicable_by_id.items()
        }

        for param_id, param in self._applicable_by_id.items():
            flat_param = self._flat_params_by_id[param_id]
            flat_param.data.copy_(_to_moun_matrix(old_values[param_id]))
            if param.grad is None:
                flat_param.grad = None
            else:
                flat_param.grad = _to_moun_matrix(param.grad.detach()).clone()

        sgd_loss = self._sgd_optimizer.step(closure)

        if self._muon_optimizer is None or self.muon_weight == 0.0:
            return sgd_loss

        muon_loss = self._muon_optimizer.step(None)

        for param_id, param in self._applicable_by_id.items():
            old_param = old_values[param_id]
            sgd_delta = param.detach() - old_param
            flat_param = self._flat_params_by_id[param_id]
            muon_delta = (flat_param.detach() - _to_moun_matrix(old_param)).view_as(param)
            param.copy_(
                old_param + (1.0 - self.muon_weight) * sgd_delta + self.muon_weight * muon_delta
            )

        return muon_loss if muon_loss is not None else sgd_loss

    def state_dict(self) -> dict[str, object]:
        """Return combined state for parent, SGD child, and optional Muon child."""
        return {
            "parent": super().state_dict(),
            "sgd": self._sgd_optimizer.state_dict(),
            "muon": None if self._muon_optimizer is None else self._muon_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore optimizer state and resynchronize child optimizer settings."""
        parent_state = state_dict.get("parent")
        if parent_state is None:
            super().load_state_dict(state_dict)
            self._sync_child_optimizers()
            return

        super().load_state_dict(parent_state)
        self._sgd_optimizer.load_state_dict(state_dict["sgd"])
        if self._muon_optimizer is not None and state_dict["muon"] is not None:
            self._muon_optimizer.load_state_dict(state_dict["muon"])
        self._sync_child_optimizers()


def build_optimizer(
    model: torch.nn.Module,
    name: str = "auto",
    lr: float = 0.0001,
    momentum: float = 0.9,
    decay: float = 1e-5,
    muon_weight: float = 0.5,
) -> torch.optim.Optimizer:
    """Create the configured optimizer and parameter groups for the model."""
    weights_with_decay, weights_without_decay, bias_weights = [], [], []
    muon_candidates: list[torch.nn.Parameter] = []
    available_normalization_layers = tuple(
        value for key, value in torch.nn.__dict__.items() if "Norm" in key
    )
    name_ = name

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fullname = f"{module_name}.{param_name}" if module_name else param_name
            if "bias" in fullname:
                bias_weights.append(param)
            elif isinstance(module, available_normalization_layers):
                weights_without_decay.append(param)
            else:
                weights_with_decay.append(param)
                if _is_muon_applicable_parameter(fullname, param):
                    muon_candidates.append(param)

    optimizers = {
        "Adam",
        "Adamax",
        "AdamW",
        "MuSGD",
        "NAdam",
        "RAdam",
        "RMSProp",
        "SGD",
        "auto",
    }
    name_ = {optimizer_name.lower(): optimizer_name for optimizer_name in optimizers}.get(
        name_.lower()
    )
    if name_ in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
        optimizer = getattr(torch.optim, name, torch.optim.Adam)(
            bias_weights, lr=lr, betas=(momentum, 0.999), weight_decay=0.0
        )
    elif name_ == "RMSProp":
        optimizer = torch.optim.RMSprop(bias_weights, lr=lr, momentum=momentum)
    elif name_ == "SGD":
        optimizer = torch.optim.SGD(bias_weights, lr=lr, momentum=momentum, nesterov=True)
    elif name_ == "MuSGD":
        return MuSGD(
            param_groups=[
                {
                    "params": bias_weights,
                    "lr": lr,
                    "momentum": momentum,
                    "weight_decay": 0.0,
                    "nesterov": True,
                },
                {
                    "params": weights_with_decay,
                    "lr": lr,
                    "momentum": momentum,
                    "weight_decay": decay,
                    "nesterov": True,
                },
                {
                    "params": weights_without_decay,
                    "lr": lr,
                    "momentum": momentum,
                    "weight_decay": 0.0,
                    "nesterov": True,
                },
            ],
            applicable_params=muon_candidates,
            muon_weight=muon_weight,
        )
    else:
        msg = f"Optimizer '{name_}' not found in list of available optimizers {optimizers}."
        raise NotImplementedError(msg)

    optimizer.add_param_group({"params": weights_with_decay, "weight_decay": decay})
    optimizer.add_param_group({"params": weights_without_decay, "weight_decay": 0.0})
    return optimizer
