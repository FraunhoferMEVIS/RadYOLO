from __future__ import annotations

import torch


def build_optimizer(
    model: torch.nn.Module,
    name: str = "auto",
    lr: float = 0.0001,
    momentum: float = 0.9,
    decay: float = 1e-5,
) -> torch.optim.Optimizer:
    weights_with_decay, weights_without_decay, bias_weights = [], [], []
    available_normalization_layers = tuple(
        v for k, v in torch.nn.__dict__.items() if "Norm" in k
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

    optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
    name_ = {x.lower(): x for x in optimizers}.get(name_.lower())
    if name_ in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
        optimizer = getattr(torch.optim, name, torch.optim.Adam)(
            bias_weights, lr=lr, betas=(momentum, 0.999), weight_decay=0.0
        )
    elif name_ == "RMSProp":
        optimizer = torch.optim.RMSprop(bias_weights, lr=lr, momentum=momentum)
    elif name_ == "SGD":
        optimizer = torch.optim.SGD(bias_weights, lr=lr, momentum=momentum, nesterov=True)
    else:
        msg = f"Optimizer '{name_}' not found in list of available optimizers {optimizers}."
        raise NotImplementedError(msg)

    optimizer.add_param_group({"params": weights_with_decay, "weight_decay": decay})
    optimizer.add_param_group({"params": weights_without_decay, "weight_decay": 0.0})
    return optimizer
