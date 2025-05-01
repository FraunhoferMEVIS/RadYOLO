from __future__ import annotations

from typing import Any

import torch
import torch.optim.optimizer


def convert_optimizer_state_dict_to_fp16(state_dict: dict[str, Any]) -> dict[str, Any]:  # -> Any:
    """Converts the state_dict of a given optimizer to FP16, focusing on the 'state' key for tensor conversions.

    This method aims to reduce storage size without altering 'param_groups' as they contain non-tensor data.
    """
    for state in state_dict["state"].values():
        for k, v in state.items():
            if k != "step" and isinstance(v, torch.Tensor) and v.dtype is torch.float32:
                state[k] = v.half()

    return state_dict
