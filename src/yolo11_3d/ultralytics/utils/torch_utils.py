from __future__ import annotations

import logging
from typing import Any

import torch
import torch.optim.optimizer

LOGGER = logging.getLogger(__name__)


def convert_optimizer_state_dict_to_fp16(state_dict: dict[str, Any]) -> dict[str, Any]:  # -> Any:
    """Converts the state_dict of a given optimizer to FP16, focusing on the 'state' key for tensor conversions.

    This method aims to reduce storage size without altering 'param_groups' as they contain non-tensor data.
    """
    for state in state_dict["state"].values():
        for k, v in state.items():
            if k != "step" and isinstance(v, torch.Tensor) and v.dtype is torch.float32:
                state[k] = v.half()

    return state_dict


class EarlyStopping:
    """Early stopping class that stops training when a specified number of epochs have passed without improvement."""

    def __init__(self, patience: int = 50) -> None:
        """Initialize early stopping object.

        Args:
            patience (int, optional): Number of epochs to wait after fitness stops improving before stopping.
        """
        self.best_fitness: float = 0.0  # i.e. mAP
        self.best_epoch: int = 0
        self.patience: int | float = patience or float(
            "inf"
        )  # epochs to wait after fitness stops improving to stop
        self.possible_stopi: bool = False  # possible stop may occur next epoch

    def __call__(self, epoch: int, fitness: float) -> bool:
        """Check whether to stop training.

        Args:
            epoch (int): Current epoch of training
            fitness (float): Fitness value of current epoch

        Returns:
            (bool): True if training should stop, False otherwise
        """
        if fitness is None:  # check if fitness=None (happens when val=False)
            return False

        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            prefix = "EarlyStopping: "
            LOGGER.info(
                "%sTraining stopped early as no improvement observed in last %d epochs. "
                "Best results observed at epoch %d, best model saved as best.pt.\n"
                "To update EarlyStopping(patience=%d) pass a new patience value, "
                "i.e. `patience=300` or use `patience=0` to disable EarlyStopping.",
                prefix,
                self.patience,
                self.best_epoch,
                self.patience,
            )
        return stop
