from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)


class EarlyStopping:
    """Stops training when a specified number of epochs have passed without improvement."""

    def __init__(self, patience: int = 50) -> None:
        """Initialize early stopping object.

        Args:
            patience (int, optional): Number of epochs to wait after fitness stops improving
                before stopping.
        """
        self.best_fitness: float = 0.0
        self.best_epoch: int = 0
        self.patience: int | float = patience or float("inf")

    def __call__(self, epoch: int, fitness: float) -> bool:
        """Check whether to stop training.

        Args:
            epoch (int): Current epoch of training
            fitness (float): Fitness value of current epoch

        Returns:
            (bool): True if training should stop, False otherwise
        """
        if fitness is None:
            return False

        if fitness >= self.best_fitness:
            self.best_epoch = epoch
            self.best_fitness = fitness
        epochs_without_improvement = epoch - self.best_epoch
        stop = epochs_without_improvement >= self.patience
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
