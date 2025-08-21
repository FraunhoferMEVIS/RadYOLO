# Ultralytics YOLO 🚀, AGPL-3.0 license

from __future__ import annotations

import os
import random
from collections.abc import Generator, Sized
from typing import Any, NoReturn

import numpy as np
import torch
from torch.utils.data import dataloader, sampler

from yolo11_3d.nn.data import NIfTIDataset, Label


class InfiniteDataLoader(dataloader.DataLoader[Label], Sized):
    """Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Dataloader that infinitely recycles workers, inherits from DataLoader."""
        super().__init__(*args, **kwargs)
        # self.batch_sampler = _RepeatSampler(self.batch_sampler)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))  # pyright: ignore[reportArgumentType]
        self.iterator = super().__iter__()

    def __len__(self) -> int:
        """Returns the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue, reportOptionalMemberAccess]

    def __iter__(self) -> Generator[Label]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Creates a sampler that repeats indefinitely."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def reset(self) -> None:
        """Reset iterator.

        This is useful when we want to modify settings of dataset while training.
        """
        self.iterator = self._get_iterator()


class _RepeatSampler:
    """Sampler that repeats forever.

    Args:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler: sampler.BatchSampler) -> None:
        """Initializes an object that repeats a given sampler indefinitely."""
        self.sampler: sampler.BatchSampler = sampler

    def __iter__(self) -> Generator[Any, Any, NoReturn]:
        """Iterates over the 'sampler' and yields its contents."""
        while True:
            yield from iter(self.sampler)


def seed_worker(worker_id) -> None:
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(
    dataset: NIfTIDataset,
    batch: int,
    workers: int,
    shuffle: bool = True,
    rank: int = -1,
    device: torch.device | str = "cpu",
) -> InfiniteDataLoader:
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min((os.cpu_count() or 1) // max(nd, 1), workers)  # number of workers
    sampler = None
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + -1)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=(device.type if isinstance(device, torch.device) else device) != "cpu",
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )
