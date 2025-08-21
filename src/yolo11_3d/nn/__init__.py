from __future__ import annotations

from .data import NIfTIDataset
from .model import BaseModel, HardcodedModel

DATASETS: dict[str, tuple[type[NIfTIDataset], int]] = {
    "nifti": (NIfTIDataset, 3),
    # "2d-dummy": (DummyDataset, 2),
    # "3d-dummy": (DummyDataset, 3),
}

MODELS: dict[str, tuple[type[BaseModel], int]] = {
    "3d": (HardcodedModel, 3),
}

