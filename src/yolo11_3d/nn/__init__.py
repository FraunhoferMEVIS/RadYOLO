from __future__ import annotations

from .data import BaseDataset, MIPDataset, NIfTIDataset
from .model import BaseModel, Hardcoded3DModel, HardcodedModel

DATASETS: dict[str, tuple[type[BaseDataset], int]] = {
    "nifti": (NIfTIDataset, 3),
    "mip": (MIPDataset, 2),
    # "2d-dummy": (DummyDataset, 2),
    # "3d-dummy": (DummyDataset, 3),
}

MODELS: dict[str, tuple[type[BaseModel], int]] = {
    "2d": (HardcodedModel, 2),
    "3d": (Hardcoded3DModel, 3),
}

