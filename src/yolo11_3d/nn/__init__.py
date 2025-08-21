from __future__ import annotations

from .model import BaseModel, HardcodedModel

MODELS: dict[str, tuple[type[BaseModel], int]] = {
    "3d": (HardcodedModel, 3),
}

