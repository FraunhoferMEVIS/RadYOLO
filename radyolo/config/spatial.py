from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Sequence


def _validate_triplet(value: Sequence[Any], *, label: str) -> tuple[Any, Any, Any]:
    if len(value) != 3:
        msg = f"{label} requires exactly 3 values, got {len(value)}"
        raise ValueError(msg)
    return value[0], value[1], value[2]


class PatchSize(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: int = Field(gt=0)
    y: int = Field(gt=0)
    z: int = Field(gt=0)

    def get_xyz(self) -> tuple[int, int, int]:
        return (self.x, self.y, self.z)

    def get_zyx(self) -> tuple[int, int, int]:
        return (self.z, self.y, self.x)

    @classmethod
    def from_xyz(cls, value: Sequence[int]) -> Self:
        x, y, z = _validate_triplet(value, label=cls.__name__)
        return cls(x=x, y=y, z=z)

    @classmethod
    def from_zyx(cls, value: Sequence[int]) -> Self:
        z, y, x = _validate_triplet(value, label=cls.__name__)
        return cls(x=x, y=y, z=z)


class VoxelSize(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float = Field(gt=0)
    y: float = Field(gt=0)
    z: float = Field(gt=0)

    def get_xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def get_zyx(self) -> tuple[float, float, float]:
        return (self.z, self.y, self.x)

    @classmethod
    def from_xyz(cls, value: Sequence[float]) -> Self:
        x, y, z = _validate_triplet(value, label=cls.__name__)
        return cls(x=x, y=y, z=z)

    @classmethod
    def from_zyx(cls, value: Sequence[float]) -> Self:
        z, y, x = _validate_triplet(value, label=cls.__name__)
        return cls(x=x, y=y, z=z)
