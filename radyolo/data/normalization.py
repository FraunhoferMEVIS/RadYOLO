from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

import torch


class NormalizationConfig(TypedDict):
    type: Literal["none", "zscore", "ct", "255_to_01"]
    mean: NotRequired[float]
    std: NotRequired[float]
    clip_min: NotRequired[float]
    clip_max: NotRequired[float]


def normalize_image(
    image: torch.Tensor, normalization_cfg: NormalizationConfig | list[NormalizationConfig]
) -> torch.Tensor:
    """Apply per-channel normalization according to normalization_cfg.

    Supports:
    - NormalizationConfig: {type: "ct"|"zScore"|"255_to_01"|"none", ...params}
    - list[NormalizationConfig]: list of NormalizationConfig for each input channel
    """
    if image.ndim != 4:
        return image
    channels = image.shape[0]

    eps = 1e-6
    if isinstance(normalization_cfg, list):
        if len(normalization_cfg) != channels:
            msg = (
                f"Normalization config length ({len(normalization_cfg)}) does not "
                f"match number of channels ({channels})."
            )
            raise ValueError(msg)
        for c in range(channels):
            entry = normalization_cfg[c]
            if not isinstance(entry, dict) or "type" not in entry:
                msg = "Invalid normalization entry; expected dict with 'type'"
                raise ValueError(msg)

            normalization_scheme = str(entry.get("type", "none"))

            x = image[c]
            if normalization_scheme == "none":
                pass
            elif normalization_scheme == "255_to_01":
                x = x.div(255.0).clamp_(0.0, 1.0)
                image[c] = x
            elif normalization_scheme == "zscore":
                mean = torch.mean(x)
                std = torch.std(x, unbiased=False).clamp_min(eps)
                image[c] = (x - mean) / std
            elif normalization_scheme == "ct":
                clip_min = float(entry.get("clip_min", -1024.0))
                clip_max = float(entry.get("clip_max", 3071.0))
                mean = float(entry.get("mean", 0.0))
                std = float(entry.get("std", 1.0))
                std = max(std, eps)
                x = x.clamp(clip_min, clip_max)
                image[c] = (x - mean) / std
            else:
                msg = f"Unknown normalization scheme '{normalization_scheme}' for channel {c}"
                raise NotImplementedError(msg)
    elif isinstance(normalization_cfg, dict):
        normalization_scheme = normalization_cfg["type"]
        if normalization_scheme == "none":
            pass
        elif normalization_scheme == "255_to_01":
            image = image.div(255.0).clamp_(0.0, 1.0)
        elif normalization_scheme == "zscore":
            mean = torch.mean(image)
            std = torch.std(image, unbiased=False).clamp_min(eps)
            image = (image - mean) / std
        elif normalization_scheme == "ct":
            clip_min = float(normalization_cfg.get("clip_min", -1024.0))
            clip_max = float(normalization_cfg.get("clip_max", 3071.0))
            mean = float(normalization_cfg.get("mean", 0.0))
            std = float(normalization_cfg.get("std", 1.0))
            std = max(std, eps)
            image = image.clamp(clip_min, clip_max)
            image = (image - mean) / std
        else:
            msg = f"Unknown normalization scheme '{normalization_cfg}'"
            raise NotImplementedError(msg)
    else:
        msg = f"Invalid normalization config; expected string or dict but got {normalization_cfg}"
        raise TypeError(msg)

    return image
