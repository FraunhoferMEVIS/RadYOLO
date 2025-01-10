from __future__ import annotations

import logging
from pathlib import Path

import torch
import yaml
from torch.profiler import ProfilerActivity, profile

from nn.model import DetectionModel

LOGGER = logging.getLogger("main")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Hello from yolo11-3d!")
    config_path = Path() / "config" / "yolo11.yaml"
    with config_path.open() as f:
        config = yaml.safe_load(f)
    config["scale"] = "m"
    # model, _ = parse_model(config, 3)
    model = DetectionModel(config, nc=1)
    LOGGER.info(model)

    inputs = torch.randn((16, 3, 320, 320))
    with profile(
        activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True
    ) as prof:
        result = model(inputs)
    LOGGER.info(prof.key_averages().table("self_cpu_memory_usage"))
    LOGGER.info("Done!")
    # print(result)


if __name__ == "__main__":
    main()
