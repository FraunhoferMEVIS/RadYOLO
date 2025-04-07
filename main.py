from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import yaml
from torch.profiler import ProfilerActivity, profile

from nn.model import DetectionModel, HardcodedModel
from nn.util import model_to_dict

LOGGER = logging.getLogger("main")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Hello from yolo11-3d!")
    scale = "n"
    config_path = Path() / "config" / "yolo11.yaml"
    with config_path.open() as f:
        config = yaml.safe_load(f)
    config["scale"] = scale
    # model, _ = parse_model(config, 3)
    model1 = DetectionModel(config, nc=1)
    model2 = HardcodedModel(input_channels=3, num_classes=1, scale=scale)
    LOGGER.info(model1)

    inputs = torch.randn((16, 3, 320, 320))
    with profile(
        activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True
    ) as prof1:
        result1 = model1(inputs)
    # LOGGER.info(prof1.key_averages().table("self_cpu_memory_usage"))

    with profile(
        activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True
    ) as prof2:
        result2 = model2(inputs)
    # LOGGER.info(prof2.key_averages().table("self_cpu_memory_usage"))
    LOGGER.info("Done!")

    model1_dict = model_to_dict(model1)
    model1_dict = model1_dict["children"][0]

    model2_dict = model_to_dict(model2)

    del model1_dict["type"]
    del model1_dict["parameters"]
    del model2_dict["type"]
    del model2_dict["parameters"]

    model1_json = json.dumps(model1_dict, sort_keys=True, indent=2)
    model2_json = json.dumps(model2_dict, sort_keys=True, indent=2)
    # LOGGER.info(model1_json)
    LOGGER.info("Models are equal: %s", model1_json == model2_json)

    # print(result1)


def train() -> None:
    trainer = Trainer("config/custom.yaml")
    trainer.train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
