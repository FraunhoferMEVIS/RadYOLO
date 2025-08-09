from __future__ import annotations

import datetime
import logging
import sys
from pathlib import Path

import click
import torch
import tqdm_loggable
import tqdm_loggable.utils
import yaml
from tqdm_loggable.tqdm_logging import tqdm_logging

from yolo11_3d.nn import DATASETS, VALIDATORS
from yolo11_3d.nn.config import DatasetConfig, model_checkpoint_from_torch
from yolo11_3d.nn.train import Trainer
from yolo11_3d.nn.util import ModelEMA, export_model, select_device
from yolo11_3d.ultralytics.data.build import build_dataloader
from yolo11_3d.ultralytics.utils.plotting import plot_results as plot_results_

LOGGER = logging.getLogger(__name__)


class StdoutFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.WARNING


@click.group("yolo11-3d")
@click.version_option()
@click.option("-v", "--verbose", "verbose", is_flag=True, help="Enable verbose logging.")
@click.option("-d", "--debug", "debug", is_flag=True, help="Enable debug logging.")
def main(verbose: bool, debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO if verbose else logging.WARNING

    simple_formatter = logging.Formatter(
        # fmt="%(asctime)s %(levelname)-9s %(name)-30s: %(message)s",
        fmt="%(asctime)s %(levelname)-9s %(spacer)s: %(message)s",
        datefmt="[%H:%M:%S%z]",
        defaults={"spacer": " " * 11},
    )

    out_handler = logging.StreamHandler(sys.stdout)
    out_handler.setLevel(level)
    out_handler.addFilter(StdoutFilter())
    out_handler.setFormatter(simple_formatter)
    LOGGER.addHandler(out_handler)

    err_handler = logging.StreamHandler(sys.stderr)
    err_handler.setLevel(logging.WARNING)
    err_handler.setFormatter(simple_formatter)
    LOGGER.addHandler(err_handler)

    LOGGER.setLevel(logging.DEBUG)
    # logging.basicConfig(
    #     level=level,
    #     # handlers=[out_handler, err_handler],
    # )

    tqdm_logging.set_level(logging.INFO)
    tqdm_logging.set_log_rate(datetime.timedelta(minutes=1))
    tqdm_logger = logging.getLogger("tqdm_loggable.tqdm_logging")
    tqdm_handler = logging.StreamHandler(sys.stdout)
    tqdm_handler.setFormatter(
        logging.Formatter(
            # fmt="%(asctime)s %(levelname)-9s %(spacer)s%(message)s",
            fmt="%(asctime)s %(levelname)-9s %(message)s",
            datefmt="[%H:%M:%S%z]",
            # defaults={"spacer": " " * (30 - 11)},
        )
    )
    tqdm_logger.addHandler(tqdm_handler)
    tqdm_logger.setLevel(logging.INFO)

    yolo_handler = logging.StreamHandler(sys.stdout)
    if tqdm_loggable.utils.is_interactive_session():
        yolo_handler.setFormatter(logging.Formatter())
    else:
        yolo_handler.setFormatter(simple_formatter)

    yolo_logger = logging.getLogger("yolo")
    yolo_logger.addHandler(yolo_handler)
    yolo_logger.setLevel(level)

    if debug:
        LOGGER.debug("Debug logging enabled.")
    elif verbose:
        LOGGER.info("Verbose logging enabled.")


@main.command("train")
@click.argument(
    "config",
    envvar="YOLO113D_TRAIN_CONFIG",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
def train(config: Path) -> int:
    """Train a yolo11-3d model.

    CONFIG specifies the path to a training configuration.
    """
    try:
        trainer = Trainer(cfg=config)
        trainer.train()
    except Exception as e:
        LOGGER.exception("An unexpected error occurred", exc_info=e)
        return 1
    else:
        return 0


@main.command("validate")
@click.argument(
    "model_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.argument(
    "validator_type",
    type=click.Choice(tuple(VALIDATORS.keys()), case_sensitive=False),
)
@click.argument(
    "dataset_type",
    type=click.Choice(tuple(DATASETS.keys()), case_sensitive=False),
)
@click.argument(
    "dataset_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    metavar="DATASET_CONFIG",
)
def validate(model_path: Path, validator_type: str, dataset_type: str, dataset_path: Path) -> int:
    """Validate a yolo11-3d model.

    MODEL specifies the path to a model checkpoint.
    VALIDATOR_TYPE specifies the type of validator to use.
    DATASET_TYPE specifies the type of dataset to use.
    DATASET_CONFIG specifies the path to a dataset configuration.
    """
    try:
        if not model_path.exists():
            LOGGER.error(f"Model checkpoint {model_path} does not exist.")
            return 1

        if not dataset_path.exists():
            LOGGER.error(f"Dataset configuration {dataset_path} does not exist.")
            return 1

        with dataset_path.open() as f:
            yaml_config = yaml.safe_load(f)
        dataset_config = DatasetConfig.model_validate(yaml_config)

        dataset_test_path = dataset_config.test or dataset_config.val
        if dataset_test_path is None:
            LOGGER.error("No test or validation dataset path specified in the configuration.")
            return 1

        checkpoint = model_checkpoint_from_torch(model_path)
        model = checkpoint["ema"]
        train_args = checkpoint["train_args"]
        device = select_device(train_args.device)
        model.device = device
        model.to(device)
        ema = ModelEMA(model)

        dataset_cls, _ = DATASETS[dataset_type]

        dataset = dataset_cls(
            src_path=dataset_test_path,
            imgsz=tuple(train_args.imgsz),
            augment=False,
            batch_size=1,
            stride=int(model.stride.max().item())
            if model and hasattr(model, "stride") and isinstance(model.stride, torch.Tensor)
            else 32,
            min_stride=int(model.stride.min().item())
            if model and hasattr(model, "stride") and isinstance(model.stride, torch.Tensor)
            else 8,
            normalization=train_args.normalization
        )

        dataloader = build_dataloader(
            dataset=dataset,
            batch=1,
            workers=0,  # No workers for validation
            shuffle=False,
            device=device,
        )

        validator_cls, _ = VALIDATORS[validator_type]

        validator = validator_cls(
            dataloader=dataloader,
            save_dir=model_path.parents[1] / "validation",
            args=train_args,
            names=dataset_config.names,
        )

        validator(trainer=None, ema=ema, amp=False)
    except Exception as e:
        LOGGER.exception("An unexpected error occurred", exc_info=e)
        return 1
    else:
        return 0


@main.command("export")
@click.argument(
    "model_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
def export(model_path: Path) -> int:
    """Export a ModelCheckpoint to ONNX-Format.

    This command expects a PATH to a ModelCheckpoint loadable by `torch.load()`, usually
    a file ending with `.pt`.

    It exports the containing model to ONNX next to it.
    """
    try:
        # TODO(muellergroh) load model checkpoint
        checkpoint = model_checkpoint_from_torch(model_path)

        model = checkpoint["ema"]
        train_args = checkpoint["train_args"]
        input_channels = getattr(model, "input_channels", 3)
        input_shape = (input_channels, *train_args.imgsz)

        export_model(model, input_shape, model_path, copy=False)
    except Exception as e:
        LOGGER.exception("An unexpected error occurred", exc_info=e)
        return 1
    else:
        return 0


@main.command("onnx-compare")
@click.argument(
    "model_a",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.argument(
    "model_b",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
def onnx_compare(model_a: Path, model_b: Path) -> int:
    """Not implemented!"""
    LOGGER.error("This command is not implemented yet.")
    return 1


@main.command("plot-results")
@click.argument(
    "result_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
def plot_results(result_path: Path) -> int:
    """Plot data from a results.csv path produced during training.

    This command is intended for the use-case of plotting intermediate results of a running training.
    """
    try:
        plot_results_(result_path)
    except Exception as e:
        LOGGER.exception("An unexpected error occurred", exc_info=e)
        return 1
    else:
        return 0
