from __future__ import annotations

import logging
from pathlib import Path

import click

from yolo11_3d.nn.train import Trainer

LOGGER = logging.getLogger(__name__)


@click.group("yolo11-3d")
@click.version_option()
@click.option("-v", "--verbose", "verbose", is_flag=True, help="Enable verbose logging.")
@click.option("-d", "--debug", "debug", is_flag=True, help="Enable debug logging.")
def main(verbose: bool, debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
    )
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
        num_classes = getattr(model, "num_classes", 3)
        input_shape = (num_classes, *train_args.imgsz)

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
