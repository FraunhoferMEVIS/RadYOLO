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

from yolo11_3d.config.config import DatasetConfig, model_checkpoint_from_torch
from yolo11_3d.data.data import NIfTIDatasetWithMasks
from yolo11_3d.export.onnx_export import export_model
from yolo11_3d.training.train import Trainer
from yolo11_3d.validation.validate import Validator
from yolo11_3d.utils.util import ModelEMA, select_device
from yolo11_3d.data.data import build_dataloader
from yolo11_3d.validation.plotting import plot_results as plot_results_
from yolo11_3d.data.data_splits import k_fold_data_splits
from yolo11_3d.data.analyze_dataset import analyze_dataset as analyze_dataset_fn
from yolo11_3d.data.convert_to_nndetection import convert_dataset_to_nndetection

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

    tqdm_logging.set_level(logging.INFO)
    tqdm_logging.set_log_rate(datetime.timedelta(minutes=1))
    tqdm_logger = logging.getLogger("tqdm_loggable.tqdm_logging")
    tqdm_handler = logging.StreamHandler(sys.stdout)
    tqdm_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-9s %(message)s",
            datefmt="[%H:%M:%S%z]",
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

    # Note: Do not register subcommands here; Click builds help before invoking this.


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


@main.command("test")
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
    "dataset_folder",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    metavar="DATASET_CONFIG",
)
@click.option("--fold", default=0, help="Fold from dataset splits file to use.")
@click.option("--group", default="test", help="Group from datas splits to use. Can be train, val or test.")
def test(model_path: Path, dataset_folder: Path, fold: int, group: str) -> int:
    """Test a yolo11-3d model.

    MODEL: specifies the path to a model checkpoint.
    dataset_folder: specifies the path to a dataset folder.
    fold: specifies which fold to use from the data splits file.
    group: specifies which group from data split to use. Can be train, val or test.
    """
    try:
        if not model_path.exists():
            LOGGER.error(f"Model checkpoint {model_path} does not exist.")
            return 1

        if not dataset_folder.exists():
            LOGGER.error(f"Dataset configuration {dataset_folder} does not exist.")
            return 1

        dataset_path = dataset_folder / "dataset.yaml"
        with dataset_path.open() as f:
            yaml_config = yaml.safe_load(f)
        dataset_config = DatasetConfig.model_validate(yaml_config)

        checkpoint = model_checkpoint_from_torch(model_path)
        model = checkpoint["ema"]
        train_config = checkpoint["train_config"]
        device = select_device(train_config.device)
        model.device = device
        model.to(device)
        ema = ModelEMA(model)

        dataset = NIfTIDatasetWithMasks(
            src_path=dataset_folder,
            fold=fold,
            group=group,
            imgsz=tuple(train_config.imgsz),
            augment=False,
            min_stride=int(model.stride.min().item())
            if model and hasattr(model, "stride") and isinstance(model.stride, torch.Tensor)
            else 8,
            normalization=train_config.normalization,
            load_masks = train_config.segmentation_output,
        )

        dataloader = build_dataloader(
            dataset=dataset,
            batch_size=1,
            workers=2,
            shuffle=False,
            device=device,
        )

        validator = Validator(
            dataloader=dataloader,
            save_dir=model_path.parents[1] / "validation" / group,
            config=train_config,
            names=dataset_config.names,
            save_predictions=True,
        )

        validator(trainer=None, ema=ema)
    except Exception as e:
        LOGGER.exception("An unexpected error occurred", exc_info=e)
        return 1
    else:
        return 0


@main.command("onnx-export")
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
def onnx_export(model_path: Path) -> int:
    """Export a ModelCheckpoint to ONNX-Format.

    This command expects a PATH to a ModelCheckpoint loadable by `torch.load()`, usually
    a file ending with `.pt`.

    It exports the containing model to ONNX next to it.
    """
    try:
        print(f"Exporting model {model_path} to onnx...")
        checkpoint = model_checkpoint_from_torch(model_path)

        model = checkpoint["ema"]
        train_config = checkpoint["train_config"]
        input_channels = getattr(model, "input_channels", 3)
        input_shape = (input_channels, *train_config.imgsz)

        export_model(model, input_shape, model_path, copy=False)
    except Exception as e:
        LOGGER.exception("An unexpected error occurred", exc_info=e)
        return 1
    else:
        return 0


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


@main.command("create-data-splits")
@click.argument(
    "dataset_path",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option("--seed", default=42, show_default=True, help="RNG seed for shuffling before splitting.")
@click.option("--k", default=5, show_default=True, help="Number of folds to generate. Must be >=3.")
@click.option(
    "--group-by-prefix/--no-group-by-prefix",
    default=False,
    show_default=True,
    help="If enabled, keep all cases sharing the same prefix in the same fold.",
)
@click.option(
    "--prefix-sep",
    default="_",
    show_default=True,
    help="Separator used to extract the prefix (text before first occurrence).",
)
def create_data_splits(dataset_path: Path, seed: int, k: int, group_by_prefix: bool, prefix_sep: str) -> int:
    """Create k-fold data splits from DATASET_PATH/images and write data_splits.json."""
    try:
        out_path = k_fold_data_splits(
            dataset_path=dataset_path,
            seed=seed,
            k=k,
            group_by_prefix=group_by_prefix,
            prefix_sep=prefix_sep,
        )
        LOGGER.info("Data splits written to %s", out_path)
    except Exception as e:
        LOGGER.exception("Failed to create data splits", exc_info=e)
        return 1
    else:
        return 0


@main.command("analyze-dataset")
@click.argument(
    "dataset_path",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option("--workers", default=0, show_default=True, help="Number of parallel workers to use (0 = CPU count).")
def analyze_dataset(dataset_path: Path, workers: int) -> int:
    """Analyze a dataset folder and write dataset_fingerprint.json with summary statistics.

    DATASET_PATH should contain images/ (.nii.gz), labels/ (.txt) and optional masks/ (.nii.gz).
    """
    try:
        out_path = analyze_dataset_fn(dataset_path=dataset_path, workers=workers)
        LOGGER.info("Dataset fingerprint written to %s", out_path)
    except Exception as e:
        LOGGER.exception("Failed to analyze dataset", exc_info=e)
        return 1
    else:
        return 0


@main.command("convert-to-nndetection")
@click.argument(
    "dataset_path",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    "--out-root",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path.cwd(),
    show_default=True,
    help="Root folder containing TaskXXX_* folders (det_data).",
)
@click.option("--task-id", type=int, default=100, show_default=True, help="Task number (>=100 recommended).")
@click.option("--task-name", type=str, default=None, help="Task name (defaults to dataset folder name).")
@click.option("--fold", type=int, default=0, show_default=True, help="Fold index from data_splits.json.")
@click.option(
    "--include-val-in-train/--no-include-val-in-train",
    default=True,
    show_default=True,
    help="Add validation cases to training set (imagesTr/labelsTr).",
)
@click.option(
    "--modality",
    type=str,
    default="CT",
    show_default=True,
    help="Modality string for modalities mapping in dataset.yaml.",
)
@click.option(
    "--workers",
    type=int,
    default=0,
    show_default=True,
    help="Number of parallel processes (0 = use CPU count).",
)
def convert_to_nndetection(dataset_path: Path, out_root: Path, task_id: int, task_name: str | None, fold: int, include_val_in_train: bool, modality: str, workers: int) -> int:
    """Convert a YOLO11_3D dataset to nnDetection Task folder structure.

    DATASET_PATH must contain images/ (.nii.gz), labels/ (.txt) and optional masks/ (.nii.gz).
    """
    try:
        task_dir = convert_dataset_to_nndetection(
            dataset_dir=dataset_path,
            out_root=out_root,
            task_id=task_id,
            task_name=task_name,
            fold=fold,
            include_val_in_train=include_val_in_train,
            modality_name=modality,
            workers=workers,
        )
        LOGGER.info("Created %s", task_dir)
    except Exception as e:
        LOGGER.exception("Failed to convert dataset", exc_info=e)
        return 1
    else:
        return 0


@main.command("predict")
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
    "input_dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.argument(
    "output_dir",
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    "--conf",
    default=0.1,
    type=float,
    help="Confidence threshold (overrides training config).",
)
@click.option(
    "--iou",
    default=0.2,
    type=float,
    help="IoU threshold for NMS (overrides training config).",
)
@click.option(
    "--device",
    default="0",
    type=str,
    help="Device for inference (e.g. cpu, 0). Overrides training config device.",
)
def predict(
    model_path: Path,
    input_dir: Path,
    output_dir: Path,
    conf: float | None,
    iou: float | None,
    device: str | None,
) -> int:
    """Run inference on .nii.gz images in INPUT_DIR and write JSON (and masks) to OUTPUT_DIR."""
    try:
        from yolo11_3d.inference.predict import PredictionConfig, run_prediction
        return run_prediction(
            PredictionConfig(
                model_path=model_path,
                input_dir=input_dir,
                output_dir=output_dir,
                conf=conf,
                iou=iou,
                device=device,
            )
        )
    except Exception as e:  # pragma: no cover
        LOGGER.exception("Prediction failed", exc_info=e)
        return 1
