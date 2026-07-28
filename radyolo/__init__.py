from __future__ import annotations

import datetime
import json
import logging
import os
import sys
from pathlib import Path

import click
import yaml

from radyolo.config.config import load_model_checkpoint
from radyolo.config.propose_model_config import propose_model_config as propose_model_config_fn
from radyolo.config.spatial import PatchSize, VoxelSize
from radyolo.data.analyze_dataset import analyze_dataset as analyze_dataset_fn
from radyolo.data.convert_to_integer_encoded_masks import (
    convert_dataset_to_integer_encoded_masks,
)
from radyolo.data.convert_to_nndetection import convert_dataset_to_nndetection
from radyolo.data.convert_to_nnunet import convert_dataset_to_nnunet
from radyolo.data.data_splits import k_fold_data_splits, train_val_test_data_split
from radyolo.data.object_labels import create_json_labels_from_txt
from radyolo.data.preprocess_data import preprocess_dataset as preprocess_dataset_fn
from radyolo.export.onnx_export import build_export_metadata, export_model
from radyolo.inference.merge_prediction_masks import (
    merge_prediction_masks as merge_prediction_masks_fn,
)
from radyolo.inference.predict import PredictionConfig, run_prediction
from radyolo.models.model import load_checkpoint_model
from radyolo.training.train import LOG_LEVEL_ENV, Trainer
from radyolo.utils.logging_utils import configure_tqdm_logging
from radyolo.validation.evaluate_predictions import evaluate_prediction_folder
from radyolo.validation.plotting import plot_results as plot_results_

LOGGER = logging.getLogger(__name__)


def _parse_normalization_option(
    _ctx: click.Context,
    _param: click.Parameter,
    value: str | None,
) -> dict | list[dict] | None:
    if value is None:
        return None
    try:
        parsed_value = yaml.safe_load(value)
    except yaml.YAMLError as exc:
        msg = f"Invalid normalization config: {exc}"
        raise click.BadParameter(msg) from exc
    if not isinstance(parsed_value, (dict, list)):
        msg = "Normalization must parse to a dict or list of dicts."
        raise click.BadParameter(msg)
    return parsed_value


def _parse_metadata_overwrite_option(
    _ctx: click.Context,
    _param: click.Parameter,
    value: tuple[str, ...],
) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for entry in value:
        if "=" not in entry:
            msg = f"Metadata overwrite entries must have the format KEY=VALUE, but got '{entry}'."
            raise click.BadParameter(msg)
        key, raw_value = entry.split("=", 1)
        key = key.strip()
        try:
            parsed_value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            msg = f"Invalid metadata value for '{key}': {exc}"
            raise click.BadParameter(msg) from exc
        overrides[key] = parsed_value
    return overrides


class StdoutFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.WARNING


@click.group("radyolo")
@click.version_option()
@click.option(
    "-v", "--verbose", "verbose", is_flag=True, help="Enable verbose logging (deprecated)."
)
@click.option("-q", "--quiet", "quiet", is_flag=True, help="Enable quiet logging.")
@click.option("-d", "--debug", "debug", is_flag=True, help="Enable debug logging.")
def main(verbose: bool, quiet: bool, debug: bool) -> None:
    if debug:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO
    os.environ[LOG_LEVEL_ENV] = str(level)

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

    configure_tqdm_logging(
        level=logging.INFO,
        log_rate=datetime.timedelta(minutes=1),
        stream_handler_factory=lambda: logging.StreamHandler(sys.stdout),
    )

    if debug:
        LOGGER.debug("Debug logging enabled.")
    elif quiet:
        LOGGER.info("Quiet logging enabled.")

    if verbose:
        LOGGER.warning(
            "Verbose logging is deprecated and the new default. "
            "Use '--debug' or '--quiet' to change logging level."
        )

    # Note: Do not register subcommands here; Click builds help before invoking this.


@main.command("train")
@click.argument(
    "config",
    metavar="CONFIG",
    envvar="RADYOLO_TRAIN_CONFIG",
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
    """Train a radyolo model.

    CONFIG specifies the path to a training configuration.
    """
    try:
        trainer = Trainer(cfg=config)
        trainer.train()
    except Exception as e:
        LOGGER.exception("An unexpected error occurred", exc_info=e)
        msg = "Training failed. See logs for details."
        raise click.ClickException(msg) from e
    else:
        return 0


@main.command("onnx-export")
@click.argument(
    "model_path",
    metavar="MODEL-PATH",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    "--internal-dtype",
    type=click.Choice(["float16", "float32"], case_sensitive=False),
    default="float16",
    show_default=True,
    help="Internal compute dtype for the exported ONNX graph. Inputs and outputs stay float32.",
)
@click.option(
    "--metadata-overwrite",
    "metadata_overrides",
    multiple=True,
    callback=_parse_metadata_overwrite_option,
    help=(
        "Override exported ONNX metadata property as KEY=VALUE. "
        "Can be passed multiple times. VALUE is parsed as YAML/JSON; "
        'example: --metadata-overwrite "normalization={type: none}".'
    ),
)
def onnx_export(
    model_path: Path,
    internal_dtype: str,
    metadata_overrides: dict[str, object],
) -> int:
    """Export a ModelCheckpoint to ONNX-Format.

    This command expects a PATH to a ModelCheckpoint loadable by `torch.load()`, usually
    a file ending with `.pt`.

    It exports the containing model to ONNX next to it.
    """
    try:
        LOGGER.info(f"Exporting model {model_path} to onnx...")
        loaded_checkpoint = load_checkpoint_model(model_path, device_override="cpu")
        train_config = loaded_checkpoint.train_config
        model = loaded_checkpoint.model

        input_shape = (train_config.input_channels, *train_config.patch_size.get_zyx())
        metadata = build_export_metadata(
            train_config=train_config,
            checkpoint_path=model_path,
            metadata_overrides=metadata_overrides,
        )
        export_model(
            model,
            input_shape,
            model_path,
            internal_dtype=internal_dtype.lower(),
            metadata=metadata,
        )
    except Exception as e:  # pragma: no cover
        LOGGER.exception("An unexpected error occurred", exc_info=e)
        msg = "ONNX export failed. See logs for details."
        raise click.ClickException(msg) from e
    else:
        return 0


@main.command("plot-results")
@click.argument(
    "result_path",
    metavar="RESULT-PATH",
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

    This command is intended for the plotting intermediate results of a running training.
    """
    try:
        plot_results_(result_path)
    except Exception as e:
        LOGGER.exception("An unexpected error occurred", exc_info=e)
        msg = "Plotting results failed. See logs for details."
        raise click.ClickException(msg) from e
    else:
        return 0


@main.command("create-k-fold-split")
@click.argument(
    "dataset_path",
    metavar="DATASET-PATH",
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
    "--seed", default=42, show_default=True, help="RNG seed for shuffling before splitting."
)
@click.option(
    "--k",
    default=5,
    show_default=True,
    help="Number of folds to generate. Must be >=3 when test groups are enabled, otherwise >=2.",
)
@click.option(
    "--create-test-group/--no-create-test-group",
    default=True,
    show_default=True,
    help="Reserve a separate test group per fold. Disable this to generate train/val-only folds.",
)
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
@click.option(
    "--split-file-name",
    default="data_splits.json",
    show_default=True,
    type=str,
    help="Name of the split file to write inside the dataset folder.",
)
def create_data_splits(
    dataset_path: Path,
    seed: int,
    k: int,
    create_test_group: bool,
    group_by_prefix: bool,
    prefix_sep: str,
    split_file_name: str,
) -> int:
    """Create k-fold data splits from DATASET_PATH/images and write a split file."""
    try:
        out_path = k_fold_data_splits(
            dataset_path=dataset_path,
            seed=seed,
            k=k,
            split_file_name=split_file_name,
            create_test_group=create_test_group,
            group_by_prefix=group_by_prefix,
            prefix_sep=prefix_sep,
        )
        LOGGER.info("Data splits written to %s", out_path)
    except Exception as e:
        LOGGER.exception("Failed to create data splits", exc_info=e)
        msg = "Creating data splits failed. See logs for details."
        raise click.ClickException(msg) from e
    else:
        return 0


@main.command("create-train-val-test-split")
@click.argument(
    "dataset_path",
    metavar="DATASET-PATH",
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
    "--seed", default=42, show_default=True, help="RNG seed for shuffling before splitting."
)
@click.option(
    "--train-percentage",
    default=70.0,
    show_default=True,
    type=float,
    help="Percentage of cases assigned to the training split.",
)
@click.option(
    "--val-percentage",
    default=15.0,
    show_default=True,
    type=float,
    help="Percentage of cases assigned to the validation split.",
)
@click.option(
    "--test-percentage",
    default=15.0,
    show_default=True,
    type=float,
    help="Percentage of cases assigned to the test split. May be 0 to keep the test split empty.",
)
@click.option(
    "--group-by-prefix/--no-group-by-prefix",
    default=False,
    show_default=True,
    help="If enabled, keep all cases sharing the same prefix in the same split.",
)
@click.option(
    "--prefix-sep",
    default="_",
    show_default=True,
    help="Separator used to extract the prefix (text before first occurrence).",
)
@click.option(
    "--split-file-name",
    default="data_splits.json",
    show_default=True,
    type=str,
    help="Name of the split file to write inside the dataset folder.",
)
def create_train_val_test_split(  # noqa: PLR0913
    dataset_path: Path,
    seed: int,
    train_percentage: float,
    val_percentage: float,
    test_percentage: float,
    group_by_prefix: bool,
    prefix_sep: str,
    split_file_name: str,
) -> int:
    """Create one train/val/test split from DATASET_PATH/images and write a split file."""
    try:
        out_path = train_val_test_data_split(
            dataset_path=dataset_path,
            seed=seed,
            split_file_name=split_file_name,
            train_percentage=train_percentage,
            val_percentage=val_percentage,
            test_percentage=test_percentage,
            group_by_prefix=group_by_prefix,
            prefix_sep=prefix_sep,
        )
        LOGGER.info("Data splits written to %s", out_path)
    except Exception as e:
        LOGGER.exception("Failed to create train/val/test split", exc_info=e)
        msg = "Creating train/val/test split failed. See logs for details."
        raise click.ClickException(msg) from e
    else:
        return 0


@main.command("analyze-dataset")
@click.argument(
    "dataset_path",
    metavar="DATASET-PATH",
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
    "--workers",
    default=0,
    show_default=True,
    help="Number of parallel workers to use (0 = CPU count).",
)
@click.option(
    "--output-file",
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    default=None,
    help="Optional output path for the dataset fingerprint JSON.",
)
def analyze_dataset(dataset_path: Path, workers: int, output_file: Path | None) -> int:
    """Analyze a dataset folder and write dataset_fingerprint.json with summary statistics.

    DATASET_PATH should contain images/ (.nii.gz), labels/ (.txt) and optional masks/ (.nii.gz).
    """
    try:
        out_path = analyze_dataset_fn(
            dataset_path=dataset_path,
            workers=workers,
            output_file=output_file,
        )
        LOGGER.info("Dataset fingerprint written to %s", out_path)
    except Exception as e:
        LOGGER.exception("Failed to analyze dataset", exc_info=e)
        msg = "Dataset analysis failed. See logs for details."
        raise click.ClickException(msg) from e
    else:
        return 0


@main.command("propose-model-config")
@click.argument(
    "dataset_path",
    metavar="DATASET-PATH",
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
    "--model-scale",
    type=click.Choice(
        ["n", "s", "m", "l", "x"],
        case_sensitive=False,
    ),
    default="m",
    help="Scale of the RadYOLO model. Larger models are slower, but potentially more accurate.",
)
@click.option(
    "--output",
    type=click.Path(
        file_okay=True, dir_okay=False, writable=True, resolve_path=True, path_type=Path
    ),
    default=None,
    help="Optional output path for the proposed training config YAML.",
)
@click.option(
    "--training-mode",
    type=click.Choice(["automatic", "wholeimage", "patchwise"], case_sensitive=False),
    default="automatic",
    show_default=True,
    help=(
        "Choose the training scheme explicitly or let propose-model-config decide "
        "from the proposed patch size."
    ),
)
@click.option(
    "--workers",
    default=0,
    show_default=True,
    help="Number of parallel workers to use when analyze-dataset needs to be run.",
)
def propose_model_config(
    dataset_path: Path,
    model_scale: str,
    output: Path | None,
    training_mode: str,
    workers: int,
) -> int:
    """Propose a training config for DATASET_PATH for a target model scale."""
    normalized_training_mode = training_mode.lower()
    try:
        result = propose_model_config_fn(
            dataset_path=dataset_path,
            model_scale=model_scale.lower(),
            training_mode=normalized_training_mode,
            output_path=output,
            workers=workers,
        )
        LOGGER.info(
            "Proposed config written to %s",
            result.output_path,
        )
        if result.analysis_ran:
            LOGGER.info("Dataset fingerprint written to %s", result.fingerprint_path)
        if normalized_training_mode == "wholeimage":
            LOGGER.info("Selected training mode: wholeimage (requested explicitly).")
        elif normalized_training_mode == "patchwise":
            LOGGER.info(
                "Selected training mode: patchwise (requested explicitly)."
            )
        elif result.training_mode == "wholeimage":
            LOGGER.info(
                "Selected training mode: wholeimage because the median image fits into "
                "the proposed patch size."
            )
        else:
            LOGGER.info(
                "Selected training mode: patchwise because the median image exceeds the "
                "proposed patch size."
            )
    except Exception as e:
        LOGGER.exception("Failed to propose model config", exc_info=e)
        msg = "Model config proposal failed. See logs for details."
        raise click.ClickException(msg) from e
    else:
        return 0


@main.command("preprocess-data")
@click.argument(
    "dataset_path",
    metavar="DATASET-PATH",
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
    "output_path",
    metavar="OUTPUT-PATH",
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    "--workers",
    default=0,
    show_default=True,
    help="Number of parallel workers to use (0 = CPU count).",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Reprocess and replace existing per-case outputs instead of skipping already preprocessed cases.",
)
@click.option(
    "--image-extent",
    nargs=3,
    type=int,
    default=None,
    help="Resample images and masks to a fixed image size as x y z.",
)
@click.option(
    "--voxel-size",
    nargs=3,
    type=float,
    default=None,
    help="Resample images and masks to a fixed voxel size as x y z.",
)
@click.option(
    "--normalization",
    default="{type: none}",
    show_default=True,
    callback=_parse_normalization_option,
    help=(
        "YAML or JSON normalization config, for example '{type: zscore}' or "
        "'{type: ct, mean: 0, std: 1, clip_min: -1024, clip_max: 3071}'."
    ),
)
@click.option(
    "--image-dtype",
    type=click.Choice(["float32", "float16", "int16"], case_sensitive=False),
    default="float32",
    show_default=True,
    help="Output dtype for saved preprocessed images.",
)
@click.option(
    "--split-file-name",
    default="data_splits.json",
    show_default=True,
    type=str,
    help="Name of the split file to copy from the input dataset to the output dataset.",
)
def preprocess_data(  # noqa: PLR0913
    dataset_path: Path,
    output_path: Path,
    workers: int,
    overwrite: bool,
    image_extent: tuple[int, int, int] | None,
    voxel_size: tuple[float, float, float] | None,
    normalization: dict | list[dict],
    image_dtype: str,
    split_file_name: str,
) -> int:
    """Preprocess a NIfTI dataset and store images as .b2nd outputs in OUTPUT_PATH."""
    try:
        if image_extent is not None and voxel_size is not None:
            msg = "Only one of --image-extent or --voxel-size can be specified."
            raise click.BadParameter(msg)
        target_image_size = PatchSize.from_xyz(image_extent) if image_extent is not None else None
        target_voxel_size = VoxelSize.from_xyz(voxel_size) if voxel_size is not None else None
        out_path = preprocess_dataset_fn(
            dataset_path=dataset_path,
            output_path=output_path,
            workers=workers,
            overwrite=overwrite,
            target_image_size=target_image_size,
            target_voxel_size=target_voxel_size,
            normalization=normalization,
            image_dtype=image_dtype.lower(),
            split_file_name=split_file_name,
        )
        LOGGER.info("Preprocessed dataset written to %s", out_path)
    except Exception as e:
        LOGGER.exception("Failed to preprocess dataset", exc_info=e)
        msg = "Dataset preprocessing failed. See logs for details."
        raise click.ClickException(msg) from e
    else:
        return 0


@main.command("convert-to-nndetection")
@click.argument(
    "dataset_path",
    metavar="DATASET-PATH",
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
@click.option(
    "--task-id", type=int, default=100, show_default=True, help="Task number (>=100 recommended)."
)
@click.option(
    "--task-name", type=str, default=None, help="Task name (defaults to dataset folder name)."
)
@click.option(
    "--fold", type=int, default=0, show_default=True, help="Fold index from data_splits.json."
)
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
@click.option(
    "--split-file-name",
    default="data_splits.json",
    show_default=True,
    type=str,
    help="Name of the split file to read from the input dataset.",
)
def convert_to_nndetection(
    dataset_path: Path,
    out_root: Path,
    task_id: int,
    task_name: str | None,
    fold: int,
    include_val_in_train: bool,
    modality: str,
    workers: int,
    split_file_name: str,
) -> int:
    """Convert a RadYOLO dataset to nnDetection Task folder structure.

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
            split_file_name=split_file_name,
        )
        LOGGER.info("Created %s", task_dir)
    except Exception as e:
        LOGGER.exception("Failed to convert dataset", exc_info=e)
        msg = "nnDetection conversion failed. See logs for details."
        raise click.ClickException(msg) from e
    else:
        return 0


@main.command("convert-to-nnunet")
@click.argument(
    "dataset_path",
    metavar="DATASET-PATH",
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
    help="Root folder containing DatasetXXX_* folders (nnUNet_raw).",
)
@click.option(
    "--dataset-id",
    "task_id",
    type=int,
    default=100,
    show_default=True,
    help="Dataset number for DatasetXXX_Name.",
)
@click.option(
    "--task-id",
    "task_id",
    type=int,
    help="Alias for --dataset-id.",
    hidden=True,
)
@click.option(
    "--dataset-name",
    "task_name",
    type=str,
    default=None,
    help="Dataset name (defaults to dataset folder name).",
)
@click.option(
    "--task-name",
    "task_name",
    type=str,
    help="Alias for --dataset-name.",
    hidden=True,
)
@click.option(
    "--fold", type=int, default=0, show_default=True, help="Fold index from data_splits.json."
)
@click.option(
    "--include-val-in-train/--no-include-val-in-train",
    default=True,
    show_default=True,
    help="Add validation cases to imagesTr/labelsTr so exported nnU-Net splits remain usable.",
)
@click.option(
    "--modality",
    type=str,
    default="CT",
    show_default=True,
    help="Base modality name for dataset.json channel_names.",
)
@click.option(
    "--workers",
    type=int,
    default=0,
    show_default=True,
    help="Number of parallel processes (0 = use CPU count).",
)
@click.option(
    "--split-file-name",
    default="data_splits.json",
    show_default=True,
    type=str,
    help="Name of the split file to read from the input dataset.",
)
@click.option(
    "--export-split-file-name",
    default="splits_final.json",
    show_default=True,
    type=str,
    help="Name of the nnU-Net split file to write into the output dataset folder.",
)
def convert_to_nnunet(  # noqa: PLR0913
    dataset_path: Path,
    out_root: Path,
    task_id: int,
    task_name: str | None,
    fold: int,
    include_val_in_train: bool,
    modality: str,
    workers: int,
    split_file_name: str,
    export_split_file_name: str,
) -> int:
    """Convert a RadYOLO dataset to nnU-Net Dataset folder structure.

    DATASET_PATH must contain images/ (.nii.gz), labels/ (.txt) and optional masks/ (.nii.gz).
    """
    try:
        dataset_dir = convert_dataset_to_nnunet(
            dataset_dir=dataset_path,
            out_root=out_root,
            dataset_id=task_id,
            dataset_name=task_name,
            fold=fold,
            include_val_in_train=include_val_in_train,
            modality_name=modality,
            workers=workers,
            split_file_name=split_file_name,
            export_split_file_name=export_split_file_name,
        )
        LOGGER.info("Created %s", dataset_dir)
    except Exception as e:
        LOGGER.exception("Failed to convert dataset", exc_info=e)
        msg = "nnU-Net conversion failed. See logs for details."
        raise click.ClickException(msg) from e
    else:
        return 0


@main.command("convert-to-integer-encoded-masks")
@click.argument(
    "dataset_path",
    metavar="DATASET-PATH",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
def convert_to_integer_encoded_masks(dataset_path: Path) -> int:
    """Create integer encoded instance masks from a RadYOLO dataset."""
    try:
        output_dir = convert_dataset_to_integer_encoded_masks(dataset_path)
        LOGGER.info("Created %s", output_dir)
    except Exception as e:
        LOGGER.exception("Failed to convert dataset", exc_info=e)
        msg = "Integer-encoded mask conversion failed. See logs for details."
        raise click.ClickException(msg) from e
    else:
        return 0


@main.command("create-json-labels-from-txt")
@click.argument(
    "labels_dir",
    metavar="LABELS-DIR",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
def create_json_labels_from_txt_command(labels_dir: Path) -> int:
    """Create .json object label files from all .txt labels in LABELS-DIR."""
    try:
        created_paths = create_json_labels_from_txt(labels_dir)
        LOGGER.info("Created %d JSON label files in %s", len(created_paths), labels_dir)
    except Exception as e:
        LOGGER.exception("Failed to create JSON labels from TXT", exc_info=e)
        msg = "Creating JSON labels from TXT failed. See logs for details."
        raise click.ClickException(msg) from e
    else:
        return 0


@main.command("predict")
@click.argument(
    "model_path",
    metavar="MODEL-PATH",
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
    metavar="INPUT-DIR",
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
    metavar="OUTPUT-DIR",
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
    help="Confidence threshold.",
)
@click.option(
    "--nms-iou",
    "--iou",
    default=0.1,
    type=float,
    help="IoU threshold for NMS.",
)
@click.option(
    "--device",
    default="0",
    type=str,
    help="Device for inference (e.g. cpu, 0).",
)
@click.option(
    "--clip-boxes-to-image/--no-clip-boxes-to-image",
    default=True,
    help="Override whether predicted 3D boxes are clipped to image bounds before export.",
)
@click.option(
    "--overwrite-patch-size",
    nargs=3,
    type=int,
    default=None,
    help="Override the checkpoint patch size for inference as x y z.",
)
@click.option(
    "--overwrite-normalization",
    default=None,
    callback=_parse_normalization_option,
    help=(
        "Override the checkpoint normalization config with YAML or JSON, for example "
        "'{type: zscore}' or '{type: ct, mean: 0, std: 1, clip_min: -1024, clip_max: 3071}'."
    ),
)
@click.option(
    "--overwrite-voxel-size",
    nargs=3,
    type=float,
    default=None,
    help="Override the checkpoint voxel size for inference as x y z.",
)
@click.option(
    "--patch-overlap",
    default=0.5,
    type=float,
    show_default=True,
    help="Sliding-window overlap fraction for patchwise inference.",
)
@click.option(
    "--patch-border-weight",
    default=0.9,
    type=float,
    show_default=True,
    help="Minimum confidence multiplier for detections touching patch borders.",
)
@click.option(
    "--preprocessing-on-gpu/--no-preprocessing-on-gpu",
    "preprocessing_on_gpu",
    default=True,
    show_default=True,
    help=(
        "For patchwise inference, move the full image to the inference device before "
        "preprocessing and patch extraction. Disable this to preprocess on CPU and move only "
        "each extracted patch for model execution."
    ),
)
@click.option(
    "--num-load-workers",
    default=2,
    type=click.IntRange(min=1),
    show_default=True,
    help="Number of loader processes that read input volumes.",
)
@click.option(
    "--num-write-workers",
    default=2,
    type=click.IntRange(min=1),
    show_default=True,
    help="Number of writer processes that export JSON and masks.",
)
@click.option(
    "--save-masks/--no-save-masks",
    default=True,
    show_default=True,
    help="Export segmentation masks when the model outputs mask coefficients.",
)
@click.option(
    "--tta-flipping/--no-tta-flipping",
    default=False,
    show_default=True,
    help="Run flip-based test-time augmentation over all spatial axis combinations.",
)
@click.option(
    "--ensemble-model-path",
    "ensemble_model_paths",
    multiple=True,
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    help="Additional checkpoint path to include in a model ensemble. Can be repeated.",
)
@click.option(
    "--ensembling-iou",
    default=0.5,
    type=float,
    show_default=True,
    help="Minimum IoU for matching detections during TTA/model ensembling.",
)
@click.option(
    "--ensembling-conf",
    default=0.1,
    type=float,
    show_default=True,
    help="Minimum confidence for keeping aggregated detections after TTA/model ensembling.",
)
@click.option(
    "--ensembling-class-agnostic/--no-ensembling-class-agnostic",
    default=False,
    show_default=True,
    help="Match ensemble detections across classes instead of only within the same class.",
)
@click.option(
    "--split-file",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    default=None,
    help=(
        "Optional data split JSON file. If set, only cases from the selected split fold/group "
        "are predicted."
    ),
)
@click.option(
    "--split-fold",
    default=0,
    type=click.IntRange(min=0),
    show_default=True,
    help="Fold index to use from --split-file.",
)
@click.option(
    "--split-group",
    default="test",
    type=click.Choice(["train", "val", "test"], case_sensitive=False),
    show_default=True,
    help="Group to use from --split-file.",
)
def predict(
    model_path: Path,
    input_dir: Path,
    output_dir: Path,
    conf: float | None,
    nms_iou: float | None,
    device: str,
    clip_boxes_to_image: bool,
    overwrite_patch_size: tuple[int, int, int] | None,
    overwrite_normalization: dict | list[dict] | None,
    overwrite_voxel_size: tuple[float, float, float] | None,
    patch_overlap: float,
    patch_border_weight: float,
    preprocessing_on_gpu: bool,
    num_load_workers: int,
    num_write_workers: int,
    save_masks: bool,
    tta_flipping: bool,
    ensemble_model_paths: tuple[Path, ...],
    ensembling_iou: float,
    ensembling_conf: float,
    ensembling_class_agnostic: bool,
    split_file: Path | None,
    split_fold: int,
    split_group: str,
) -> int:
    """Run inference on .nii.gz images in INPUT_DIR and write JSON (and masks) to OUTPUT_DIR."""
    try:
        return run_prediction(
            PredictionConfig(
                model_path=model_path,
                input_dir=input_dir,
                output_dir=output_dir,
                confidence_threshold=conf,
                nms_iou_threshold=nms_iou,
                device=device,
                clip_boxes_to_image=clip_boxes_to_image,
                overwrite_patch_size=(
                    PatchSize.from_xyz(overwrite_patch_size)
                    if overwrite_patch_size is not None
                    else None
                ),
                overwrite_normalization=overwrite_normalization,
                overwrite_voxel_size=(
                    VoxelSize.from_xyz(overwrite_voxel_size)
                    if overwrite_voxel_size is not None
                    else None
                ),
                patch_overlap=patch_overlap,
                patch_border_weight=patch_border_weight,
                preprocessing_on_gpu=preprocessing_on_gpu,
                num_load_workers=num_load_workers,
                num_write_workers=num_write_workers,
                save_masks=save_masks,
                tta_flipping=tta_flipping,
                ensemble_model_paths=ensemble_model_paths,
                ensembling_iou=ensembling_iou,
                ensembling_confidence_threshold=ensembling_conf,
                ensembling_class_agnostic=ensembling_class_agnostic,
                split_file=split_file,
                split_fold=split_fold,
                split_group=split_group.lower(),
            )
        )
    except Exception as e:  # pragma: no cover
        LOGGER.exception("Prediction failed", exc_info=e)
        msg = "Prediction failed. See logs for details."
        raise click.ClickException(msg) from e


@main.command("evaluate-predictions")
@click.argument(
    "predictions_dir",
    metavar="PREDICTIONS-DIR",
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
    "labels_dir",
    metavar="LABELS-DIR",
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
    "--output",
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    default=Path("evaluation_results"),
    show_default=True,
    help=(
        "Output directory where metrics.json, metrics_per_case.json, metadata.json, "
        "and curves.json will be written."
    ),
)
def evaluate_predictions(predictions_dir: Path, labels_dir: Path, output: Path) -> int:
    """Compute detection metrics for prediction JSON files against RadYOLO label files."""
    try:
        summary = evaluate_prediction_folder(predictions_dir, labels_dir)
        output.mkdir(parents=True, exist_ok=True)
        metrics_payload = dict(summary["metrics"])
        per_case_payload = metrics_payload.pop("per_case", [])
        (output / "metrics.json").write_text(
            json.dumps(metrics_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output / "metrics_per_case.json").write_text(
            json.dumps(per_case_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output / "metadata.json").write_text(
            json.dumps(summary["metadata"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output / "curves.json").write_text(
            json.dumps(summary["curves"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except ValueError as exc:
        LOGGER.exception("Prediction evaluation failed", exc_info=exc)
        raise click.ClickException(str(exc)) from exc
    except Exception as e:  # pragma: no cover
        LOGGER.exception("Prediction evaluation failed", exc_info=e)
        msg = "Prediction evaluation failed. See logs for details."
        raise click.ClickException(msg) from e

    click.echo(f"Wrote evaluation files to: {output!s}")
    return 0


@main.command("merge-prediction-masks")
@click.argument(
    "predictions_dir",
    metavar="PREDICTIONS-DIR",
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
    "--confidence-threshold",
    default=0.1,
    type=float,
    show_default=True,
    help="Minimum detection confidence to include a mask in the merged output.",
)
def merge_prediction_masks(predictions_dir: Path, confidence_threshold: float) -> int:
    """Merge per-detection prediction masks into one semantic mask per case."""
    try:
        output_dir = merge_prediction_masks_fn(
            predictions_dir=predictions_dir,
            confidence_threshold=confidence_threshold,
        )
    except Exception as e:  # pragma: no cover
        LOGGER.exception("Merging prediction masks failed", exc_info=e)
        msg = "Merging prediction masks failed. See logs for details."
        raise click.ClickException(msg) from e

    click.echo(f"Wrote merged masks to: {output_dir!s}")
    return 0
