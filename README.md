# YOLO11-3D

YOLOv11 from ultralytics modified to work with 3D images.

## Installation

This project uses [`uv` for package managing](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) and provides a `pyproject.toml`. It can therefore be installed using any standards compliant package manager, e. g.

`uv build .`

`pip install .`

## Usage

After installation this project provides a cli `yolo11-3d` with built-in help.

Available commands are `train`, `test`, `plot-results`, `analyze-dataset`, `create-data-splits`, `onnx-export` and `convert-to-nndetection`.

For further details please refer to the respective `--help` invocation.

### Train

To train a model, call

```
uv run -m yolo11_3d train <path_to_config_yaml>
```

where `<path_to_config_yaml>` points to the path of a configuration file for the desired training.

### Test

The `test` command can be used to run a trained on the `test` split of the data (and also the `train` and `val` splits):

```
uv run -m yolo11_3d test <path_to_model_checkpoint> <path_to_dataset_folder>
```

## Dataset Format

The dataset format is organized as follows:
```
<data_directory>
    images
        <image_1_filename>.nii.gz
        <image_2_filename>.nii.gz
        <image_3_filename>.nii.gz
        ...
    labels
        <image_1_filename>.txt
        <image_2_filename>.txt
        <image_3_filename>.txt
        ...
    masks [Optional]
        <image_1_filename>.nii.gz
        <image_2_filename>.nii.gz
        <image_3_filename>.nii.gz
        ...
    dataset.yaml
    data_splits.json
```

The label .txt files need to have one line per object which states the object class and the center and extent in relative image coordinates. The format looks as follows:

`class x_center y_center z_center x_extent y_extent z_extent`

an example could look like

```
3 0.4 0.3 0.8 0.1 0.2 0.1
0 0.9 0.6 0.7 0.1 0.1 0.3
1 0.5 0.5 0.6 0.2 0.4 0.1
```

The masks folder is optional. If it exists the masks should have one channel per object containing a binary mask for each object. The channel index needs to match the index of the object in the labels .txt file. There can also be missing mask files. In these cases the segmentation loss for the respective case is just ignored.

In the dataset folder you need a dataset.yaml file, which states the label names and looks like this:
```
names:
  0: spleen
  1: right_kidney
  2: left_kidney
  3: gall_bladder
```

### Analyze dataset

To automatically analyze the dataset to get statistics about the image sizes, intensities and labels, you can call the command:

```
uv run -m yolo11_3d analyze-dataset <path_to_dataset_folder>
```

### Create data splits

To automtically create a data split file, you can call:

```
uv run -m yolo11_3d create-data-splits <path_to_dataset_folder>
```

By default it splits the data using a 5-fold cross validation scheme, where each group uses three of the folds as `train` set, one as `val` set and one as `test` set. The `val` set is used to compute the validation metrics during training and select the best model, while the `test` set is left out of the training to serve as independent test data in the end.

### Convert YOLO11_3D dataset to nnDetection

You can convert a dataset in this repo's format into nnDetection's Task structure:

```pwsh
uv run -m yolo11_3d convert-to-nndetection <path-to-yolo11_3d-dataset> --out-root <det_data_root> --task-id 100 --task-name MyDataset --fold 0
```

This creates `<det_data_root>/Task100_MyDataset/` with `raw_splitted/imagesTr`, `labelsTr` and optional `imagesTs`, `labelsTs`, plus a `dataset.yaml`.

## Docker Image

You can build the docker image using the command

```bash
docker build -t yolo11-3d:latest .
```

## Development

The source code can be found in `src/`

### Tests

End-to-end CLI tests live in `tests/` and create a tiny synthetic dataset (two 16×16×16 volumes) to exercise all commands quickly on CPU. Install dev dependencies and run:

```bash
cd ./tests
uv run pytest
```

Or with plain pip:
```bash
pip install -e .[dev]
cd ./tests
pytest
```

Optional coverage:
```bash
pytest --cov=yolo11_3d --cov-report=term-missing
```