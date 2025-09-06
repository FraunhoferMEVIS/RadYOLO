# yolo11-3d

YOLOv11 from ultralytics modified to work with 3D images.

## Installation

This project uses `uv` for package managing and provides a `pyproject.toml`. It can therefore be installed using any standards compliant package manager, e. g.

`uv build .`

`pip install .`

## Usage

After installation this project provides a cli `yolo11-3d` with built-in help.

Available commands are:
- train
- validate
- onnx-export
- plot-results
- analyze-dataset
- create-data-splits
- convert-to-nndetection

For further details please refer to the respective `--help` invocation.

### Convert YOLO11_3D dataset to nnDetection

You can convert a dataset in this repo's format into nnDetection's Task structure:

```
pwsh
uv run -m yolo11_3d convert-to-nndetection <path-to-yolo11_3d-dataset> --out-root <det_data_root> --task-id 100 --task-name MyDataset --fold 0
```

This creates `<det_data_root>/Task100_MyDataset/` with `raw_splitted/imagesTr`, `labelsTr` and optional `imagesTs`, `labelsTs`, plus a `dataset.yaml`.

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


## Docker Image

You can build the docker image using the command

```bash
docker build -t yolo11-3d:latest .
```

## Development

The source code can be found in `src/`

The `yolo11-3d`-package is split up into two subpackage trees.

The `ultralytics` subpackage contains code copied from the ultralytics repository. To account for 3D data part of the code was slightly modified. Therefore all code in this package is subject to their respective license.

The `nn` subpackage contains code which was heavily modified or developed from scratch and is therefore not considered to be subject to the ultralytics license.