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
- export
- plot-results
- onnx-compare

For further details please refer to the respective `--help` invocation.

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