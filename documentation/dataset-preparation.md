# Dataset Preparation

## Expected dataset layout

```text
<data_directory>
    images
        <image_1_filename>.nii.gz
        <image_2_filename>.nii.gz
        <image_3_filename>.nii.gz
        ...
    labels
      <image_1_filename>.json
      <image_2_filename>.json
      <image_3_filename>.json
        ...
    masks [Optional]
        <image_1_filename>.nii.gz
        <image_2_filename>.nii.gz
        <image_3_filename>.nii.gz
        ...
    dataset.yaml
    data_splits.json
```

## Label file format

The preferred object label format is JSON with one file per case in `labels/`.
Each file contains a list of objects, analogous to the prediction format.

Example `labels/<case>.json`:

```json
[
  {
    "class": 3,
    "box_relative_xyzwhd": [0.4, 0.3, 0.8, 0.1, 0.2, 0.1]
  },
  {
    "class": 0,
    "box_relative_xyzwhd": [0.8, 0.6, 0.7, 0.1, 0.1, 0.3]
  }
]
```

Where:

- `class` is the integer class id.
- `box_relative_xyzwhd` is `[x_center, y_center, z_center, x_extent, y_extent, z_extent]`,
  normalized to the image size.

Backward compatibility:

- Legacy `labels/<case>.txt` files are still supported but deprecated and produce a warning.

To migrate legacy labels in place:

```bash
uv run -m radyolo create-json-labels-from-txt <path-to-dataset>/labels
```

## Optional masks

The `masks/` folder is optional.

If present, each file should provide a NIfTI (.nii.gz) with one channel per object containing a binary mask. Channel index must match the object index in the corresponding label file. Missing mask files are allowed; in that case the segmentation loss for that case is ignored.

## dataset.yaml

The dataset folder must include a `dataset.yaml` with class names and normalization setup.

Example:

```yaml
names:
  0: spleen
  1: right_kidney
  2: left_kidney
  3: gall_bladder
normalization:
  0: CT
  1: MRPerChannel
  2: MRPerChannel
```

Normalization modes:

- `CT`: computes intensity mean/std on training data and applies clipped z-score normalization.
- `MRPerChannel`: z-score normalization per image channel.
- `MRGlobal`: must be configured for all channels; computes one shared mean/std across channels.

## data_splits.json

`data_splits.json` defines which cases belong to `train`, `val`, and `test` groups.
Training and inference commands can use this file to select folds and groups consistently.

For the explicit JSON structure and examples, see [Data Splits](./data-splits.md).
