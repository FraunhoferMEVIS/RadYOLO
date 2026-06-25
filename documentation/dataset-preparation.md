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

## Label file format

Each `.txt` file contains one object per line using relative coordinates:

`class x_center y_center z_center x_extent y_extent z_extent`

Example:

```text
3 0.4 0.3 0.8 0.1 0.2 0.1
0 0.9 0.6 0.7 0.1 0.1 0.3
1 0.5 0.5 0.6 0.2 0.4 0.1
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
