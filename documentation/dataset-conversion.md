# Dataset Conversion Helpers

## Convert RadYOLO dataset to nnDetection

```bash
uv run -m radyolo convert-to-nndetection <path-to-radyolo-dataset> --out-root <det-data-root> --task-id 100 --task-name MyDataset --fold 0
```

Creates `<det-data-root>/Task100_MyDataset/` containing:

- `raw_splitted/imagesTr`
- `raw_splitted/labelsTr`
- optional `raw_splitted/imagesTs`
- optional `raw_splitted/labelsTs`
- `dataset.yaml`

## Convert RadYOLO dataset to nnU-Net

```bash
uv run -m radyolo convert-to-nnunet <path-to-radyolo-dataset> --out-root <nnunet-raw-root> --dataset-id 100 --dataset-name MyDataset --fold 0
```

This command converts a RadYOLO dataset into an nnU-Net raw dataset folder:

- Output folder: `<nnunet-raw-root>/Dataset100_MyDataset/`
- Creates `imagesTr/` and `labelsTr/`
- Creates `imagesTs/` and `labelsTs/` when a test split exists
- Writes `dataset.json` with `channel_names`, `labels`, `numTraining`, and `file_ending`
- Writes `mapping.csv` mapping original RadYOLO case/image names to exported nnU-Net files

How splits are used:

- Uses `data_splits.json` by default (configurable via `--split-file-name`)
- Uses one fold selected by `--fold`
- `train` cases are always exported to training
- `val` cases are included in training by default (`--include-val-in-train`)
- `test` cases are exported to `imagesTs/labelsTs`
- Also exports an nnU-Net split JSON file (default `splits_final.json`, configurable via `--export-split-file-name`)

How labels are generated:

- If object masks are available and compatible, semantic labelmaps are composed from masks.
- Otherwise, labelmaps are rasterized from bounding boxes.
- Class encoding is integer-based with background `0` and object class value `class_id + 1`.

## Convert to integer encoded masks

```bash
uv run -m radyolo convert-to-integer-encoded-masks <path-to-radyolo-dataset>
```

This command creates a new folder `masks_integer_encoded/` inside the dataset and writes one
single-channel mask per case (`.nii.gz`) with integer class IDs.

Behavior per case:

- Reads case IDs from `images/` (`.nii` and `.nii.gz` supported)
- Uses labels from `labels/<case>.json` (falls back to deprecated `labels/<case>.txt`)
- If masks in `masks/<case>.nii.gz` are present and compatible, uses them
- Otherwise falls back to box-based rasterization from label boxes
- If a case has no labels, writes an all-zero mask

Label encoding:

- `0` is background
- Object voxels are written as `class_id + 1`

This makes the masks directly usable as semantic segmentation targets with explicit background.
