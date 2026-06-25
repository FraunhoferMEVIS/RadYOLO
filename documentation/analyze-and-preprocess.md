# Analyze And Preprocess Data

## Analyze dataset

To compute dataset statistics (image sizes, foreground intensities, labels):

```bash
uv run -m radyolo analyze-dataset <path-to-dataset-folder>
```

## Preprocess dataset images to arrays

RadYOLO supports dataset preprocessing as an alternative to on-the-fly data preprocessing. This potentially saves time for datasets when large downsampling rates are used or if the training is done patchwise on large input volumes.
To convert a dataset in predefined NIfTI format to compressed `.b2nd` ([blosc2](https://github.com/Blosc/python-blosc2)) arrays:

```bash
uv run -m radyolo preprocess-data --image-extent x y z --normalization "{type: zscore}" --image-dtype float16 <path-to-dataset-folder> <path-to-output-folder>
```

Behavior:

- Copies `labels` and dataset metadata (`dataset.yaml` and `data_splits.json`).
- Converts `images/*.nii.gz` to `images/*.b2nd` in the output dataset.
- If `masks/*.nii.gz` exist, converts them to `masks/*.b2nd`.

Resampling options:

- `--voxel-size x y z` resamples to fixed voxel size.
- `--image-extent x y z` resamples to fixed image extent.

Storage/chunking details:

- With fixed image extent, each array is stored as one full-image chunk.
- With fixed voxel size, chunk/block sizes are chosen automatically from shape and dtype for efficient patch loading.
