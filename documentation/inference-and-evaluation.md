# Inference And Evaluation

## Predict

Run model inference with:

```bash
uv run -m radyolo predict <path-to-model-checkpoint> <input-image-folder> <output-folder>
```

- `input-image-folder` should contain NIfTI files, one file per case.
- By default, prediction reuses checkpoint normalization and patch size.
- You can override preprocessing for inference only:
  - `--overwrite-patch-size x y z`
  - `--overwrite-normalization "{type: ...}"`
- Patch sizes have to be multiples of 32.
- Output contains detections and optional segmentation masks in `output-folder`.

The prediction for each case is written as one JSON file containing a list of detections. Example:

```json
[
  {
    "class": 0,
    "class_name": "class_0",
    "confidence": 0.525784,
    "box_image_xyzxyz": [343.13, 268.64, 117.26, 366.47, 292.20, 125.61],
    "box_world_xyzxyz": [-115.64, -31.11, -101.34, -95.57, -10.86, -80.47],
    "box_relative_xyzxyz": [0.670168, 0.524693, 0.820018, 0.715762, 0.570697, 0.878409]
  },
  ...
]
```

For each `*_xyzxyz` box, values are ordered as `[x_min, y_min, z_min, x_max, y_max, z_max]`.

## Evaluate predictions

Compute detection metrics from prediction JSON files and reference labels:

```bash
uv run -m radyolo evaluate-predictions <prediction_folder> <reference_label_folder>
```

Files are matched by case name via filename stem. Metrics are computed only for cases present in both folders.

## Merge prediction masks

Merge per-detection masks into one semantic mask per case with integer-encoded classes. Because 0 is used as background label, all the object classes will be use the label `class_id + 1`.

```bash
uv run -m radyolo merge-prediction-masks <predictions_dir>
```
