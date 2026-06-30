# Propose Model Configuration

To generate a training configuration adapted to dataset characteristics and model scale:

```bash
uv run -m radyolo propose-model-config <path-to-dataset-folder> --model-scale s
```

You can override the training scheme selection (wholeimage or patchwise) explicitly, which is chose automatically otherwise:

```bash
uv run -m radyolo propose-model-config <path-to-dataset-folder> --model-scale s --training-mode patchwise
```

Behavior:

- Starts from `config/config.yaml`.
- Reuses `dataset_fingerprint.json` when available (otherwise runs dataset analysis automatically).
- Fills in dataset-specific class count, input channel count, normalization setup, segmentation head setting, and a stride-compatible patch size.
- Computes the proposed patch by scaling the median resampled image extent to approximately `256^3` voxels while preserving the median axis ratios.
- In `automatic` training-mode, chooses `wholeimage` when the median resampled image fits into the proposed patch size and `patchwise` otherwise.

Currently the patch size estimation is purely heuristic and does not take memory consumption into account. Training with segmentation outputs usually takes up considerably more memory (especially if there are many objects per image) than training only with bounding box outputs.