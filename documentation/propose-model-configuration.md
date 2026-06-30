# Propose Model Configuration

To generate a training configuration adapted to dataset characteristics and model scale:

```bash
uv run -m radyolo propose-model-config <path-to-dataset-folder> --model-scale s
```

Behavior:

- Reuses `dataset_fingerprint.json` when available (otherwise runs dataset analysis automatically).
- Starts from `config/config.yaml`.
- Fills in dataset-specific class count, input channel count, normalization setup, segmentation head setting, and a stride-compatible patch size.
- Computes the proposed patch by scaling the median resampled image extent to approximately `256^3` voxels while preserving the median axis ratios.

Currently the patch size estimation is purely heuristic and does not take memory consumption into account. Training with segmentation outputs usually takes up considerably more memory (especially if there are many objects per image) than training only with bounding box outputs.