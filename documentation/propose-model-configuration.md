# Propose Model Configuration

To generate a training configuration adapted to dataset characteristics and model scale:

```bash
uv run -m radyolo propose-model-config <path-to-dataset-folder> --model-scale m
```

Behavior:

- Reuses `dataset_fingerprint.json` when available.
- Otherwise runs dataset analysis automatically.
- Starts from `config/config.yaml`.
- Fills in dataset-specific class count, input channel count, normalization setup, segmentation head setting, and a stride-compatible patch size.
- Computes the proposed patch by scaling the median resampled image extent to approximately `256^3` voxels while preserving the median axis ratios.
