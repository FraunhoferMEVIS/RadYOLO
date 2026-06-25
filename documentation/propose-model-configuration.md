# Propose Model Configuration

To generate a training configuration adapted to dataset characteristics, model scale, and GPU RAM budget:

```bash
uv run -m radyolo propose-model-config <path-to-dataset-folder> --model-scale m --gpu-ram-limit-gb 11
```

Behavior:

- Reuses `dataset_fingerprint.json` when available.
- Otherwise runs dataset analysis automatically.
- Starts from `config/config.yaml`.
- Fills in dataset-specific class count, input channel count, normalization setup, segmentation head setting, and a stride-compatible patch size for the requested GPU memory budget.
