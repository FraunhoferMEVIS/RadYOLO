# Data Splits

## Create k-fold split

```bash
uv run -m radyolo create-k-fold-split <path-to-dataset-folder>
```

Default behavior is 5-fold cross validation with train/val/test assignment per fold:

- 3 folds for training
- 1 fold for validation
- 1 fold for testing

Validation is used for model selection during training, while test data remains held out to compute test metrics once the training is finished.

## Create one train/val/test split

You can also create a single split configuration:

```bash
uv run -m radyolo create-train-val-test-split <path-to-dataset-folder>
```

Use `--help` on each command for detailed options like percentages, seed, grouping by case prefix, and output split filename.

## data_splits.json format

`data_splits.json` is a JSON array where each item represents one split (for example, one fold).
Each split object contains string arrays named `train`, `val`, and `test`.

Minimal example (single split):

```json
[
	{
		"train": ["case_001", "case_002", "case_003"],
		"val": ["case_004"],
		"test": ["case_005"]
	}
]
```

Example with two folds:

```json
[
	{
		"train": ["case_001", "case_002", "case_003"],
		"val": ["case_004"],
		"test": ["case_005"]
	},
	{
		"train": ["case_004", "case_005", "case_001"],
		"val": ["case_002"],
		"test": ["case_003"]
	}
]
```

Case names are filename stems (without extension) matching files in `images/`, `labels/`, and optional `masks/`.
