# Local S1 training and MLflow

[training/s1.py](../neuralsignal/training/s1.py) trains XGBoost from local Parquet data. The CLI is:

```bash
uv run ns train s1 configs/training/sabotage_s1.yaml
```

Edit `dataset.path` to a real local run directory or Parquet file first.
An S3/MinIO URI is not a supported training path. Download the dataset first.
`remote collect --train-config PATH` instead trains from the freshly unpacked
directory, overriding the training config's dataset path.

## Input validation

With a manifest, only listed shards are read; an explicitly unfinished run,
escaping/duplicate paths, checksum mismatch, or row-count mismatch fails.
Without a manifest, the directory reader loads `features/part-*.parquet` and
cannot perform those integrity checks. A single Parquet file also bypasses
manifest checks.

`dataset.label_column` defaults to `label`; values must be numeric 0 or 1.
Generic collected `labels_json` is not automatically converted to a target.
MALT sample metadata triggers [run pooling](dataset-imports.md), with finite
features and complete sample/completion records required. Both classes and enough
examples for stratification are necessary. The eight-example smoke dataset is
not a training dataset.

Features use `include_sets`, `include_columns`, and `exclude_columns`.
Unmatched requested names are ignored unless the final selection is empty.
The trainer uses `XGBClassifier` with histogram trees, random state 42, a
stratified 75/25 split, and threshold 0.5. The Python function accepts a random
state override. `model.params` is passed to XGBoost by both local and remote
training commands; split settings are not read from YAML. Defaults are 300 trees,
max depth 6, learning rate 0.1, CPU histogram training with four threads,
`binary:logistic` objective and logloss evaluation. These restore the XGBoost
model family, not v1 Hyperopt tuning or cross-validation.

## MLflow

An empty MLflow config disables logging in direct `train_s1`/CLI use. The remote
training wrapper adds an `extra_params` mapping, so its training path invokes
MLflow even when no other MLflow settings were supplied.

Supported settings: `tracking_uri`, `experiment_name`, `run_name`,
`registered_model_name`, and `extra_params`. The checked-in config logs to `http://z440.lan:5000`. The training
host must resolve and reach `z440.lan`; logging failures propagate rather than
silently falling back to local storage.
The implementation logs:

- `dataset_path`, `feature_count`, `model_type: xgboost`, XGBoost parameters, plus extra params;
- the [implemented metrics](s1-evaluation.md);
- `selected_features.json`;
- the XGBoost model under artifact path `model`, optionally registered.

When remote collection also mirrors to MinIO, it adds `feature_dataset_uri` as
an extra parameter. Full config, manifest hash, split IDs, calibration, confusion
matrix, and code revision are not automatically logged. Without MLflow the CLI
prints results but does not save a separate model file; the Python API returns
model, metrics, and feature columns.
