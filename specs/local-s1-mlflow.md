# Local S1 training and MLflow

[training/s1.py](../neuralsignal/training/s1.py) trains XGBoost from local Parquet data. The CLI is:

```bash
uv run ns train configs/training/sabotage_s1.yaml --run runs/remote/complete-run
```

Omit CONFIG to select a YAML from `configs/`; omit `--run` to select a feature
run from `runs/`. Explicit `--run` overrides `dataset.path` in the YAML. Scripts
must supply both paths because selection requires a terminal. An S3/MinIO URI
is not a supported training path: download the dataset first.

`ns run [CONFIG] --train-config PATH` performs remote collection, verified
download, and local training in one command. Omit `--train-config` to use launch
YAML's `train_config` or choose interactively before launch. Both this command
and `ns collect CONFIG --remote --train-config PATH` train on the freshly unpacked
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

MLflow reporting is enabled by default for CLI and Python training, including
an empty config. Set `mlflow.enabled: false` to disable it. The tracking URI is
chosen from YAML `tracking_uri`, then `MLFLOW_TRACKING_URI`, then
`http://z440.lan:5000`. The default experiment is `neuralsignal-s1`.

Supported settings: `enabled`, `tracking_uri`, `experiment_name`, `run_name`,
`registered_model_name`, and `extra_params`. All reporting exceptions, including
setup, metrics, artifact upload, and registration errors, are logged as warnings
without failing completed training. Local artifacts are saved before reporting.
Default HTTP requests time out after five seconds with no retries; explicit
`MLFLOW_HTTP_REQUEST_TIMEOUT` and `MLFLOW_HTTP_REQUEST_MAX_RETRIES` environment
settings override these defaults. Interruptions still cancel the command.

The implementation logs:

- `dataset_path`, `feature_count`, `model_type: xgboost`, XGBoost parameters, plus extra params;
- the [implemented metrics](s1-evaluation.md);
- `selected_features.json`, `metrics.json`, and `classification_report.json`;
- confusion matrices as labeled JSON, CSV, and PNG;
- the XGBoost model under artifact path `model`, optionally registered.

When remote collection also mirrors to MinIO, it adds `feature_dataset_uri` as
an extra parameter. Full config, manifest hash, split IDs, calibration, and code revision are not
automatically logged to MLflow.

## Durable local outputs

Every successful fit writes a unique directory under `runs/s1/` containing:

- `model.ubj`: reloadable XGBoost model;
- `metrics.json` and `selected_features.json`;
- `training.json`: source path, label column, selected feature settings, model
  parameters, random seed, split fraction, and prediction threshold;
- `predictions.csv`: held-out row indexes, labels, scores, and predictions;
- `confusion_matrix.json` and `classification_report.json`.

Undefined metrics are `null` in saved JSON. Separate training attempts get
separate directories. The CLI prints metrics, selected columns, and `out`;
the complete workflow prints `training_metrics` and `training_output_dir`.
The Python result includes `model`, `metrics`, `feature_columns`, and
`output_dir`; Python callers may override `output_root` for another destination.
