# Configuration

## Goal

Configuration should be explicit, composable, and safe to run locally or on
RunPod. NeuralSignal v2 should not use a global mutable singleton.

## Precedence

Configuration values resolve in this order:

1. CLI flags
2. YAML config file
3. built-in defaults

Environment variables are used for secrets and provider credentials only.

## Config Families

```text
configs/
  datasets/
    malt_import.yaml
  feature_collection/
    malt_features.yaml
  training/
    sabotage_s1.yaml
  sdk/
    local.yaml
```

## Feature Collection Config

This is the core runner config for RunPod and local GPU collection jobs.

```yaml
run:
  name: malt_sabotage_features
  seed: 42
  output_uri: s3://neuralsignal-runs/feature-runs
  local_work_dir: /workspace/neuralsignal-runs

dataset:
  source: hf
  name: metr-evals/malt-transcripts-public
  split: train
  normalizer: malt_transcripts
  max_rows: null

judge:
  model: google/flan-t5-large
  device: cuda
  dtype: auto
  generation:
    max_new_tokens: 1
    do_sample: false

batching:
  batch_size: 16
  length_bucket_size: 256
  oom_retry_min_batch_size: 1

features:
  materialize:
    - name: zones
      enabled: true
      version: v1
      config:
        output_format: name_and_value_columns
    - name: layer_distribution
      enabled: true
      version: v1
      config:
        output_format: name_and_value_columns
    - name: T-F-diff
      enabled: true
      version: v1
      config:
        output_format: name_and_value_columns
    - name: logit-lens
      enabled: false
      version: v1
      config:
        output_format: name_and_value_columns

storage:
  shard_size_rows: 10000
  format: parquet
  compression: zstd
  save_scans: false
```

## Feature Set Rules

- `features.materialize` controls which feature columns are computed and stored.
- Disabled feature sets are ignored by the runner.
- Every enabled feature set writes columns with a stable prefix:
  `{feature_set_name}__{feature_name}`.
- Every feature dataset manifest records enabled feature sets, configs, and
  versions.
- S1 training may select a subset of materialized feature columns without
  rerunning feature collection.

## Training Config

```yaml
mlflow:
  tracking_uri: file:./mlruns
  experiment_name: neuralsignal-s1

dataset:
  uri: s3://neuralsignal-local/feature-datasets/malt_sabotage_v1
  manifest: manifest.json

features:
  include_sets:
    - zones
    - layer_distribution
  include_columns: []
  exclude_columns: []

model:
  detector: sabotage
  type: xgboost
  registered_name: neuralsignal-sabotage-s1
```

## Environment Variables

```text
HF_TOKEN
RUNPOD_API_KEY
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
NEURALSIGNAL_S3_ENDPOINT_URL
MLFLOW_TRACKING_URI
```

Secrets must not be written to manifests, logs, MLflow params, or exception
messages.
