# Local S1 Training and MLflow

## Goal

S1 models are trained locally from synced feature datasets. MLflow is the local
system of record for trained models, metrics, configs, and dataset provenance.

RunPod does not train S1 models and does not need MLflow access.

## Training Inputs

S1 training reads feature datasets from local disk or local MinIO:

```text
s3://neuralsignal-local/feature-datasets/{dataset_version}/manifest.json
s3://neuralsignal-local/feature-datasets/{dataset_version}/features/part-*.parquet
```

The manifest provides:

- source dataset name and revision;
- run id;
- row counts;
- labels;
- judge model;
- prompt template hash;
- feature set version;
- shard checksums.

## MLflow Tracking

Each training run logs:

- S1 model artifact;
- training config;
- feature dataset URI;
- feature dataset manifest hash;
- detector name;
- feature set version;
- train/validation/test split ids;
- global metrics;
- per-slice metrics;
- threshold selection;
- calibration data;
- confusion matrix.

## Required Metrics

Global:

- AUROC
- AUPRC
- F1
- precision
- recall
- false positive rate
- false negative rate

Operational:

- training rows
- test rows
- feature count
- training duration
- inference latency per row

Slices:

- dataset split
- label type
- task family
- source model
- prompt template version

## CLI

```text
ns train s1 configs/sabotage_s1.yaml
```

Evaluation-only:

```text
ns eval s1 configs/sabotage_s1.yaml
```

## Config Sketch

```yaml
mlflow:
  tracking_uri: file:./mlruns
  experiment_name: neuralsignal-s1

dataset:
  uri: s3://neuralsignal-local/feature-datasets/malt_sabotage_v1
  manifest: manifest.json

model:
  detector: sabotage
  type: logistic_regression
  registered_name: neuralsignal-sabotage-s1

splits:
  train: train
  validation: validation
  test: test
```

## Model Registry Policy

Only locally trained S1 models are registered in MLflow. Feature collection runs
are referenced by URI and manifest hash, not registered as models.

This keeps MLflow focused on model lineage and performance rather than large
intermediate data movement.
