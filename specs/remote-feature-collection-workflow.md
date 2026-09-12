# Remote Feature Collection Workflow

## Centerpiece Workflow

NeuralSignal v2 treats RunPod as disposable GPU feature-collection capacity.
The local machine remains the source of truth for orchestration, training, MLflow,
and final model registration.

## End-to-End Flow

1. A local CLI command creates a run manifest and starts a RunPod pod with the
   correct image, code version, dataset config, credentials, and output location.
2. The RunPod job loads the dataset, runs the indirect judge model, collects
   activations, computes feature rows, and writes feature shards locally inside
   the pod.
3. The RunPod job uploads feature shards and metadata to S3-compatible storage.
4. The local CLI waits for completion, downloads the uploaded artifacts, verifies
   manifest checksums, and terminates the pod.
5. A local training command reads the synced feature dataset, creates S1 models,
   logs metrics and artifacts to local MLflow, and records the feature dataset
   URI from the local MinIO bucket.

## Storage Topology

Remote object storage:

```text
s3://neuralsignal-runs/feature-runs/{run_id}/
  manifest.json
  features/
    part-00000.parquet
    part-00001.parquet
  logs/
    worker.log
```

Local synced storage:

```text
data/runs/{run_id}/
  manifest.json
  features/
    part-00000.parquet
    part-00001.parquet
```

Local MinIO curated dataset copy:

```text
s3://neuralsignal-local/feature-datasets/{dataset_version}/
  manifest.json
  features/
    part-*.parquet
```

MLflow stores S1 model artifacts and points back to the MinIO dataset URI.

## CLI Commands

Launch and wait:

```text
ns remote collect configs/malt_features.yaml
```

Equivalent explicit commands:

```text
ns remote launch configs/malt_features.yaml
ns remote wait <run_id>
ns remote sync <run_id>
ns remote terminate <run_id>
```

Training:

```text
ns train s1 configs/sabotage_s1.yaml
```

## RunPod Job Contract

The pod entrypoint receives:

```text
--run-id
--config
--s3-output-uri
--dataset-cache-dir
--batch-size
--shard-size
```

It must:

- write feature shards locally first;
- upload completed shards atomically;
- update `manifest.json` after each successful shard upload;
- write checksums for every uploaded shard;
- exit non-zero on unrecoverable failure;
- never require MLflow connectivity.

## Local Orchestrator Contract

The local CLI must:

- create a unique run id;
- start the pod;
- stream or poll job status;
- sync only completed shards;
- verify row counts and checksums;
- terminate the pod on success, failure, or interrupt;
- leave enough metadata to resume download or diagnose failure.

## Failure Behavior

If feature collection fails, completed uploaded shards remain valid. The local
CLI records the failed state and still attempts pod termination.

If local sync fails, the user can rerun:

```text
ns remote sync <run_id>
```

If pod termination fails, the CLI reports the pod id and keeps retry metadata.

## Why MLflow Is Local Only

S1 model creation happens locally, so MLflow should track local training runs,
metrics, feature dataset versions, and model artifacts. RunPod should not be a
required MLflow client. That keeps remote jobs simple and avoids networking
problems between transient pods and the local MLflow server.
