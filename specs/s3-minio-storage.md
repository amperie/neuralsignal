# S3 and MinIO Storage

## Goal

Remote S3-compatible storage moves feature datasets from RunPod to the local
machine. Local MinIO stores curated feature datasets that local S1 training and
MLflow can reference.

## Remote Bucket Layout

```text
s3://neuralsignal-runs/
  feature-runs/{run_id}/
    manifest.json
    features/part-*.parquet
    logs/worker.log
```

## Local MinIO Layout

```text
s3://neuralsignal-local/
  feature-datasets/{dataset_version}/
    manifest.json
    features/part-*.parquet
  mlflow-artifacts/
```

## Upload Semantics

RunPod writes shards locally first, then uploads to a temporary object name, then
publishes the final shard path.

```text
features/.part-00000.parquet.tmp
features/part-00000.parquet
```

The manifest is updated only after the final object exists and its checksum has
been computed.

## Sync Semantics

Local sync:

1. downloads `manifest.json`;
2. downloads missing completed shards;
3. verifies SHA-256 checksums;
4. writes local sync state;
5. optionally copies verified shards into local MinIO as a curated dataset.

## Source of Truth

- Remote S3 is the source of truth for active feature collection runs.
- Local MinIO is the source of truth for curated feature datasets used in local
  S1 training.
- Local disk is a cache and working copy.

## Credentials

Credential values come from environment variables. They are never written to
manifests, logs, configs, or MLflow params.
