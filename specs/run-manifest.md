# Run Manifest

## Goal

The run manifest is the durable contract across local orchestration, RunPod
execution, S3 upload, local sync, MinIO curation, and MLflow provenance.

## Manifest Path

```text
s3://neuralsignal-runs/feature-runs/{run_id}/manifest.json
```

Local synced copy:

```text
data/runs/{run_id}/manifest.json
```

## Run States

```text
created
pod_starting
running
uploading
completed
failed
terminated
sync_completed
```

## Required Fields

```json
{
  "schema_version": "run_manifest.v1",
  "run_id": "2026-09-08T001122Z-malt-sabotage",
  "state": "completed",
  "created_at": "2026-09-08T00:11:22Z",
  "updated_at": "2026-09-08T01:30:00Z",
  "code": {
    "git_sha": "...",
    "git_branch": "codex/v2-refactor",
    "image": "neuralsignal:feature-collector-v2"
  },
  "remote": {
    "provider": "runpod",
    "pod_id": "...",
    "gpu": "..."
  },
  "dataset": {
    "source": "hf",
    "name": "metr-evals/malt-transcripts-public",
    "split": "train",
    "revision": null
  },
  "features": {
    "schema_version": "features.v1",
    "materialized_sets": []
  },
  "rows": {
    "expected": null,
    "written": 100000,
    "uploaded": 100000
  },
  "shards": []
}
```

## Shard Entry

```json
{
  "index": 0,
  "state": "uploaded",
  "path": "features/part-00000.parquet",
  "rows": 10000,
  "bytes": 12345678,
  "sha256": "...",
  "started_at": "...",
  "completed_at": "..."
}
```

## Failure Entry

```json
{
  "state": "failed",
  "failure": {
    "stage": "feature_collection",
    "message": "CUDA out of memory after retrying batch size 1",
    "retryable": false
  }
}
```

The failure message must not include secrets.

## Resume Rules

- Uploaded shards with valid checksums are immutable.
- Local sync skips already verified shards.
- Remote retry skips uploaded shard indexes.
- A new run id is required if config hash, code SHA, dataset revision, or
  materialized feature sets change.
