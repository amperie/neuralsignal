# S3, MinIO, and local storage

The remote worker's default transport is a bundle:

```text
s3://<handoff-bucket>/feature-runs/<run-id>/bundle.zip
s3://<handoff-bucket>/feature-runs/<run-id>/bundle.zip.sha256
```

Inside the ZIP: `manifest.json`, `features/part-*.parquet`, and available worker
logs. The worker does not upload individual shards atomically or publish a live
manifest. Both bundle objects must exist before the launcher proceeds.

Local collection stores `<target-dir>/<run-id>.zip`, its checksum sidecar, and
unpacked `<target-dir>/<run-id>/`. These are the retained handoff results. Optional
`--minio-uri` uploads the extracted directory and `minio-dataset-uri.txt` using
MinIO endpoint/credentials. The remote S3 bundle objects are deleted first.
Versioned S3 buckets may retain noncurrent object versions until lifecycle expiry.

## Prefix sync

`ns download S3_PREFIX --out LOCAL_DIR` expects an expanded `manifest.json` and
relative shard paths, such as a separately uploaded directory. It does not accept
a ZIP despite the CLI argument's broad help wording. It uses default boto3
configuration; that command does not load `.env` or wire the MinIO endpoint.
The Python function can receive a configured object store.

Sync clears any previous `.sync-complete`, rejects an explicitly unfinished run
and unready shards, confines paths to the destination, reuses matching local
checksums, verifies downloads, then writes a new completion marker. It is not an
atomic directory transaction and does not remove unlisted old files. Manifest-backed
training ignores those files. Missing run state is accepted for legacy manifests.

## Integrity boundaries

Bundle download checks SHA-256; staged extraction preserves old results on ZIP
read/extract errors. Prefix sync verifies shard checksums. Training with a manifest
also verifies per-shard row counts and duplicate paths. None of these checks prove
that feature calculations are correct; use collection tests and inspect results.

Training accepts local paths only. MinIO is an optional mirror, not an implicit
training filesystem or a provisioned service. [Credentials](required-secrets.md).
