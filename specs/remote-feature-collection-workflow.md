# Remote collection and recovery

The supported launch is:

```bash
uv run ns collect --remote configs/remote/malt_smoke.yaml --run-id YOUR_NEW_RUN_ID
```

Use a new run ID and an image built from the intended code. The local feature
config is encoded into the pod environment; source code comes from the image.

## Actual lifecycle order

1. Load environment/config/manifest and optional Terraform/forwarded secrets.
2. Select a GPU and launch a pod (dry-run stops before launch).
3. Worker normalizes examples, lazily loads the judge, collects and writes shards.
4. Worker marks its manifest completed, creates a ZIP and SHA-256 sidecar, uploads both.
5. Launcher waits for both objects and attempts pod termination in `finally`.
6. Download and verify the ZIP checksum, delete remote ZIP/sidecar, extract locally.
7. Optionally mirror the directory to MinIO and run local S1 training.

The bundle contains the manifest, features, and any log files present when it was
created. Its worker log does not contain the later bundle-upload messages.
The result reports `run_id`, `pod_id`, `bundle_uri`, `target_dir`, and
`training_metrics` (null when training was not requested).

## Failure behavior

Waiting errors, timeout, and Ctrl-C trigger a termination attempt. Termination
failure can prevent subsequent download. Worker process failure before upload
is not automatically detected from the exit-code file; polling may wait until
the local timeout. Partial shards and failure logs are not automatically synced.
There is no remote resume, keep-pod flag, or dedicated terminate/status CLI.

S3 missing objects are treated as not ready; authorization/service/connection
errors propagate. Use a finite timeout and keep the launcher running. If cleanup
cannot be confirmed, inspect the pod in RunPod and terminate it there.

Download verifies the ZIP checksum, but collection does not validate semantic
manifest success or per-shard row counts before reporting handoff success.
Inspect the output. Extraction stages the archive before replacing old results,
so invalid archives do not erase the existing directory. Remote objects are
deleted before extraction/mirroring/training; local ZIP and checksum are retained
under the target parent directory for recovery.

`remote sync S3_PREFIX LOCAL_DIR` is only for an expanded manifest/shard layout,
not the handoff ZIP. It cannot recover an already deleted remote bundle.
[Storage details](s3-minio-storage.md) · [Provider settings](runpod-orchestration-reference.md).
