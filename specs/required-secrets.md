# Required Secrets

## Rule

Secrets must not be committed to the repo. They must not be written to YAML
configs, run manifests, logs, exceptions, or MLflow params.

## Local-Only Secrets

```text
RUNPOD_API_KEY
```

`RUNPOD_API_KEY` is used by the local CLI to create, inspect, and terminate
RunPod pods. It must not be forwarded into the pod.

## RunPod Secrets

Forward these through ignored `runpod.secrets` or `.env.runpod`:

```text
HF_TOKEN
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_SESSION_TOKEN
AWS_DEFAULT_REGION
NEURALSIGNAL_S3_BUCKET
NEURALSIGNAL_S3_ENDPOINT_URL
```

`HF_TOKEN` is required for gated Hugging Face datasets such as
`metr-evals/malt-transcripts-public`.

The AWS values should be the scoped credentials created by
`infra/terraform/s3-handoff`. They should allow RunPod to upload feature shards,
manifests, and logs under `feature-runs/`.

For AWS S3, `NEURALSIGNAL_S3_ENDPOINT_URL` can be omitted. For S3-compatible
providers, it is required.

## Local MinIO and MLflow Secrets

```text
MINIO_ACCESS_KEY
MINIO_SECRET_KEY
MLFLOW_TRACKING_URI
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
```

Local S1 training reads curated feature datasets from local disk or local MinIO
and logs trained S1 models, metrics, configs, and dataset provenance to local
MLflow.

## Suggested Ignored Files

```text
.env
.env.local
.env.runpod
runpod.secrets
```
