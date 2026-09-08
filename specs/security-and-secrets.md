# Security and Secrets

## Goal

RunPod, Hugging Face, S3, MinIO, and MLflow integration must avoid leaking
credentials into configs, manifests, logs, exceptions, or model metadata.

## Secret Sources

Secrets come from environment variables:

```text
HF_TOKEN
RUNPOD_API_KEY
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_SESSION_TOKEN
AWS_DEFAULT_REGION
NEURALSIGNAL_S3_ENDPOINT_URL
MINIO_ACCESS_KEY
MINIO_SECRET_KEY
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
```

## Rules

- Do not store secrets in YAML configs.
- Do not store secrets in run manifests.
- Do not log resolved credential values.
- Do not log signed URLs.
- Redact provider error messages before writing persistent logs.
- Prefer short-lived credentials where practical.

## Local Files

`.env` files are allowed for local development but must remain gitignored.

Example:

```text
.env
.env.local
.env.runpod
```

## Manifest Redaction

Allowed:

```json
{
  "s3_endpoint": "configured",
  "hf_token": "configured"
}
```

Forbidden:

```json
{
  "hf_token": "hf_..."
}
```

## MLflow Logging

MLflow params may include dataset URIs and manifest hashes. They must not include
credentials, presigned URLs, or raw environment dumps.
