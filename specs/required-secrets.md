# Credentials and environment

Start with [.env.example](../.env.example). Keep real credentials out of committed
files. The remote collection command reads `.env` by default without overwriting
existing process variables. Other commands do not automatically load that file.

## Local launcher

- `RUNPOD_API_KEY` (alias `RUNPOD_KEY`) authenticates pod operations; removed from worker env.
- Standard boto3 credentials/profile, optional `AWS_SESSION_TOKEN`, and AWS region
  provide local handoff download/delete access.
- `NEURALSIGNAL_S3_ENDPOINT_URL` configures the lifecycle's local S3 client.
- `NEURALSIGNAL_S3_BUCKET` supplies a default destination when config omits one.

Terraform outputs are merged into **forwarded worker secrets**, not installed
as the local boto3 credentials. Configure local AWS access independently.

## Forwarded worker environment

The launcher forwards `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` from the local environment,
then merges Terraform bucket/region/RunPod key outputs, then `--secrets-file`
KEY=VALUE entries. Manifest `env` is also available. An arbitrary `.env.runpod`
is not automatically loaded as a forwarded secrets file.

Typical worker values:

```dotenv
HF_TOKEN=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_SESSION_TOKEN=
AWS_DEFAULT_REGION=us-east-1
NEURALSIGNAL_S3_BUCKET=
NEURALSIGNAL_S3_ENDPOINT_URL=
```

Use a Hugging Face token with dataset/model access when required. The source
adapter checks both token variable names. S3 endpoint is optional for AWS S3.
For custom forwarding, pass a secrets file explicitly; ordinary local AWS env
values are not automatically copied to the worker.

## MinIO and MLflow

Mirroring uses `NEURALSIGNAL_MINIO_ENDPOINT_URL` or `MINIO_ENDPOINT_URL`,
`MINIO_ACCESS_KEY`, and `MINIO_SECRET_KEY`. MLflow uses training config settings
and its normal environment, such as `MLFLOW_TRACKING_URI`, username and password.
RunPod collection does not invoke MLflow or local MinIO services.

Public SSH keys belong in RunPod account settings. Private key paths configure
the local printed SSH command only. [Security details](security-and-secrets.md).
