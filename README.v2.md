# NeuralSignal v2

NeuralSignal v2 is a refactor focused on one workflow: collect activation-derived feature datasets quickly on remote GPU workers, bring those datasets back to the local machine, train S1 models locally, and track the trained models in local MLflow.

The v2 design intentionally keeps the SDK surface small and indirect-only. Raw scans are no longer the default artifact. Feature sets are materialized during collection, stored in dataset shards, and filtered later during S1 training.

## Centerpiece workflow

1. A local CLI command loads `.env`, the feature collection config, the RunPod manifest, and Terraform S3 outputs.
2. The CLI launches a RunPod pod with the configured Docker image and passes the run config to the worker.
3. The RunPod worker collects activations and configured feature sets, writes local feature shards and metadata, bundles the run, and uploads the bundle plus checksum to S3.
4. The local CLI waits for the S3 bundle. When it appears, or when the user presses Ctrl+C, the CLI terminates the pod and downloads whatever completed bundle is available.
5. The CLI deletes the handoff bundle from S3, unpacks it into the target directory, and can mirror the unpacked dataset to MinIO.
6. Local S1 training runs from the feature dataset and logs metrics, selected features, model artifacts, and dataset links to local MLflow.

The handoff S3 bucket is treated as temporary transport, not durable storage. Durable local storage is the unpacked run directory, MinIO, and MLflow.

## Current status

Implemented in this branch:

- v2 specs in `specs/`.
- Feature collection configs with selectable feature sets.
- Local feature shard writer and run manifests.
- S3 bundle upload, download, checksum, delete, and unpack helpers.
- RunPod pod payload generation and launch/terminate API wrapper.
- Remote lifecycle CLI that can dry-run, launch, wait for an S3 bundle, terminate, unpack, mirror to MinIO, and optionally train S1.
- Terraform scaffold for a private S3 handoff bucket and scoped IAM user/key using AWS profile `qc`.
- Padding-mask feature tests and a batch invariance test harness.
- Local S1 training with MLflow logging.

Still deliberately limited:

- The remote worker currently supports JSONL dataset input.
- The CLI feature extractor is still a placeholder; wiring the real judge-model activation pipeline into the v2 collector is the next core implementation step.
- Hugging Face MALT import is specified but not fully wired into the remote worker yet.
- `logit-lens` should stay disabled until its feature path is made padding-aware.

## CLI help

The CLI is exposed as `ns` when installed through the project environment. Every command supports `-h`:

```powershell
uv run ns -h
uv run ns remote -h
uv run ns remote collect -h
uv run ns features collect-local -h
uv run ns train s1 -h
```

Main command groups:

- `ns dataset import`: validate/import dataset sources.
- `ns features collect-local`: collect local feature shards without raw scan persistence.
- `ns remote collect`: run the full RunPod/S3/local lifecycle.
- `ns remote sync`: download an existing remote feature run.
- `ns train s1`: train and log a local S1 model.

## Remote collection example

Dry-run first. This prints the RunPod payload with secrets redacted and does not launch a pod:

```powershell
uv run ns remote collect configs/feature_collection/example_runpod_jsonl.yaml `
  --manifest configs/runpod_manifest.yaml `
  --run-id malt-smoke-001 `
  --secrets-file runpod.secrets `
  --env-file .env `
  --terraform-dir infra/terraform/s3-handoff `
  --target-dir runs/remote `
  --minio-uri s3://neuralsignal-datasets/malt-smoke-001 `
  --train-config configs/training/sabotage_s1.yaml `
  --dry-run
```

Run for real by removing `--dry-run`:

```powershell
uv run ns remote collect configs/feature_collection/example_runpod_jsonl.yaml `
  --manifest configs/runpod_manifest.yaml `
  --run-id malt-smoke-001 `
  --secrets-file runpod.secrets `
  --env-file .env `
  --terraform-dir infra/terraform/s3-handoff `
  --target-dir runs/remote `
  --minio-uri s3://neuralsignal-datasets/malt-smoke-001 `
  --train-config configs/training/sabotage_s1.yaml `
  --timeout-seconds 7200
```

## Local feature collection example

```powershell
uv run ns features collect-local configs/feature_collection/example_runpod_jsonl.yaml `
  --input-jsonl data/examples.jsonl `
  --out runs/local/example
```

## Local S1 training example

```powershell
uv run ns train s1 configs/training/sabotage_s1.yaml
```

The training config points at a feature dataset path and can include MLflow settings. The remote lifecycle can also call this automatically after downloading a bundle.

## Feature-set configuration

Feature sets are configured at collection time because they define what columns exist in the dataset. The current pattern is:

```yaml
features:
  materialize:
    - name: zones
      enabled: true
    - name: layer_distribution
      enabled: true
    - name: T-F-diff
      enabled: true
    - name: logit-lens
      enabled: false
```

Training then selects columns from the materialized feature dataset. This lets us keep collection expensive but simple: collect known-good feature sets once, then iterate quickly over S1 model configs locally.

## Padding and batch invariance

The padding bug came from padded batch tokens changing features for shorter prompts. The v2 fix is to pass the tokenizer attention mask into feature extraction and exclude padding positions before computing token-dependent feature summaries.

This applies to attention-derived and MLP-derived features when the feature aggregates over the token axis. It is not only an attention feature problem. If an MLP activation tensor includes padded token positions and the feature summarizes across tokens, padding can contaminate it too.

Run the configured batch test:

```powershell
uv run pytest neuralsignal/tests/batch_invariance.py -q
```

That test compares each row processed alone against the same row inside a padded batch. The expected result is exact or near-exact feature equality within the configured tolerance.

The config lives at:

```text
configs/tests/batch_invariance.yaml
```

## Terraform S3 handoff bucket

The Terraform scaffold is in:

```text
infra/terraform/s3-handoff
```

It creates:

- A private S3 bucket for temporary handoff bundles.
- Public access blocking.
- Bucket versioning and lifecycle cleanup.
- Server-side encryption.
- A scoped IAM user and access key for RunPod/local handoff access.

The AWS profile is `qc`:

```powershell
aws configure sso --profile qc
terraform -chdir=infra/terraform/s3-handoff init
terraform -chdir=infra/terraform/s3-handoff apply
terraform -chdir=infra/terraform/s3-handoff output -json
```

Do not make the bucket public read/write. It would be convenient, but it is the wrong default: it allows accidental overwrite, deletion, unexpected storage cost, and public poisoning of model-training inputs. The scoped IAM user/key is a better low-friction compromise.

## Required secrets

Start from `.env.example` and create a local `.env`. Do not commit real secrets.

Local `.env` values:

```dotenv
RUNPOD_API_KEY=
AWS_PROFILE=qc
AWS_DEFAULT_REGION=us-east-1
NEURALSIGNAL_S3_BUCKET=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
MINIO_ENDPOINT_URL=http://localhost:9000
MINIO_ACCESS_KEY=
MINIO_SECRET_KEY=
MLFLOW_TRACKING_URI=http://localhost:5000
```

Optional `runpod.secrets` values are forwarded to the pod:

```dotenv
HF_TOKEN=
AWS_DEFAULT_REGION=
NEURALSIGNAL_S3_BUCKET=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
```

If Terraform has been applied, the local CLI can read S3 bucket and key outputs from `infra/terraform/s3-handoff` and merge them with `.env`.

## RunPod image

Build the worker image with one of:

```powershell
scripts/build_runpod_image.ps1 -Image ghcr.io/amperie/neuralsignal-runpod-base:latest
```

```bash
scripts/build_runpod_image.sh ghcr.io/amperie/neuralsignal-runpod-base:latest
```

Push the image to the registry used by `configs/runpod_manifest.yaml` before launching real jobs.

## MALT dataset direction

The target first external import is:

```text
https://huggingface.co/datasets/metr-evals/malt-transcripts-public
```

The intended importer should normalize rows into the v2 dataset shape:

```json
{"id":"...","input":"...","output":"...","label":0,"metadata":{}}
```

For now, remote collection expects JSONL in that normalized shape. The MALT-specific Hugging Face adapter should produce that JSONL or stream equivalent examples into the same `DatasetExample` interface.

## Development checks

Focused v2 test set:

```powershell
uv run pytest neuralsignal/tests/test_batch_invariance_config_v2.py neuralsignal/tests/test_bundle_v2.py neuralsignal/tests/test_remote_job_v2.py neuralsignal/tests/test_remote_lifecycle_v2.py neuralsignal/tests/test_cli_v2.py neuralsignal/tests/test_sdk_public_surface_v2.py neuralsignal/tests/test_feature_sets_masking_v2.py neuralsignal/tests/test_feature_runner_v2.py neuralsignal/tests/test_runpod_v2.py neuralsignal/tests/test_s3_sync_v2.py neuralsignal/tests/test_s1_training_v2.py neuralsignal/tests/test_v2_foundation.py neuralsignal/tests/test_padding_masking.py neuralsignal/tests/test_dataset_sources_v2.py -q
```

Compile check:

```powershell
uv run python -m py_compile neuralsignal/cli/main.py neuralsignal/remote/lifecycle.py neuralsignal/remote/job.py
```
