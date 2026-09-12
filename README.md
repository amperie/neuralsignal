# NeuralSignal

NeuralSignal v2 is an indirect-only experiment and SDK stack for collecting activation-derived feature datasets, training local S1 classifiers, and tracking model artifacts in MLflow.

The current workflow persists compact feature shards and manifests instead of raw activation scans. Remote GPU workers handle expensive feature collection; local storage, MinIO, and MLflow are the durable records.

## Workflow

1. Load `.env`, a feature collection config, the RunPod manifest, and Terraform S3 handoff outputs.
2. Launch a RunPod worker with the configured Docker image and run config.
3. Collect judge-model activations, materialize configured feature sets, write feature shards, and upload a bundle plus checksum to S3.
4. Wait for the bundle locally, terminate the pod, download and unpack the run, and optionally mirror it to MinIO.
5. Train an S1 model locally from the feature dataset and log metrics, selected features, and model artifacts to MLflow.

The S3 bucket is temporary transport only. The durable outputs are the unpacked run directory, optional MinIO mirror, and MLflow artifacts.

## Public SDK

The package exports a small inference surface:

```python
from neuralsignal import NeuralSignal

ns = NeuralSignal(detectors=["sabotage"], evaluator=my_evaluator)
result = ns.evaluate("user prompt", "assistant response")
```

The public SDK does not expose dataset creation, training, remote execution, raw scan persistence, or direct-mode APIs.

## CLI

Install the project environment, then use the `ns` command:

```powershell
uv run ns -h
uv run ns dataset import -h
uv run ns features collect-local -h
uv run ns remote collect -h
uv run ns remote sync -h
uv run ns train s1 -h
```

Command groups:

- `ns dataset import`: validate/import dataset sources.
- `ns features collect-local`: collect local feature shards without raw scan persistence.
- `ns remote collect`: run the full RunPod/S3/local lifecycle.
- `ns remote sync`: download an existing remote feature run.
- `ns train s1`: train and log a local S1 model.

## Local Feature Collection

```powershell
uv run ns features collect-local configs/feature_collection/example_runpod_jsonl.yaml `
  --input-jsonl data/examples.jsonl `
  --out runs/local/example
```

The input JSONL rows should normalize to this shape:

```json
{"id":"example-1","input":"...","output":"...","label":0,"metadata":{}}
```

## Remote Collection

Dry-run first. This prints a redacted RunPod payload and does not launch a pod:

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

Run for real by removing `--dry-run` and setting a timeout:

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

## Local S1 Training

```powershell
uv run ns train s1 configs/training/sabotage_s1.yaml
```

The training config points at a feature dataset path and may include MLflow settings. The current trainer uses scikit-learn logistic regression.

## Feature Sets

Feature sets are selected during collection because they define the dataset columns:

```yaml
features:
  materialize:
    - name: zones
      enabled: true
    - name: layer_distribution
      enabled: true
    - name: T-F-diff
      enabled: false
    - name: logit-lens
      enabled: false
```

`logit-lens` should stay disabled until its feature path is made padding-aware.

## Terraform S3 Handoff

The Terraform scaffold in `infra/terraform/s3-handoff` creates a private S3 bucket for temporary bundles plus a scoped IAM user/key for RunPod and local handoff access.

```powershell
aws configure sso --profile qc
terraform -chdir=infra/terraform/s3-handoff init
terraform -chdir=infra/terraform/s3-handoff apply
terraform -chdir=infra/terraform/s3-handoff output -json
```

Do not make the bucket public. Public write access would allow accidental overwrite, deletion, storage cost surprises, and poisoning of model-training inputs.

## Secrets

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

## RunPod Image

```powershell
scripts/build_runpod_image.ps1 -Image ghcr.io/amperie/neuralsignal-runpod-base:latest
```

```bash
scripts/build_runpod_image.sh ghcr.io/amperie/neuralsignal-runpod-base:latest
```

Push the image to the registry used by `configs/runpod_manifest.yaml` before launching real jobs.

## Development Checks

Focused v2 test set:

```powershell
uv run pytest neuralsignal/tests/test_batch_invariance_config_v2.py neuralsignal/tests/test_bundle_v2.py neuralsignal/tests/test_remote_job_v2.py neuralsignal/tests/test_remote_lifecycle_v2.py neuralsignal/tests/test_cli_v2.py neuralsignal/tests/test_sdk_public_surface_v2.py neuralsignal/tests/test_feature_sets_masking_v2.py neuralsignal/tests/test_feature_runner_v2.py neuralsignal/tests/test_runpod_v2.py neuralsignal/tests/test_s3_sync_v2.py neuralsignal/tests/test_s1_training_v2.py neuralsignal/tests/test_v2_foundation.py neuralsignal/tests/test_padding_masking.py neuralsignal/tests/test_dataset_sources_v2.py -q
```

Compile check:

```powershell
uv run python -m py_compile neuralsignal/cli/main.py neuralsignal/remote/lifecycle.py neuralsignal/remote/job.py
```
## RunPod smoke test (macOS / Linux)

Start Docker Desktop first. Authenticate to GHCR using your GitHub username and
a token with package write access (enter the token at the password prompt),
then build and push a Linux AMD64 image, including on Apple Silicon:

```bash
docker login ghcr.io -u YOUR_GITHUB_USERNAME
bash scripts/build_runpod_image.sh
docker buildx imagetools inspect ghcr.io/amperie/neuralsignal-runpod-base:latest
```

Both build scripts push by default. For a local-only build, use
`PUSH=0 bash scripts/build_runpod_image.sh`, or pass `-Push:$false` to the
PowerShell script.

Check the published image entrypoint without a GPU:

```bash
docker run --rm --pull always --platform linux/amd64 ghcr.io/amperie/neuralsignal-runpod-base:latest --help
```

The package must be public, or `runpod.container_registry_auth_id` in
`configs/runpod_manifest.yaml` must identify RunPod registry credentials with
read access. If publishing under another owner, change the image argument and
the manifest's `runpod.image` together.

Install the local environment and inspect the launch payload:

```bash
uv sync --frozen
uv run ns remote collect configs/remote/malt_smoke.yaml --dry-run
```

Launch the eight-example GPU smoke test after the image is published:

```bash
uv run ns remote collect configs/remote/malt_smoke.yaml
```

Edit `configs/remote/malt_smoke.yaml` to change the GPU, timeout, output directory,
or run ID. Set a new `run_id` before each repeat run. The referenced feature
config controls the model and eight-example limit.

This uses `.env` and Terraform handoff outputs by default, streams the MALT
source, limits collection to eight normalized examples, and downloads the
result under `runs/remote/<run-id>`. The CLI attempts pod termination when the
bundle is ready, on interruption, or after the timeout. Keep the local command
running until it finishes. The timeout includes model and dataset loading.
This smoke test does not invoke MinIO or S1 training. Local tests and a dry-run
do not verify GPU execution; that requires the published image and a real pod.

Remote launch YAMLs under `configs/remote/` can also be passed directly to
`ns remote collect`, or supplied with `--launch-config`. Explicit command-line
options override launch YAML values; otherwise the fallback timeout is 1800
seconds and the polling interval is 30 seconds. For repeat runs, override the
sample YAML's fixed `run_id` with a fresh `--run-id`.

The image stores code and its environment under `/opt/neuralsignal`, leaving
`/workspace` for caches and run outputs. Its entrypoint forwards worker arguments,
runs collection in the `ns` tmux session (`tmux attach -t ns`), streams logs,
and returns the worker's exit
code. `--help` runs directly without starting a tmux session.

Choose a GPU by target VRAM (available GPUs within ±25% are listed with hourly
prices, cheapest first):

```bash
uv run ns remote collect configs/remote/malt_smoke.yaml -gb 24
```

For 24 GB, the range is 18–30 GB. Select a numbered GPU, or use `--yes` to
choose the cheapest matching GPU with a known price automatically:

```bash
uv run ns remote collect configs/remote/malt_smoke.yaml -gb 24 --yes
```

The smoke launch YAML also defaults to `gpu_vram_gb: 24`, so `-gb` is optional.
`--gpu-vram-gb` remains an alias for `-gb`. Use `--gpu-id` for an exact GPU
instead. Add `--dry-run` to query and select without launching a pod.

The local launcher logs preparation, GPU selection, pod status changes, elapsed
wait time, download, cleanup, and optional training with timestamps. When RunPod
assigns the public SSH endpoint, it prints `ssh root@<ip> -p <port>` and
`tmux attach -t ns`. The SSH service may need a moment to finish starting.
Add your public SSH key to your RunPod account settings before launching; use
`-i <private-key-path>` with the printed command if your key is not a default SSH
identity. No private key is sent to the pod.

The image exposes TCP port 22 and starts an SSH server with public-key
authentication using RunPod's injected public keys. Rebuild and push the image
after changing its entrypoint. Worker logs show model initialization, each batch,
shard writes, and bundle upload; view them in tmux or in
`/workspace/neuralsignal-runs/<run-id>/runpod.log`. Local status polling does not
stream the worker logs. Set `NEURALSIGNAL_LOG_LEVEL=DEBUG` for more local detail.

Interactive terminals show GPU models in cyan and prices in green. Logs use dim
timestamps and source names, with subtle level colors (cyan for info, yellow for
warnings, red for errors). Redirected output and worker log files stay plain;
set `NO_COLOR=1` to disable terminal colors.

## Manual GitHub image build

The workflow `.github/workflows/build-runpod-image.yml` builds `Dockerfile.runpod`
on a Linux AMD64 GitHub-hosted runner and pushes to
`ghcr.io/amperie/neuralsignal-runpod-base`. It runs only when triggered manually,
and publishes both `latest` and `sha-<full-commit-sha>`. Its run summary includes
the image digest. The selected branch's committed code is built; local edits
are not included. Every successful run updates `latest`, including branch runs.

Setup:

1. Commit and push the workflow. It must also exist on the repository's default
   branch (`main`) for GitHub to display the manual run button. Select
   `codex/v2-refactor` when running if that is the code you want to build.
2. Ensure repository Actions settings allow GitHub's checkout action and Docker's
   login, Buildx, and build/push actions.
3. For the existing GHCR package, open **Package settings → Manage Actions access**,
   add `amperie/neuralsignal`, and grant **Write** access if it does not already
   have it. The workflow requests `contents: read` and `packages: write`.
4. Open **Actions → Build and publish RunPod image → Run workflow**, select the
   branch, and run it.

No custom GitHub secrets are required. The workflow authenticates using the
short-lived `GITHUB_TOKEN` supplied automatically by GitHub. Do not upload
RunPod, AWS, Hugging Face, or SSH private keys for this image build; those are
runtime settings. No Docker Hub account or token is required.

For RunPod to pull the image, make the GHCR package public or configure a
RunPod registry credential with read access and set `container_registry_auth_id`
in the pod manifest. Image publication does not change package visibility.

References: [GitHub image publishing](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images),
[manual workflows](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow),
[package access](https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility).
