# RunPod image and publishing

[Dockerfile.runpod](../Dockerfile.runpod) uses CUDA 12.4.1 runtime on Ubuntu 22.04,
includes the OpenMP runtime required by XGBoost, installs dependencies with `uv sync --frozen`, and installs copied NeuralSignal
source under `/opt/neuralsignal`. `/workspace` is for caches and run outputs.
The image does not fetch source code from Git at startup. Rebuild after source,
dependency, or entrypoint changes; config-only changes are sent at launch.

## Local build

```bash
docker login ghcr.io -u YOUR_GITHUB_USERNAME
bash scripts/build_runpod_image.sh ghcr.io/amperie/neuralsignal-runpod-base:latest
```

Both this script and `scripts/build_runpod_image.ps1` target Linux AMD64 and push
by default. Local-only: `PUSH=0 bash scripts/build_runpod_image.sh` or PowerShell
`-Push:$false`. Keep the published image reference and pod manifest in agreement.
The package must be pullable by RunPod: public, or through the manifest's registry
auth ID. This is runtime infrastructure setup, not a test performed by the build.

## GitHub workflow

[build-runpod-image.yml](../.github/workflows/build-runpod-image.yml) is manually
triggered. It builds the selected branch's committed code on Linux AMD64, using
`GITHUB_TOKEN` with package-write permission, and publishes `latest` plus
`sha-<full-commit-sha>` to the repository owner's GHCR namespace. Its summary
includes the image digest. Every successful branch build updates shared `latest`;
use an explicit commit tag/digest in the manifest for a reproducible launch.
Select the intended current branch, not an obsolete development branch.

Local uncommitted changes are only included in a local build, not a GitHub build.
No runtime RunPod/AWS/Hugging Face credentials are needed for the image workflow.

## Entrypoint

```bash
docker run --rm --pull always --platform linux/amd64 ghcr.io/amperie/neuralsignal-runpod-base:latest --help
```

The worker accepts `--run-id` (or `NEURALSIGNAL_RUN_ID`) and optional `--config`.
Without a config path it reads `NEURALSIGNAL_FEATURE_CONFIG_B64`. S3 destination,
batching, and dataset settings are in that config, not separate worker flags.

The shell entrypoint starts public-key SSH when keys are available, runs the
worker in tmux session `ns`, streams logs, and returns its exit code. `--help`
runs directly. Artifacts/logs are under `/workspace/neuralsignal-runs/<run-id>`;
cache defaults in the manifest point to `/workspace/cache/huggingface`.

A help check verifies the entrypoint, not GPU collection. See
[current validation](../docs/current-state.md) and [secrets](required-secrets.md).
