# RunPod Orchestration Reference

## Source Reference

Use the NURESA repo as the implementation reference for pod lifecycle handling:

```text
E:\Programming\NURESA
E:\Programming\NURESA\scripts\runpod_launch.py
E:\Programming\NURESA\configs\runpod_manifest.yaml
E:\Programming\NURESA\Dockerfile.runpod
E:\Programming\NURESA\scripts\runpod_entrypoint.sh
E:\Programming\NURESA\docs\runpod.md
```

Do not copy NURESA experiment-specific concepts into NeuralSignal. Reuse the
RunPod lifecycle shape: manifest-driven launch, config upload, wait/poll,
optional log following, status, termination, and dry-run planning.

## Useful NURESA Patterns

NeuralSignal should adopt these patterns:

- a `configs/runpod_manifest.yaml` file for image, GPU, volume, env, registry
  auth, polling, timeout, and termination defaults;
- a local wrapper command that forwards to a RunPod launcher module;
- a one-shot command that launches, waits, syncs outputs, and terminates by
  default;
- explicit `status` and `terminate` commands for manual recovery;
- `--dry-run` support that prints selected GPU and pod payload without launching;
- `--no-terminate` support for debugging;
- optional log following that writes `<pod-id>.log`;
- local config upload/encoding so config-only edits do not require an image
  rebuild;
- source code fetched at pod startup from git or a pinned code package;
- runtime secrets read from ignored local env files and passed into the pod env;
- `RUNPOD_API_KEY` used locally but not forwarded into the pod;
- Ctrl-C behavior that attempts or prompts for pod termination;
- startup failure detection with useful messages.

## NeuralSignal-Specific Changes

NURESA uses RunPod to run experiments. NeuralSignal v2 uses RunPod only to
collect feature datasets.

Therefore NeuralSignal's remote entrypoint should do exactly this:

```text
load dataset -> run judge model -> collect activations -> compute enabled
feature sets -> write parquet shards -> upload shards and manifest -> exit
```

It should not:

- train S1 models;
- contact MLflow;
- register models;
- depend on local services;
- persist raw scans by default.

## Proposed Commands

```text
ns remote plan configs/feature_collection/malt_features.yaml
ns remote collect configs/feature_collection/malt_features.yaml
ns remote launch configs/feature_collection/malt_features.yaml
ns remote wait <pod_id>
ns remote sync <run_id>
ns remote terminate <pod_id>
ns remote status <pod_id>
```

`collect` is the default happy path:

```text
plan -> launch -> wait -> sync -> terminate
```

## Manifest-Driven Pod Defaults

NeuralSignal should add:

```text
configs/runpod_manifest.yaml
```

Example shape:

```yaml
runpod:
  image: ghcr.io/amperie/neuralsignal-runpod-base:latest
  container_registry_auth_id: null
  cloud_type: SECURE
  gpu_count: 1
  gpu_type_ids: []
  gpu_ram_gb: 24
  volume_gb: 75
  volume_mount_path: /workspace
  network_volume_id: null
  poll_seconds: 60
  timeout_minutes: 0
  terminate_on_exit: true

env:
  HF_HOME: /workspace/cache/huggingface
  TRANSFORMERS_CACHE: /workspace/cache/huggingface
  NEURALSIGNAL_RUN_WORKDIR: /workspace/neuralsignal-runs

secrets:
  file: runpod.secrets
  forward:
    - HF_TOKEN
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_DEFAULT_REGION
    - NEURALSIGNAL_S3_ENDPOINT_URL
```

## Launch Payload Requirements

The pod payload must include:

- image name;
- GPU type/count or GPU RAM requirement;
- volume size and mount path;
- registry auth id when using a private image;
- environment variables;
- encoded feature collection config;
- run id;
- output S3 URI;
- entrypoint command.

## Local Config Upload

The launcher should support config-only iteration without rebuilding the image.
The local feature collection YAML should be encoded into the pod environment or
uploaded as a small startup artifact, then written inside the pod before the job
starts.

This is critical for fast experiment iteration.

## Lifecycle Behavior

On success:

```text
launch pod
wait for job completion
download and verify feature dataset
copy curated dataset to local MinIO
terminate pod
```

On job failure:

```text
download manifest/logs if available
sync completed valid shards
mark run failed locally
terminate pod unless --no-terminate
```

On local interrupt:

```text
attempt graceful termination unless --no-terminate
print manual terminate command if cleanup fails
```

## Implementation Notes

Start by adapting the NURESA launcher structure, but keep the NeuralSignal code
smaller:

- one provider module: `neuralsignal.remote.runpod`;
- one job entrypoint: `neuralsignal.remote.job`;
- one sync module: `neuralsignal.remote.sync`;
- one shared manifest module: `neuralsignal.storage.manifests`.

Avoid implementing NURESA's multi-worker sweep planner in the first pass.
NeuralSignal can add multi-pod sharding later after single-pod collection,
upload, sync, and termination are reliable.
