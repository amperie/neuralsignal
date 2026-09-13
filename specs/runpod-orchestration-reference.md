# RunPod provider reference

The current implementation is [remote/runpod.py](../neuralsignal/remote/runpod.py)
and [remote/lifecycle.py](../neuralsignal/remote/lifecycle.py). No external
repository is needed as an implementation prerequisite.

## Manifest-to-payload mapping

| Manifest field under `runpod` | Payload/use |
| --- | --- |
| `image` | `imageName` |
| `gpu_count` | `gpuCount`, default 1 |
| `gpu_type_ids`, `gpu_type_priority` | Requested types/priority |
| `volume_gb` | `containerDiskInGb`, default 75; not a network-volume allocation |
| `volume_mount_path` | `volumeMountPath`, default `/workspace` |
| `cloud_type` | `cloudType`, default SECURE |
| `container_registry_auth_id` | Optional `containerRegistryAuthId` |
| `ports` | Default `["22/tcp"]` |
| `ssh_key_path` | Printed local SSH command only |

The payload sets GPU compute, public-IP support, encoded config, run ID, and
`dockerStartCmd: ["--run-id", run_id]`. Top-level manifest `env` is merged with
forwarded values. `RUNPOD_API_KEY` and `RUNPOD_KEY` are removed from worker env.

Manifest `poll_seconds`, `timeout_minutes`, and `terminate_on_exit` are not used
by the local lifecycle. Set `remote_collect.poll_seconds/timeout_seconds` in
launch YAML or CLI options. There is no network-volume ID support in this builder.

## GPU selection

`--gpu-vram-gb N` lists available GPUs within ±25% of N, sorted by known
price then capacity/name. `--yes` chooses the cheapest known-price match.
`--gpu-id` bypasses interactive selection. CLI GPU choice overrides the other
launch-YAML selection mode. Dry-run can still call the provider for these queries.

## Progress and SSH

Polling reports desired pod status and the first/changed SSH endpoint, not remote
worker progress. The printed key path comes from explicit option, manifest,
`NEURALSIGNAL_SSH_KEY_PATH`/`RUNPOD_SSH_KEY_PATH`, or a detected standard local key.
No private key is forwarded. SSH service startup may lag endpoint assignment.
Worker access uses public keys injected as `SSH_PUBLIC_KEY` or `PUBLIC_KEY`.

See [lifecycle](remote-feature-collection-workflow.md),
[CLI](cli-contract.md), and [image](runpod-image.md).
