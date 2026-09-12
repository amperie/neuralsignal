# CLI reference

The installed `ns` entry point calls [cli/main.py](../neuralsignal/cli/main.py).
Run from the repository root so the checked-in relative config paths resolve.

| Command | Current behavior |
| --- | --- |
| `ns dataset import jsonl PATH` | Reads local JSONL and prints normalized row count; does not write an imported dataset. |
| `ns features collect-local CONFIG --input-jsonl PATH --out DIR` | Collects local features and writes a completed manifest. |
| `ns remote collect [CONFIG] [OPTIONS]` | Launches RunPod and collects a checksum-protected bundle. |
| `ns remote sync S3_PREFIX LOCAL_DIR` | Downloads an expanded manifest and ready shards; not a ZIP downloader. |
| `ns train s1 CONFIG` | Trains from `dataset.path` locally; prints metrics and selected columns. |

There are no separate launch/wait/status/terminate, evaluation-only, or SDK smoke
commands. Use each supported command's `--help` for argument syntax.

## Remote collection options

`CONFIG` can be a feature config or a YAML containing `remote_collect`.
`--launch-config PATH` loads launch options explicitly. Supported overrides:

```text
--manifest --run-id --env-file --secrets-file --terraform-dir --target-dir
--train-config --minio-uri --poll-seconds --timeout-seconds --ssh-key-path
-gb / --gpu-vram-gb   OR   --gpu-id
--yes --dry-run
```

CLI values override launch YAML, then defaults apply: manifest
`configs/runpod_manifest.yaml`, `.env`, Terraform directory
`infra/terraform/s3-handoff`, output `runs/remote`, polling 30 seconds, timeout
1800 seconds. Config and run ID are required. Paths are relative to the current
working directory, not the launch YAML. Use a fresh run ID on each launch.

Dry-run prints a redacted payload without launching; it can read Terraform outputs
and query/prompt for GPU selection. There is no keep-pod option.

## Output and errors

Successful collection/training commands print JSON; logs and interactive GPU
prompts can accompany it. Logs use timestamps, and worker logs are separate from
local status polling. `NO_COLOR` disables terminal colors;
`NEURALSIGNAL_LOG_LEVEL` controls application verbosity.

Success returns 0. Argparse usage errors return 2. Ctrl-C returns 130 after
confirmed cleanup (or 1 if cancellation cleanup cannot be confirmed). Other
uncaught errors generally exit 1 with an exception; there is no stage-specific
exit-code taxonomy or durable launcher resume record.

[Workflow and recovery](remote-feature-collection-workflow.md) ·
[Configuration](configuration.md)
