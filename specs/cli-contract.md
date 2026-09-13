# CLI reference

The installed `ns` entry point calls [cli/main.py](../neuralsignal/cli/main.py).
Run from the repository root so the checked-in relative config paths resolve.

| Command | Current behavior |
| --- | --- |
| `ns validate PATH` | Reads local JSONL and prints normalized row count; does not write an imported dataset. |
| `ns collect CONFIG --input PATH --out DIR` | Collects local features and writes a completed manifest. |
| `ns collect --remote [CONFIG] [OPTIONS]` | Launches RunPod and collects a checksum-protected bundle. |
| `ns download S3_PREFIX --out LOCAL_DIR` | Downloads an expanded manifest and ready shards; not a ZIP downloader. |
| `ns train [CONFIG] [--run PATH]` | Selects omitted paths interactively; trains locally, saves under `runs/s1/`, and reports metrics. |
| `ns run [CONFIG] [--train-config PATH]` | Reads the configured dataset on RunPod, collects/downloads features, then trains S1 locally. |

There are no separate launch/wait/status/terminate, evaluation-only, or SDK smoke
commands. Use each supported command's `--help` for argument syntax.

## Help and examples

`ns -h` explains the overall workflow. `ns run -h` documents the complete workflow. `ns validate -h`, `ns collect -h`,
`ns train -h`, and `ns download -h` include examples and command-specific
requirements. `--help` is equivalent to `-h` at every level. Terminal help
colors headings cyan, flags yellow, and examples green. Pipes, redirected output,
`TERM=dumb`, and the presence of `NO_COLOR` disable colors.

```bash
ns validate data/examples.jsonl
ns collect configs/feature_collection/example_runpod_jsonl.yaml --input data/examples.jsonl --out runs/local/example
ns collect configs/remote/malt_smoke.yaml --remote --run-id smoke-001 --dry-run
ns collect configs/remote/malt_smoke.yaml --remote --run-id smoke-002 --gpu-vram-gb 24 --yes
ns download s3://my-bucket/feature-runs/example --out runs/downloaded/example
ns train configs/training/sabotage_s1.yaml --run runs/downloaded/example
```

Config arguments are YAML paths, not named presets. Omitted config paths open a colored
recursive YAML picker for `configs/`. Local collection requires `--input` and
`--out`. Remote options require `--remote`, and remote
collection rejects `--input`: its dataset is defined in the feature config.
Training accepts `--run PATH` to override YAML `dataset.path`; without `--run`,
it shows a colored picker of feature runs under `runs/`. S1 output folders and
other non-feature entries are displayed but cannot be selected. Menus require
a terminal and allow `q` to cancel; scripts must supply paths explicitly. Download expects an expanded manifest and shards,
not a `bundle.zip` object.

The former `dataset import jsonl`, `features collect-local`, `remote collect`,
`remote sync`, and `train s1` command forms have been removed. Use `--input`
instead of `--input-jsonl`, `--out` instead of `--target-dir`, and
`--gpu-vram-gb` instead of `-gb`.

## Complete workflow

`ns run` implies remote collection and S1 training. Choose a collection config
and training config when omitted. `--train-config` overrides launch YAML's
`train_config`. A missing run ID is generated uniquely. Dataset source settings
come from the feature YAML; the worker must be able to access that dataset.
The freshly downloaded directory is passed directly to training. `--dry-run`
only previews collection; it never trains. See `ns run -h` for examples.

MLflow is enabled by default; reporting exceptions produce warnings, while local
training results are saved under `runs/s1/<timestamp>-<unique-id>/` and printed.
See [training outputs and MLflow](local-s1-mlflow.md) for defaults and artifacts.

## Remote collection options

`CONFIG` can be a feature config or a YAML containing `remote_collect`.
`--launch-config PATH` loads launch options explicitly. Supported overrides:

```text
--manifest --run-id --env-file --secrets-file --terraform-dir --out
--train-config --minio-uri --poll-seconds --timeout-seconds --ssh-key-path
--gpu-vram-gb   OR   --gpu-id
--yes --dry-run
```

CLI values override launch YAML, then defaults apply: manifest
`configs/runpod_manifest.yaml`, `.env`, Terraform directory
`infra/terraform/s3-handoff`, output `runs/remote`, polling 30 seconds, timeout
1800 seconds. Config and run ID are required. Paths are relative to the current
working directory, not the launch YAML. Use a fresh run ID on each launch.
`--out` overrides the launch YAML's `target_dir`; downloaded results are
extracted into `OUT/RUN_ID`. The `target_dir` YAML key is unchanged.

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
