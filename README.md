# NeuralSignal

NeuralSignal collects judge-model activation features, writes Parquet datasets,
and trains local binary S1 classifiers. The public SDK wraps a caller-supplied
inference evaluator. It does not yet load a trained detector automatically.

## Current state

The implementation includes the bug-sweep fixes and the XGBoost S1 trainer.
The September 12, 2026 local verification passed **177 tests**, including real
small T5/LongT5 CPU models and an XGBoost/MLflow save-load test.

The latest `malt-smoke-20260912-001` run (pod `1ljc4sl5wudxa3`, collected at
12:53 local time) passed the int8 CUDA feature smoke checks: eight examples,
one shard, 1,598 zone features and 5,530 distribution features, all finite, with
matching bundle/shard checksums. It supersedes the earlier empty run using the
same ID. It does not establish S1 quality or suitability of the partial smoke
dataset for training. See [verification and limitations](docs/current-state.md).

## Install and test

On macOS, install XGBoost’s OpenMP runtime once with `brew install libomp`.
Run commands from the repository root:

```bash
uv sync --frozen
uv run ns --help
uv run pytest neuralsignal/tests -q
```

The CLI has five commands: `run`, `validate`, `collect`, `train`, and `download`.
Use `ns collect CONFIG --remote` for RunPod execution; collection runs locally
by default. Pass YAML paths directly, or omit them to choose from `configs/`.
The picker lists only YAML configs that fit the command, including nested files.
Each command's help includes examples, configuration details, and data requirements:

```bash
uv run ns run -h
uv run ns validate -h
uv run ns collect -h
uv run ns train -h
uv run ns download -h
```

Help uses colored headings, flags, and examples in terminals. Set `NO_COLOR=1`
to disable colors; redirected or piped help is always plain text.

## Complete workflow

```bash
uv run ns run
```

Choose a collection/launch YAML and a training YAML from the colored menus.
The worker reads the dataset configured in the collection YAML, collects features
on RunPod, and sends back a verified bundle. The command terminates the pod,
downloads and extracts the features, then trains S1 on that downloaded run.
Keep the command running until training completes.

For explicit configuration:

```bash
uv run ns run configs/feature_collection/example_runpod_malt.yaml --train-config configs/training/s1.yaml --gpu-vram-gb 24 --yes
```

RunPod credentials, the handoff storage setup, and a published worker image are
required as described below. Dataset files must be accessible to the worker;
this command does not upload arbitrary local JSONL files. Use complete labeled
training data, not the partial smoke dataset. A unique run ID is generated when
neither the command nor launch YAML provides one. `--dry-run` previews collection
without launching or training.

Training prints metrics and saves a model, predictions, and reports under
`runs/s1/<timestamp>-<unique-id>/`. MLflow reporting is enabled by default;
reporting errors are logged while local results remain available.

## Collect features

For local JSONL collection:

```bash
uv run ns validate data/examples.jsonl
uv run ns collect configs/feature_collection/example_runpod_jsonl.yaml --input data/examples.jsonl --out runs/local/example
```

Supply your own input file. The example collection config uses CUDA and int8;
change its model settings for a different environment. JSONL rows use:

```json
{"id":"example-1","input":"question","output":"response","labels":["sabotage"],"metadata":{}}
```

Collection preserves `labels` as `labels_json` and other JSONL annotations
as metadata. Training can use an existing numeric target, map a categorical
column or metadata field, or create a target from label membership. MALT
sample features are pooled per run before training; incomplete coverage warns
and uses the available samples. See [datasets](specs/dataset-imports.md) and [training](specs/local-s1-mlflow.md).

Use a fresh output directory. Existing feature shards are rejected to prevent
mixing runs. Set `extraction.mode: model` for activation features; omitted mode
means `placeholder`, which writes character counts for pipeline testing.

## RunPod smoke test

Prepare local credentials using [.env.example](.env.example) and the
[Terraform setup](infra/terraform/s3-handoff/README.md). Local S3 credentials and
worker credentials are configured separately; see [required secrets](specs/required-secrets.md).

Publish an image containing the current code before launching. With Docker
running and GHCR access configured:

```bash
docker login ghcr.io -u YOUR_GITHUB_USERNAME
bash scripts/build_runpod_image.sh
docker run --rm --pull always --platform linux/amd64 ghcr.io/amperie/neuralsignal-runpod-base:latest --help
```

The build script pushes Linux AMD64 by default. The PowerShell equivalent is
`scripts/build_runpod_image.ps1`. A manual GitHub workflow is also available;
see [image publishing](specs/runpod-image.md). Select the intended current
branch; source edits are baked into the image, not fetched at pod startup.

Inspect the launch configuration, then launch using a fresh run ID:

```bash
uv run ns collect --remote configs/remote/malt_smoke.yaml --run-id YOUR_NEW_RUN_ID --dry-run
uv run ns collect --remote configs/remote/malt_smoke.yaml --run-id YOUR_NEW_RUN_ID
```

Replace `YOUR_NEW_RUN_ID` before running. The launch YAML requests a GPU near
24 GB VRAM, an eight-example limit, and a 30-minute timeout. GPU selection may
prompt even on dry-run. `--yes` chooses the cheapest priced match; `--gpu-id`
selects an exact GPU. CLI options override launch YAML values.

Keep the local command running. It waits for `bundle.zip` and its checksum,
attempts pod termination, downloads and verifies the bundle, deletes the remote
handoff objects, then extracts locally. Results go under `runs/remote/<run-id>`.
This smoke config does not request MinIO mirroring or S1 training, so
`training_metrics: null` is expected. For acceptance, check for eight written
rows, actual Parquet shards, and both enabled feature prefixes; handoff success
alone does not validate features.

Add a public SSH key to the RunPod account before launch. The launcher prints
the SSH command when an endpoint appears; inside the pod use `tmux attach -t ns`.
Set `--ssh-key-path` for a different private key. Worker logs are at
`/workspace/neuralsignal-runs/<run-id>/runpod.log`; local polling does not stream
them. See [lifecycle and recovery](specs/remote-feature-collection-workflow.md).

## Train S1 locally

Select a config and an existing feature run interactively with `uv run ns train`,
or supply both paths explicitly:

```bash
uv run ns train configs/training/s1.yaml --run runs/remote/complete-run
```

The trainer uses XGBoost, a stratified 75/25 split, and threshold
0.5. The training config logs metrics and the XGBoost model to
`http://z440.lan:5000`. Reporting failures produce warnings and do not fail
training. `--run` overrides `dataset.path`; without `--run`, the CLI lists
available runs for selection. MALT training pools completions into samples
and samples into runs before splitting; partial smoke-test runs are unsuitable
for training. MinIO data must first be downloaded to a local path.

For target selection, the neutral `configs/training/s1.yaml` leaves the target
unspecified. The CLI shows available columns and labels, then asks which values
are positive and negative. An explicit numeric column can be selected with
`--target-column outcome` (or `--target-column metadata.outcome`); label membership
can be selected with `--positive-label gives_up`. Existing configs with
`dataset.label_column` still work. No dataset is assumed to contain sabotage.

Your target definition and class counts are saved in `training.json`, and split
assignments in `split.json`. The small smoke run remains insufficient for training;
selecting a valid label now reports a clear data error instead of a traceback.
See [target definitions](specs/local-s1-mlflow.md#target-definitions) for YAML mappings.

## Public SDK

```python
from neuralsignal import NeuralSignal

# my_evaluator(examples, detectors) must return one score row per example.
ns = NeuralSignal(detectors=["sabotage"], evaluator=my_evaluator)
result = ns.evaluate("question", "response")
print(result.to_json())
```

`my_evaluator` is application-provided. See the [SDK contract](specs/sdk-public-surface.md).

## Documentation

- [Current state and validation](docs/current-state.md)
- [Architecture](specs/v2-architecture.md) and [migration status](specs/migration-plan.md)
- [CLI reference](specs/cli-contract.md) and [configuration](specs/configuration.md)
- [MALT monitoring](docs/malt_agent_monitoring_plan.md) and [dataset adapters](specs/dataset-imports.md)
- [Feature schema](specs/feature-schema.md), [manifest](specs/run-manifest.md), [padding](specs/padding-aware-batching.md), [LongT5](specs/longt5-instrumentation.md)
- [Remote workflow](specs/remote-feature-collection-workflow.md), [provider reference](specs/runpod-orchestration-reference.md), [image](specs/runpod-image.md), [storage](specs/s3-minio-storage.md)
- [Training and MLflow](specs/local-s1-mlflow.md), [evaluation](specs/s1-evaluation.md), [tests](specs/testing-strategy.md)
- [Secrets](specs/required-secrets.md), [security](specs/security-and-secrets.md), [Terraform resources](specs/terraform-s3-handoff.md)
