# Current architecture

NeuralSignal separates feature experiments from the small public evaluator API.
Activations are collected in memory, reduced into Parquet feature rows, and used
for local S1 training. The current CLI does not persist raw scans.

| Component | Implementation |
| --- | --- |
| Public evaluator/results | `neuralsignal/sdk/client.py`, `results.py` |
| Config/env loading | `neuralsignal/config/` |
| Example schema/adapters | `neuralsignal/datasets/v2.py`, `sources/` |
| Batch collection/model extraction | `neuralsignal/features/runner.py`, `model_extractor.py` |
| Feature selection/MALT pooling | `neuralsignal/features/selection.py`, `malt_runs.py` |
| Hooks/collector/tensor reduction | `neuralsignal/core/modules/` |
| Padding helpers | `neuralsignal/inference/masking.py` |
| Training and metrics | `neuralsignal/training/s1.py` |
| RunPod lifecycle/job/prefix sync | `neuralsignal/remote/` |
| Manifests/Parquet/S3/bundles | `neuralsignal/storage/` |
| CLI | `neuralsignal/cli/main.py` |

## Data flow

```text
normalized examples -> judge generation -> in-memory activations -> feature shards
remote shards + manifest + logs -> ZIP + checksum -> S3 handoff -> local directory
local feature dataset -> optional MALT run pooling -> S1 fit/test -> optional MLflow
```

MinIO mirroring is optional. Training reads local files; it does not directly
stream from MinIO. The public SDK calls a user-provided evaluator rather than
connecting automatically to this training pipeline.

A compatibility `sdk_config` singleton remains in core modules, mainly for
logging/config access. It is not a replacement SDK configuration API. Legacy
backend, automation, scan-first dataset, and v1 public SDK paths were removed.

See [schema](feature-schema.md), [manifest](run-manifest.md),
[remote lifecycle](remote-feature-collection-workflow.md), and
[verified state](../docs/current-state.md) for implemented contracts and limits.
