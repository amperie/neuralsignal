# Migration status

This file records current migration status rather than an unimplemented cutover
plan. The repository uses the v2 CLI, dataset adapters, Parquet feature storage,
local S1 trainer, and evaluator-based public SDK.

## Completed in code

- Removed the v1 public SDK, scan-first dataset/backend paths, and old automation files.
- Added JSONL/Hugging Face/MALT adapters, batched model extraction, feature shards,
  S3 bundle handoff, MinIO mirroring, and local training/MLflow integration.
- Added real T5/LongT5 CPU instrumentation tests and the bug-sweep regressions.

## Boundaries still present

Core instrumentation and feature processors remain reused legacy components.
The compatibility `sdk_config` singleton still exists. The SDK needs an injected
evaluator and does not automatically load judge or S1 models. Remote resume,
OOM recovery, full provenance, and broad batch-invariance guarantees are absent.

The collected September 12 smoke run was empty; passing local tests does not
complete the remote validation milestone. See [current state](../docs/current-state.md)
for evidence and the required rebuilt-image smoke verification.
