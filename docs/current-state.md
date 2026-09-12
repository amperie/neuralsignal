# Current implementation and verification

Reviewed September 12, 2026 against the bug-sweep fixes and current XGBoost changes.
These documents describe the repository implementation, not a deployment guarantee.

## Verified locally

`uv run pytest neuralsignal/tests -q` passed **175 tests**. Coverage includes
JSONL/MALT normalization, local feature writing, training, mocked RunPod/S3 flows,
checksum failures, configuration errors, and cleanup. Tiny randomly initialized
T5 and LongT5 models run real CPU generation with real instrumentation. The T5
integration uses the smoke config's collector and both enabled feature processors,
with a synthetic tokenizer and small model rather than the pretrained judge.

The sweeps corrected empty-output success, collector defaults and layer filters,
T5 dispatch, batch token limits, hook cleanup, intentional-abort handling, feature
column checks, stale shard reuse, corrupt-bundle extraction, sync status/path checks,
S3 error propagation, training integrity checks, and invalid labels/missing values.
See [testing](../specs/testing-strategy.md) for the regression files.

## Remote result

The latest `malt-smoke-20260912-001` bundle, from pod `1ljc4sl5wudxa3` at
12:53 local time, contains eight unique examples and one valid shard. Checksums
match; all 1,598 zone and 5,530 distribution features are finite. Worker logs
confirm FLAN-T5-large loaded int8 on CUDA. This passes feature collection and
handoff checks and supersedes the earlier zero-row attempt with the same run ID.
One represented MALT run is incomplete, so these rows are not a training dataset.

## S1 and tracking

The current classifier is XGBoost with configurable `model.params`; v1 Hyperopt
and cross-validation have not been restored. The training config targets
`http://z440.lan:5000`. A local isolated MLflow test verifies metrics, parameters,
selected features, registration, model reload and matching predictions. This
machine currently cannot resolve `z440.lan`, so actual remote logging is unverified.

## Remaining limits

- End-to-end training on a complete MALT dataset and remote MLflow logging remain
  unverified. The successful GPU smoke test does not demonstrate detector quality.
- The SDK requires an evaluator; automatic judge/S1 loading is absent.
- Collection has no automatic OOM retry, length bucketing, or remote resume.
- A worker failure before bundle upload may leave the launcher waiting until its
  timeout. Failure logs/partial shards are not automatically retrieved.
- The launcher checks bundle integrity, not semantic manifest success. A valid
  old empty bundle can still be collected; inspect row counts and features.
- Manifests omit code/image/model/prompt provenance and automatic schema-version
  compatibility checks. Persist that experiment context separately.
- `remote sync` consumes an expanded manifest/shard prefix, not a ZIP bundle.
- Training requires local files; it does not read an S3/MinIO URI directly.
- Padding tests cover specific feature paths, not universal model/batch invariance.
  Decoder and LongT5 global-position masks need further work; keep logit-lens
  disabled in normal collection configs.
