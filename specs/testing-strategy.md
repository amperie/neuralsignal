# Testing Strategy

## Goal

Tests should protect the v2 contracts without requiring a GPU for normal
development.

## No-GPU Tests

- config loading and precedence;
- feature-set selection and column filtering;
- run manifest validation;
- shard checksum validation;
- dataset source normalization;
- S3/MinIO path construction;
- CLI dry-run behavior.

## Padding Regression Tests

Use fake tokenizer/model outputs to verify:

- short example features are stable when evaluated alone;
- short example features are stable when batched with long examples;
- padding positions are excluded from mean/std/zone aggregations.

## Dataset Import Tests

Use small golden fixtures:

```text
tests/fixtures/malt_transcript_small.json
tests/fixtures/malt_expected_examples.parquet
```

Verify:

- deterministic example ids;
- correct input/output extraction;
- preserved source metadata;
- expected labels.

## Remote Workflow Tests

Use mocks for RunPod and S3:

- launch writes a created manifest;
- wait handles success/failure;
- sync downloads only missing shards;
- interrupt triggers termination;
- checksum mismatch fails loudly.

## Optional GPU Smoke Test

One marked test can run a tiny judge-model flow on CUDA:

```text
pytest -m gpu
```

This should not run in the default test suite.
