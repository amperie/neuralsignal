# Feature dataset schema

[The runner](../neuralsignal/features/runner.py) writes one row per normalized
example. Files are `features/part-00000.parquet`, incrementing per shard, with
zstd compression by default.

## Stored columns

```text
run_id
example_id
row_index
input
output
labels_json
metadata_json
<feature columns>
```

Input/output text is retained, not hashed. No automatic dataset ID, detector,
judge model, timestamps, prompt hash, or numeric target column is added.
Feature values use `{feature_set_name}__{feature_name}` prefixes. Registered sets
are `zones`, `layer_distribution`, `T-F-diff`, and `logit-lens`; the smoke config
enables only the first two. Placeholders use the chosen prefixes but contain
input/output character counts, not activation measurements.

## Validation

Collection rejects no enabled sets, no normalized examples, empty feature rows,
and mismatched extractor batch lengths. Model extraction additionally rejects
empty individual sets, column/value length mismatches, and duplicate columns.
The writer refuses a directory already containing feature shards.

Checksums and per-shard row counts are stored in the [manifest](run-manifest.md).
Sync verifies checksums; manifest-backed training verifies paths, duplicate paths,
checksums, and row counts and ignores files absent from the manifest.

## Versions and training selection

The run manifest has `schema_version: run_manifest.v1` and nested
`features.schema_version: features.v1`. Remote manifests copy raw materialization
entries; local CLI manifests currently record only the feature schema version.
There is no automatic column inventory/config hash or feature-version compatibility
validation, even though `FeatureSetSpec` has version/manifest helper methods.

Training selects the union of matching `include_sets` and explicit
`include_columns`, removes `exclude_columns`, and deduplicates names. With no
inclusions it selects columns containing `__`. Missing requested columns/sets
are silently unmatched unless the final selection is empty. Changing feature
calculations requires recollection and separately recorded provenance; the
current version strings alone do not identify that change.
