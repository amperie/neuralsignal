# Feature Schema

## Goal

Feature datasets are the contract between GPU feature collection and local S1
training. They must be compact, versioned, resumable, and easy to filter.

## File Format

Feature shards are parquet files:

```text
features/part-00000.parquet
features/part-00001.parquet
```

Default compression: `zstd`.

## Required Columns

```text
run_id
dataset_id
dataset_name
dataset_split
example_id
row_index
detector
label
judge_model
prompt_template_hash
feature_set_version
input_hash
output_hash
created_at
```

## Feature Columns

Feature columns use this naming convention:

```text
{feature_set_name}__{feature_name}
```

Examples:

```text
zones__layer_04_zone_02_mean
layer_distribution__layer_10_std
T-F-diff__layer_08_delta_mean
logit-lens__layer_12_true_token_rank
```

The initial registered feature-set names match the existing implementation:

```text
zones
logit-lens
T-F-diff
layer_distribution
```

## Feature Set Manifest Entry

Each materialized feature set is recorded in `manifest.json`:

```json
{
  "name": "zones",
  "enabled": true,
  "version": "v1",
  "config_hash": "sha256:...",
  "columns": [
    "zones__layer_00_zone_00_mean"
  ]
}
```

## Schema Versioning

The manifest includes:

```json
{
  "schema_version": "features.v1",
  "feature_set_version": "v2"
}
```

Rules:

- Adding feature columns is backward compatible.
- Removing or renaming feature columns requires a new feature-set version.
- Changing a feature calculation requires a new feature-set version.
- Changing only training column selection does not require recollection.

## Training-Time Feature Selection

S1 training can select columns by feature set:

```yaml
features:
  include_sets:
    - zones
    - layer_distribution
  include_columns: []
  exclude_columns:
    - zones__debug_column
```

Selection is applied after loading the full materialized dataset. This allows one
expensive RunPod collection job to support multiple S1 experiments.

## Checksums

Every shard has a SHA-256 checksum in the run manifest. Local sync must verify
the checksum before marking a shard complete.

## Compatibility Checks

Training must fail fast if:

- requested feature sets are missing;
- requested columns are missing;
- manifest schema version is unsupported;
- feature-set version is incompatible with the training config;
- row counts do not match the manifest.
