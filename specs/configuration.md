# Configuration reference

Use the checked-in [feature configs](../configs/feature_collection/),
[launch configs](../configs/remote/), [pod manifest](../configs/runpod_manifest.yaml),
and [S1 config](../configs/training/sabotage_s1.yaml) as the supported shapes.
The loader reads a YAML mapping, deep-merges programmatic overrides, then expands
environment variables in string values. It does not validate every unknown key.

## Feature collection

| Section | Consumed settings |
| --- | --- |
| `run` | `name`, `s3_output_uri`, optional exact `bundle_uri` |
| `extraction` | `mode`: `model` or `placeholder`; omission defaults to placeholder character-count features; invalid modes fail. |
| `model` | `model_name`, `device`, `quantization` (`int8`, `int4`, or an unquantized value such as `none`) |
| `generation` | `batch_size`, `max_new_tokens`, `truncation_length` |
| `prompt` | `template` with `{input}` and `{output}`; metadata also available to custom templates. |
| `instrumentation` | Encoder/decoder/FF/attention/embedding flags and `collector_config` |
| `dataset` | Adapter settings; remote worker supports positive integer `max_examples` after normalization. |
| `features` | `materialize` list of name/enabled/config entries |
| `storage` | Runner consumes `shard_size_rows`; writer uses Parquet/zstd by default. |

Legacy `indirect_config` and `indirect_instrumentation_config` are fallback aliases
inside the model extractor. Use the current names for new configs.

`format`, `compression`, and `save_scans` in example YAMLs are not forwarded as
runtime switches by the CLI/worker. Raw scans are not persisted. The writer's
Python constructor accepts a compression argument. No length buckets or OOM
retry settings are implemented.

A positive truncation length applies to both single examples and batches; zero
means no explicit truncation. The extractor defaults to 16 generated tokens.
Collector `zone_size_by_layer: {}` uses global `zone_size`; optional map `default`
overrides it. Empty layer name/index filters mean all layers. Per-layer reductions
must be integer multiples of existing zone sizes.

`zones` needs `target_zone_size`, `field_to_process`, and layer filter settings.
`layer_distribution` accepts the same name/index filters, defaults to 10 bins,
and still accepts legacy `layers_to_process` (which takes precedence).
`T-F-diff` requires an unembedding layer object; enabling it alone in plain YAML
does not supply that object. Keep it and `logit-lens` disabled in the smoke config.

## Launch and environment precedence

Explicit remote CLI options override launch YAML, then CLI defaults. The pod
manifest supplies image/GPU/container/env settings, not the local wait timeout.
See [provider mapping](runpod-orchestration-reference.md).

Remote collection loads `.env` without overwriting existing process variables.
Worker environment combines manifest `env`, forwarded Hugging Face tokens,
Terraform handoff outputs, then `--secrets-file` values. The latter overrides
previous forwarded values. Local collection/training do not automatically load
`.env`. See [credentials](required-secrets.md).

## Training

Use `dataset.path` (local directory or Parquet file), `dataset.label_column`,
`features.include_sets/include_columns/exclude_columns`, and optional `mlflow`.
The checked-in config uses `label_column: sabotage` for MALT run pooling and
`registered_model_name` for MLflow registration. Its dataset path is an example
that must be changed. `model.type: xgboost` and `model.params` configure the classifier. Split YAML
sections do not configure the current train/test split. [Training details](local-s1-mlflow.md).
