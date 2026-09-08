# CLI Contract

## Goal

The CLI owns dataset import, remote feature collection, synchronization, local
training, local evaluation, and SDK smoke tests. The public SDK remains focused
on indirect-mode detection.

## Command Groups

```text
ns dataset import
ns remote collect
ns remote launch
ns remote wait
ns remote sync
ns remote terminate
ns train s1
ns eval s1
ns sdk smoke-test
```

## Common Rules

- CLI flags override YAML config values.
- Secrets come from environment variables only.
- Commands emit structured logs.
- Commands return non-zero exit codes on failure.
- Long-running commands write resumable metadata before doing expensive work.
- Interrupted remote commands attempt pod termination unless `--keep-pod` is set.

## Dataset Import

```text
ns dataset import hf metr-evals/malt-transcripts-public --config configs/malt_import.yaml
```

Required output:

```text
data/imported/{dataset_id}/
  manifest.json
  examples/part-*.parquet
```

## Remote Feature Collection

One-shot command:

```text
ns remote collect configs/malt_features.yaml
```

Equivalent expanded flow:

```text
ns remote launch configs/malt_features.yaml
ns remote wait <run_id>
ns remote sync <run_id>
ns remote terminate <run_id>
```

`collect` should perform the full flow and terminate the pod on success, failure,
or interrupt.

## Training

```text
ns train s1 configs/sabotage_s1.yaml
```

Training reads a synced feature dataset and logs the trained S1 model, config,
metrics, and dataset provenance to local MLflow.

## Evaluation

```text
ns eval s1 configs/sabotage_s1.yaml
```

Evaluation loads an existing S1 model and writes metrics without registering a
new model version.

## Smoke Test

```text
ns sdk smoke-test --detector sabotage
```

This verifies that the installed package can load the judge model, load one S1
detector, and return a serializable `DetectionResult`.

## Exit Codes

```text
0  success
1  command/config error
2  dataset/import error
3  remote provider error
4  feature collection failed
5  sync/checksum verification failed
6  training failed
7  MLflow logging failed
```
