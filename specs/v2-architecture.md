# NeuralSignal v2 Architecture

## Goal

NeuralSignal v2 is an indirect-only anomaly detection SDK and experiment system.
The core workflow is:

1. Collect judge-model activations on GPU workers.
2. Convert activations to compact feature rows immediately.
3. Persist features and metadata, not raw scans.
4. Train S1 models locally.
5. Store trained S1 models and evaluation metrics in local MLflow.

Raw scan persistence is a debug and feature-research option only.

## Non-Goals

- No direct mode.
- No scan-first dataset format.
- No Databricks dependency.
- No remote MLflow server requirement for RunPod jobs.
- No public SDK surface for dataset creation, training, or remote execution.

## Package Layout

```text
neuralsignal/
  sdk/
    client.py
    results.py
  inference/
    judge.py
    batching.py
    masking.py
    hooks.py
  features/
    pipeline.py
    registry.py
    schemas.py
  detectors/
    detector.py
    registry.py
    s1_model.py
  datasets/
    sources/
      hf.py
      malt.py
      jsonl.py
    normalize.py
    runner.py
  training/
    train_s1.py
    evaluate_s1.py
  remote/
    runpod.py
    job.py
    sync.py
  storage/
    manifests.py
    s3.py
    local.py
  cli/
    main.py
```

## Runtime Boundaries

The SDK is for inference only:

```python
from neuralsignal import NeuralSignal

ns = NeuralSignal(detectors=["sabotage"])
result = ns.evaluate(input_text, output_text)
```

Experiment, dataset, training, and RunPod workflows live behind the CLI:

```text
ns remote collect configs/malt_features.yaml
ns dataset sync <run_id>
ns train s1 configs/sabotage_s1.yaml
ns eval s1 configs/sabotage_s1.yaml
```

## Data Model

Feature rows are the primary artifact:

```text
run_id
dataset_id
dataset_name
dataset_split
example_id
detector
label
judge_model
prompt_template_hash
feature_set_version
input_hash
output_hash
features...
```

Run metadata is stored in a manifest:

```json
{
  "run_id": "2026-09-08T001122Z-malt-sabotage",
  "kind": "feature_collection",
  "dataset": {
    "name": "metr-evals/malt-transcripts-public",
    "split": "train",
    "revision": null
  },
  "judge_model": "google/flan-t5-large",
  "feature_set_version": "v2",
  "rows": 100000,
  "shards": [
    {
      "path": "features/part-00000.parquet",
      "rows": 10000,
      "sha256": "..."
    }
  ]
}
```

## Persistence Policy

Default behavior:

```yaml
save_scans: false
save_features: true
save_predictions: true
```

Raw scans may be saved only when a run explicitly opts in:

```yaml
debug:
  save_scans: true
  scan_sample_rate: 0.01
```

This keeps normal dataset creation fast and makes remote synchronization practical.
