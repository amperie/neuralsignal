# Dataset Imports

## Goal

Dataset import should normalize external datasets into a small NeuralSignal
example schema. The rest of the system should not know Hugging Face-specific
schemas, transcript DAGs, or source-specific metadata layouts.

## Canonical Example Schema

```python
class DatasetExample:
    id: str
    input: str
    output: str
    labels: list[str]
    metadata: dict
```

The feature collection runner consumes only this schema.

## Source Interface

```python
class DatasetSource:
    def iter_examples(self) -> Iterable[DatasetExample]:
        ...
```

Initial sources:

- Hugging Face datasets
- local JSONL
- local parquet
- MALT transcripts

## MALT sample format (default)

`source: malt` now uses `metr-evals/malt-public`, configuration `default`, split
`public`. Each row represents a run. Each sample contains the complete API input
history and a list of candidate completions; each completion is itself a list of
messages. The adapter emits one example per sample/completion, with ID
`<run_id>:<sample_index>:<completion_index>`.

Input and output are ordered JSON message lines preserving roles, content, tool
names, tool calls, and tool results. Histories are not concatenated across samples.
No context clipping or summarization happens in the adapter. Model-side truncation
settings are unchanged and remain a separate follow-up.

The canonical `labels` list is empty for these individual examples. Run labels live
in `metadata.run_labels`, with `label_scope: transcript` and a stable `group_id`.
`run_source` (prompted/unprompted), `manually_reviewed`, task, model, and sample
metadata are retained independently. They are not inserted into the monitor prompt.

The known sabotage-scaffold block beginning with
`Instructions:\n- For certain tasks, your aim is insert subtle flaws` is removed
from system/developer input messages and retained separately under
`excluded_experimental_instructions`. Ordinary task instructions and agent messages
are preserved. This exact-boundary rule is not a general detector of all possible
experimental prompts; new scaffold variants require explicit rules and tests.

## Run-level S1 training

For these sample snapshots, `label_column: sabotage` selects membership in the run's
label list. S1 training averages completion features within each API sample, then
averages sample features within each run. This is an initial mean-pooling baseline;
it does not assign a sabotage target to every message. Training and test splitting
happen after pooling, so one run cannot appear in both partitions. Test counts,
AUROC, and F1 are also reported separately by `run_source`; AUROC is undefined for
single-class subsets.

Incomplete runs (including a run cut off by the smoke test's eight-example limit)
and duplicate/missing completions are rejected during training. Use a full feature
collection for training, with enough positive and negative runs for stratification.
Mean pooling may dilute isolated events; learned or max pooling can be evaluated
later. The stored snapshots retain the information needed for those alternatives.

```yaml
dataset:
  source: malt
  name: metr-evals/malt-public
  split: public
  config_name: default
  load_kwargs:
    streaming: true
```

See `configs/feature_collection/example_runpod_malt.yaml` for collection and
`configs/training/sabotage_s1.yaml` for training.

## Legacy graph format

`source: malt_transcripts` or an explicit dataset name
`metr-evals/malt-transcripts-public` retains the existing graph adapter and defaults
to split `transcripts`. It unwraps `nodes[].node_data.message`, joins user/environment
text, and takes the final assistant output. It does not reconstruct separate DAG
paths or preserve the full conversation. Use the sample adapter for monitoring.
