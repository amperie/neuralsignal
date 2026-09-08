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

## MALT Transcripts

Primary target:

```text
metr-evals/malt-transcripts-public
```

This dataset is a transcript-oriented source. The importer should flatten source
records into NeuralSignal examples and preserve original transcript metadata in
`metadata`.

The rest of NeuralSignal should not handle transcript DAGs directly. DAG parsing
belongs only in the MALT source adapter.

## MALT Import Plan

1. Load the gated Hugging Face dataset with a configured HF token.
2. Inspect available splits and columns.
3. Convert transcript records into ordered message paths.
4. Emit one or more `DatasetExample` records per transcript.
5. Preserve source ids, labels, model names, task names, and transcript metadata.
6. Write normalized examples as parquet/jsonl for repeatable local and remote
   runs.

## Import CLI

```text
ns dataset import hf metr-evals/malt-transcripts-public --out data/imported/malt
```

Optional direct feature run:

```text
ns remote collect configs/malt_features.yaml
```

## Config Sketch

```yaml
dataset:
  source: hf
  name: metr-evals/malt-transcripts-public
  split: train
  gated: true
  normalizer: malt_transcripts

output:
  format: parquet
  path: data/imported/malt
```

## Practical Note

If the simpler `metr-evals/malt-public` dataset has enough input/output examples
for the first S1 experiments, start there and add transcript DAG support second.
That lowers import risk without changing the architecture.
