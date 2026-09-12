# Dataset adapters

[DatasetExample](../neuralsignal/datasets/v2.py) has `id`, `input`, `output`,
`labels: list[str]`, and `metadata: dict`. Sources expose `iter_examples()`.
The [factory](../neuralsignal/datasets/sources/factory.py) supports local JSONL,
generic Hugging Face (`hf`/`huggingface`), MALT samples, and legacy MALT transcripts.
There is no standalone Parquet source adapter. The dataset import CLI only
inspects local JSONL and prints a count.

## JSONL and generic Hugging Face

JSONL requires `input` and `output`, accepts `id` or `example_id`, and preserves
`labels` and `metadata`. Use a list for labels. A singular top-level `label`
is not retained by this adapter. Missing IDs currently stringify to `None`.
Generic Hugging Face uses `name`, `split`, optional `config_name`, and `load_kwargs`;
its default normalizer expects input/output fields and preserves other fields
as metadata. It does not accept a normalizer name from YAML.

Feature collection serializes label lists, rather than generating numeric binary
targets for generic S1 training. Prepare the target column separately.

## MALT samples (default)

```yaml
dataset:
  source: malt
  name: metr-evals/malt-public
  split: public
  config_name: default
  load_kwargs:
    streaming: true
```

`malt`/`malt_samples` emit one example per sample/completion from each run.
IDs are `<run_id>:<sample_index>:<completion_index>`. Input and output are ordered
JSON message lines preserving roles, content, tool calls, and results. The adapter
does not summarize or truncate; model tokenization may truncate later.

Individual `labels` are empty. `metadata` retains `run_labels`, `label_scope:
transcript`, `group_id`, `run_source`, sample/completion indexes and counts, and
other run annotations. The default monitor prompt uses only input/output.

The exact sabotage-scaffold marker beginning
`Instructions:\n- For certain tasks, your aim is insert subtle flaws` is removed
from system/developer input messages and retained in
`excluded_experimental_instructions`. This is an exact rule, not a general
instruction sanitizer. Ordinary task instructions and agent admissions remain.

MALT S1 training averages completions per sample, then samples per run, before
splitting. It rejects missing/nonfinite features, incomplete recorded runs,
duplicate/missing recorded completions, and conflicting run labels/source.
`max_examples` caps normalized examples, not source rows or complete runs; the
smoke limit can therefore produce a dataset unsuitable for training.

## Legacy transcript adapter

`malt_transcripts`, or dataset name `metr-evals/malt-transcripts-public`, chooses
the legacy adapter and defaults to split `transcripts`. It can unwrap graph
messages, concatenate user/environment text, and take the final assistant output.
It does not reconstruct DAG paths or preserve the full conversation. Use the
sample adapter for the current monitoring workflow.

Fixtures: [sample](../neuralsignal/tests/fixtures/malt_simple_112756.json) and
[transcript](../neuralsignal/tests/fixtures/malt_transcript_111846.json).
[Monitoring status](../docs/malt_agent_monitoring_plan.md).
