import json

from neuralsignal.datasets.sources.jsonl import JsonlSource
from neuralsignal.datasets.sources.malt import normalize_malt_record


def test_jsonl_source_reads_canonical_examples(tmp_path):
    path = tmp_path / "examples.jsonl"
    path.write_text(json.dumps({
        "id": "ex-1",
        "input": "question",
        "output": "answer",
        "labels": ["ok"],
        "metadata": {"split": "train"},
    }) + "\n", encoding="utf-8")

    examples = list(JsonlSource(path).iter_examples())

    assert examples[0].id == "ex-1"
    assert examples[0].input == "question"
    assert examples[0].output == "answer"
    assert examples[0].labels == ["ok"]


def test_malt_normalizer_extracts_user_input_and_last_assistant_output():
    row = {
        "transcript_id": "malt-1",
        "task": "fixture",
        "labels": ["sabotage"],
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "do the task"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "revise"},
            {"role": "assistant", "content": "final answer"},
        ],
    }

    examples = list(normalize_malt_record(row))

    assert len(examples) == 1
    assert examples[0].id == "malt-1"
    assert examples[0].input == "do the task\n\nrevise"
    assert examples[0].output == "final answer"
    assert examples[0].labels == ["sabotage"]
    assert examples[0].metadata["source"] == "metr-evals/malt-transcripts-public"


def test_malt_normalizer_skips_rows_without_input_output_pair():
    assert list(normalize_malt_record({"messages": [{"role": "user", "content": "only input"}]})) == []



def test_malt_normalizer_extracts_samples_rows():
    row = {
        "id": "sample-row",
        "labels": "normal, sabotage",
        "samples": [
            {
                "input": [{"role": "user", "content": "task"}],
                "output": [{"role": "assistant", "content": "answer 1"}, {"role": "assistant", "content": "answer 2"}],
            }
        ],
    }

    examples = list(normalize_malt_record(row))

    assert len(examples) == 1
    assert examples[0].id == "sample-row:0"
    assert examples[0].input == "task"
    assert examples[0].output == "answer 2"
    assert examples[0].labels == ["normal", "sabotage"]


def test_source_factory_builds_malt_source():
    from neuralsignal.datasets.sources.factory import source_from_config
    from neuralsignal.datasets.sources.malt import MaltTranscriptSource

    source = source_from_config({"dataset": {"source": "malt", "split": "public", "config_name": "default"}})

    assert isinstance(source, MaltTranscriptSource)
    assert source.name == "metr-evals/malt-transcripts-public"
    assert source.split == "public"
    assert source.config_name == "default"


def test_huggingface_source_passes_hf_token(monkeypatch):
    from neuralsignal.datasets.sources import hf
    from neuralsignal.datasets.sources.hf import HuggingFaceSource

    calls = {}

    def fake_load_dataset(path, split, **kwargs):
        calls["name"] = path
        calls["split"] = split
        calls["kwargs"] = kwargs
        return [{"id": "a", "input": "i", "output": "o"}]

    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setattr(hf, "_load_dataset", fake_load_dataset)

    examples = list(HuggingFaceSource("repo/name", "train", config_name="default").iter_examples())

    assert examples[0].id == "a"
    assert calls["kwargs"]["token"] == "hf_secret"
    assert calls["kwargs"]["name"] == "default"


