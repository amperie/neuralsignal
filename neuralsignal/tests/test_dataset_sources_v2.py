import json
from pathlib import Path

import pytest

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

    source = source_from_config({"dataset": {"source": "malt", "split": "transcripts", "config_name": "default"}})

    assert isinstance(source, MaltTranscriptSource)
    assert source.name == "metr-evals/malt-transcripts-public"
    assert source.split == "transcripts"
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




def test_malt_real_transcript_normalizes_submission():
    # Real Hugging Face transcripts split row; no network needed for regression.
    row = json.loads((Path(__file__).parent / "fixtures" / "malt_transcript_111846.json").read_text())
    examples = list(normalize_malt_record(row))

    assert len(examples) == 1
    example = examples[0]
    assert example.id == "111846"
    assert "Your task is to find out what the right password is." in example.input
    assert example.output == "persimmoniously"
    assert example.labels == ["normal"]
    assert example.metadata["task_id"] == "password_check/6"
    assert example.metadata["run_id"] == 111846


@pytest.mark.parametrize("arguments, expected", [
    ('{"submission":"answer"}', "answer"),
    ({"submission": "answer"}, "answer"),
    ('{"submission":0}', "0"),
])
def test_malt_submit_arguments(arguments, expected):
    row = {"messages": [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "function_call": {"name": "submit", "arguments": arguments}},
    ]}
    assert list(normalize_malt_record(row))[0].output == expected


@pytest.mark.parametrize("call", [
    {"name": "bash", "arguments": '{"command":"ls"}'},
    {"name": "submit", "arguments": "invalid json"},
])
def test_malt_preserves_other_or_malformed_tool_calls(call):
    row = {"messages": [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "function_call": call},
    ]}
    assert json.loads(list(normalize_malt_record(row))[0].output) == {"function_call": call}


def test_malt_default_split_and_real_row_loading(monkeypatch):
    from neuralsignal.datasets.sources import hf
    from neuralsignal.datasets.sources.factory import source_from_config
    from neuralsignal.datasets.sources.malt import MaltTranscriptSource

    row = json.loads((Path(__file__).parent / "fixtures" / "malt_transcript_111846.json").read_text())
    def fake_load_dataset(path, split, **kwargs):
        assert path == "metr-evals/malt-transcripts-public"
        assert split == "transcripts"
        return [row]

    monkeypatch.setattr(hf, "_load_dataset", fake_load_dataset)
    assert MaltTranscriptSource().split == "transcripts"
    source = source_from_config({"dataset": {"source": "malt"}})
    assert list(source.iter_examples())[0].output == "persimmoniously"


@pytest.mark.parametrize("filename", ["smoke_runpod_malt.yaml", "example_runpod_malt.yaml"])
def test_malt_runpod_configs_use_transcripts_split(filename):
    import yaml
    from neuralsignal.datasets.sources.factory import source_from_config

    path = Path(__file__).resolve().parents[2] / "configs" / "feature_collection" / filename
    config = yaml.safe_load(path.read_text())
    assert source_from_config(config).split == "transcripts"
