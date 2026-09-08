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

