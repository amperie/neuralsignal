"""TRACE's released three-column schema; no inferred task identifiers."""
import json
import re

from neuralsignal.datasets.sources.agent_transcripts import render_content, transcript_example
from neuralsignal.datasets.sources.hf import HuggingFaceSource


class TraceSource(HuggingFaceSource):
    def __init__(self, revision):
        super().__init__("PatronusAI/trace-dataset", "train", normalize_trace_record, revision=revision)


def normalize_trace_record(row):
    if set(row) != {"trajectory_id", "conversation", "label"}:
        raise ValueError(f"TRACE schema changed: {sorted(row)}; inspect grouping fields before proceeding")
    label = row["label"]
    if not isinstance(label, str) or not re.fullmatch(r"0|1\.\d+\.\d+(?:,\s*1\.\d+\.\d+)*", label):
        raise ValueError(f"Unknown TRACE label: {label!r}")
    categories = [] if label == "0" else [item.strip() for item in label.split(",")]
    conversation = json.loads(row["conversation"])
    messages = []
    for message in conversation:
        if message["role"] not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unexpected TRACE role: {message['role']}")
        messages.append({"role": message["role"], "content": render_content(message)})
    yield transcript_example(row["trajectory_id"], messages, {
        "source": "PatronusAI/trace-dataset", "abnormal": int(bool(categories)),
        "categories": categories, "source_label": label,
    })
