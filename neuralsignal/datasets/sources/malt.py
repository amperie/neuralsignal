from __future__ import annotations

import hashlib
import json
from typing import Iterable

from neuralsignal.datasets.sources.hf import HuggingFaceSource
from neuralsignal.datasets.v2 import DatasetExample


class MaltTranscriptSource(HuggingFaceSource):
    def __init__(self, split: str = "transcripts", name: str = "metr-evals/malt-transcripts-public", **load_kwargs) -> None:
        super().__init__(name=name, split=split, normalizer=normalize_malt_record, **load_kwargs)


def normalize_malt_record(row: dict) -> Iterable[DatasetExample]:
    sample_examples = _examples_from_samples(row)
    if sample_examples:
        return sample_examples

    messages = _extract_messages(row)
    if not messages:
        return []

    user_messages = [msg for msg in messages if _role(msg) in {"user", "human", "environment"}]
    assistant_messages = [msg for msg in messages if _role(msg) in {"assistant", "model", "agent"}]
    if not user_messages or not assistant_messages:
        return []

    input_text = "\n\n".join(_content(msg) for msg in user_messages if _content(msg)).strip()
    output_text = _content(assistant_messages[-1]).strip()
    if not input_text or not output_text:
        return []

    source_id = str(row.get("id") or row.get("transcript_id") or row.get("run_id") or _stable_id(input_text, output_text))
    return [_example(row, source_id, input_text, output_text)]


def _examples_from_samples(row: dict) -> list[DatasetExample]:
    samples = row.get("samples")
    if not isinstance(samples, list):
        return []
    examples = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            continue
        input_text = _messages_text(sample.get("input") or sample.get("inputs"))
        output_text = _messages_text(sample.get("output") or sample.get("outputs"), last_only=True)
        if not input_text or not output_text:
            continue
        source_id = str(row.get("id") or row.get("transcript_id") or row.get("run_id") or _stable_id(input_text, output_text))
        examples.append(_example(row, f"{source_id}:{index}", input_text, output_text))
    return examples


def _example(row: dict, source_id: str, input_text: str, output_text: str) -> DatasetExample:
    metadata = {k: v for k, v in row.items() if k not in {"messages", "nodes", "transcript", "samples"}}
    metadata["source"] = "metr-evals/malt-transcripts-public"
    return DatasetExample(id=source_id, input=input_text, output=output_text, labels=_extract_labels(row), metadata=metadata)


def _extract_messages(row: dict) -> list[dict]:
    for key in ("messages", "nodes", "transcript"):
        value = row.get(key)
        if isinstance(value, list):
            messages = []
            for item in value:
                if isinstance(item, dict):
                    if isinstance(item.get("node_data"), dict):
                        item = item["node_data"]
                    message = item.get("message") if isinstance(item.get("message"), dict) else item
                    messages.append(message)
            return messages
    return []


def _messages_text(value, last_only: bool = False) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return _content(value).strip()
    if not isinstance(value, list):
        return ""
    messages = [item for item in value if isinstance(item, dict)]
    if last_only and messages:
        return _content(messages[-1]).strip()
    return "\n\n".join(_content(message) for message in messages if _content(message)).strip()


def _role(message: dict) -> str:
    return str(message.get("role") or message.get("speaker") or message.get("type") or "").lower()


def _content(message: dict) -> str:
    value = message.get("content") or message.get("text") or message.get("message") or ""
    if not value and isinstance(message.get("function_call"), dict):
        call = message["function_call"]
        arguments = call.get("arguments")
        if call.get("name") == "submit":
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (ValueError, TypeError):
                    pass
            if isinstance(arguments, dict) and arguments.get("submission") is not None:
                submission = arguments["submission"]
                return submission if isinstance(submission, str) else json.dumps(submission, ensure_ascii=False)
        # Keep tool-only assistant actions usable without mistaking them for prose.
        return json.dumps({"function_call": call}, ensure_ascii=False)
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        return str(value.get("text") or value.get("content") or "")
    return str(value)


def _extract_labels(row: dict) -> list[str]:
    labels = row.get("labels") or row.get("label") or row.get("classification") or []
    if isinstance(labels, str):
        return [label.strip() for label in labels.split(",") if label.strip()]
    if isinstance(labels, list):
        return [str(label) for label in labels]
    return []


def _stable_id(input_text: str, output_text: str) -> str:
    return hashlib.sha256(f"{input_text}\n{output_text}".encode("utf-8")).hexdigest()[:16]
