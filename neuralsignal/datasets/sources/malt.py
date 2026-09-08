from __future__ import annotations

import hashlib
from typing import Iterable

from neuralsignal.datasets.sources.hf import HuggingFaceSource
from neuralsignal.datasets.v2 import DatasetExample


class MaltTranscriptSource(HuggingFaceSource):
    def __init__(self, split: str = "train", name: str = "metr-evals/malt-transcripts-public", **load_kwargs) -> None:
        super().__init__(name=name, split=split, normalizer=normalize_malt_record, **load_kwargs)


def normalize_malt_record(row: dict) -> Iterable[DatasetExample]:
    messages = _extract_messages(row)
    if not messages:
        return []

    user_messages = [msg for msg in messages if _role(msg) in {"user", "human"}]
    assistant_messages = [msg for msg in messages if _role(msg) in {"assistant", "model"}]
    if not user_messages or not assistant_messages:
        return []

    input_text = "\n\n".join(_content(msg) for msg in user_messages if _content(msg)).strip()
    output_text = _content(assistant_messages[-1]).strip()
    if not input_text or not output_text:
        return []

    source_id = str(row.get("id") or row.get("transcript_id") or _stable_id(input_text, output_text))
    labels = _extract_labels(row)
    metadata = {k: v for k, v in row.items() if k not in {"messages", "nodes", "transcript"}}
    metadata["source"] = "metr-evals/malt-transcripts-public"

    return [DatasetExample(id=source_id, input=input_text, output=output_text, labels=labels, metadata=metadata)]


def _extract_messages(row: dict) -> list[dict]:
    for key in ("messages", "nodes", "transcript"):
        value = row.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _role(message: dict) -> str:
    return str(message.get("role") or message.get("speaker") or message.get("type") or "").lower()


def _content(message: dict) -> str:
    return str(message.get("content") or message.get("text") or message.get("message") or "")


def _extract_labels(row: dict) -> list[str]:
    labels = row.get("labels") or row.get("label") or row.get("classification") or []
    if isinstance(labels, str):
        return [labels]
    if isinstance(labels, list):
        return [str(label) for label in labels]
    return []


def _stable_id(input_text: str, output_text: str) -> str:
    return hashlib.sha256(f"{input_text}\n{output_text}".encode("utf-8")).hexdigest()[:16]

