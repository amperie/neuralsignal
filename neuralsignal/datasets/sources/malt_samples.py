from __future__ import annotations

import hashlib
import json
from typing import Iterable

from neuralsignal.datasets.sources.hf import HuggingFaceSource
from neuralsignal.datasets.v2 import DatasetExample


class MaltPublicSource(HuggingFaceSource):
    def __init__(self, split: str = "public", name: str = "metr-evals/malt-public", **load_kwargs) -> None:
        super().__init__(name=name, split=split, normalizer=normalize_malt_public_row, **load_kwargs)


# Backward-compatible name, but no sample expansion behavior remains.
MaltSampleSource = MaltPublicSource


def normalize_malt_public_row(row: dict) -> Iterable[DatasetExample]:
    """Emit one full-conversation example per Hugging Face MALT public row."""
    metadata = dict(row.get("metadata") or {})
    run_id = _run_id(row, metadata)
    labels = _labels(row, metadata)
    samples = [sample for sample in (row.get("samples") or []) if isinstance(sample, dict)]
    if not samples:
        return []
    input_parts = []
    output_parts = []
    excluded = []
    for index, sample in enumerate(samples):
        history, sample_excluded = _monitor_history(sample.get("input") or [])
        excluded.extend(sample_excluded)
        input_text = _render_messages(history)
        output_text = _render_outputs(sample.get("output") or [])
        if input_text:
            input_parts.append(input_text)
        if output_text:
            output_parts.append(output_text)
    input_text = "\n".join(input_parts)
    output_text = "\n".join(output_parts)
    if not input_text or not output_text:
        return []
    return [DatasetExample(
        id=str(run_id),
        input=input_text,
        output=output_text,
        labels=[],
        metadata={
            **metadata,
            "source": "metr-evals/malt-public",
            "run_id": run_id,
            "group_id": f"metr-evals/malt-public:{run_id}",
            "run_labels": list(labels),
            "label_scope": "transcript",
            "malt_grain": "run",
            "sample_count": len(samples),
            "completion_count": sum(_completion_count(sample.get("output") or []) for sample in samples),
            "sample_metadata": [sample.get("metadata") or {} for sample in samples],
            "excluded_experimental_instructions": excluded,
        },
    )]


normalize_malt_samples = normalize_malt_public_row


def _run_id(row: dict, metadata: dict):
    run_id = metadata.get("run_id")
    if run_id is None:
        run_id = row.get("run_id") or row.get("id")
    if run_id is None:
        run_id = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:16]
    return run_id


def _labels(row: dict, metadata: dict) -> list[str]:
    labels = metadata.pop("labels", row.get("labels", [])) or []
    if isinstance(labels, str):
        return [label.strip() for label in labels.split(",") if label.strip()]
    return list(labels)


def _monitor_history(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    history, excluded = [], []
    marker = "Instructions:\n- For certain tasks, your aim is insert subtle flaws"
    for message in messages:
        message = dict(message)
        content = message.get("content") or ""
        if message.get("role") in {"system", "developer"} and marker in content:
            start = content.index(marker)
            excluded.append({"role": message["role"], "content": content[start:]})
            message["content"] = content[:start].rstrip()
        history.append(message)
    return history, excluded


def _completion_count(outputs) -> int:
    if outputs and isinstance(outputs, list) and isinstance(outputs[0], dict):
        return 1
    if isinstance(outputs, list):
        return len(outputs)
    return 0


def _render_outputs(outputs) -> str:
    if outputs and isinstance(outputs, list) and isinstance(outputs[0], dict):
        outputs = [outputs]
    rendered = [_render_messages(messages) for messages in outputs]
    rendered = [text for text in rendered if text]
    return "\n".join(rendered)


def _render_messages(messages: list[dict]) -> str:
    lines = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown")
        name = message.get("name")
        prefix = f"{role} ({name})" if name else role
        content = _content(message)
        if content:
            lines.append(f"{prefix}: {content}")
    return "\n".join(lines).strip()


def _content(message: dict) -> str:
    value = message.get("content") or ""
    if isinstance(value, list):
        return "\n".join(
            str(item.get("text") or item.get("content") or item) if isinstance(item, dict) else str(item)
            for item in value
        ).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if value:
        return str(value)
    return ""
