from __future__ import annotations

import hashlib
import json
from typing import Iterable

from neuralsignal.datasets.sources.hf import HuggingFaceSource
from neuralsignal.datasets.v2 import DatasetExample


class MaltSampleSource(HuggingFaceSource):
    def __init__(self, split: str = "public", name: str = "metr-evals/malt-public", **load_kwargs) -> None:
        super().__init__(name=name, split=split, normalizer=normalize_malt_samples, **load_kwargs)


def normalize_malt_samples(row: dict) -> Iterable[DatasetExample]:
    """Emit each API completion with its history; labels belong only to the run."""
    metadata = dict(row.get("metadata") or {})
    run_id = metadata.get("run_id")
    if run_id is None:
        run_id = row.get("run_id") or row.get("id")
    if run_id is None:
        run_id = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:16]
    labels = metadata.pop("labels", row.get("labels", [])) or []
    if isinstance(labels, str):
        labels = [label.strip() for label in labels.split(",") if label.strip()]
    samples = row.get("samples") or []
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            continue
        history, excluded = _monitor_history(sample.get("input") or [])
        input_text = _render_messages(history)
        outputs = sample.get("output") or []
        # The public schema is list[completion][message]. Accept flat legacy rows too.
        if outputs and isinstance(outputs[0], dict):
            outputs = [outputs]
        for completion_index, messages in enumerate(outputs):
            output_text = _render_messages(messages)
            if not input_text or not output_text:
                continue
            yield DatasetExample(
                id=f"{run_id}:{sample_index}:{completion_index}",
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
                    "sample_index": sample_index,
                    "completion_index": completion_index,
                    "sample_count": len(samples),
                    "completion_count": len(outputs),
                    "sample_metadata": sample.get("metadata") or {},
                    "excluded_experimental_instructions": excluded,
                },
            )


def _monitor_history(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    history, excluded = [], []
    # Exact boundary in the MALT sabotage scaffold. Do not strip ordinary rules
    # or agent admissions merely because they mention sabotage.
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


def _render_messages(messages: list[dict]) -> str:
    """Render only message content, never labels or experimental metadata."""
    rendered = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        entry = {"role": message.get("role") or "unknown", "content": message.get("content") or ""}
        if message.get("name"):
            entry["name"] = message["name"]
        if message.get("function_call"):
            entry["function_call"] = message["function_call"]
        if message.get("tool_calls"):
            entry["tool_calls"] = message["tool_calls"]
        if message.get("tool_call_id"):
            entry["tool_call_id"] = message["tool_call_id"]
        if entry["content"] or entry.get("function_call") or entry.get("tool_calls"):
            rendered.append(json.dumps(entry, ensure_ascii=False))
    return "\n".join(rendered)
