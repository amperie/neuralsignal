"""Shared rendering for external agent trajectories, using the MALT renderer."""
import json

from neuralsignal.datasets.sources.malt_samples import _render_messages
from neuralsignal.datasets.v2 import DatasetExample


def transcript_example(identifier, messages, metadata):
    if not messages or messages[0]["role"] not in {"user", "system"}:
        raise ValueError(f"Missing initial context: {identifier}")
    indices = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
    if not indices:
        raise ValueError(f"Missing agent actions: {identifier}")
    split = indices[-1]
    history, response = _render_messages(messages[:split]), _render_messages(messages[split:])
    if not history or not response:
        raise ValueError(f"Empty transcript: {identifier}")
    return DatasetExample(id=identifier, input=history, output=response, labels=[], metadata=metadata)


def render_content(message):
    parts = [message.get("content") or ""]
    for key in ("tool_calls", "tool_results"):
        if message.get(key):
            parts.append(f"{key}: {json.dumps(message[key], ensure_ascii=False)}")
    if not all(isinstance(part, str) for part in parts):
        raise ValueError("Unexpected non-text message content")
    return "\n".join(part for part in parts if part)
