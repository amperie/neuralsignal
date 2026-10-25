"""FAR.AI FIBS (Falsehoods in Broad Settings) adapter.

Hugging Face dataset ``AlignmentResearch/fibs-v1`` (config ``default``) has four
splits named by quadrant: ``honest_honest``, ``honest_deceptive``,
``deceptive_honest`` and ``deceptive_deceptive``. Each row is one transcript
(``messages_json``) ending in a final assistant ``completion``. The row label is
``judge_verdict``, produced by an AI judge model rather than by humans.

Each row becomes one example (or several overlapping chunks of one example) whose
input is the ordered message history and whose output is the final response,
rendered as ``role: content`` lines in the same style as the MALT adapter. Labels
are kept in metadata and are never inserted into the judge prompt.
"""
from __future__ import annotations

import gc

import hashlib
import json
from typing import Iterable

from neuralsignal.datasets.sources.hf import HuggingFaceSource
from neuralsignal.datasets.v2 import DatasetExample

FIBS_DATASET = "AlignmentResearch/fibs-v1"
FIBS_SPLITS = ("honest_honest", "honest_deceptive", "deceptive_honest", "deceptive_deceptive")

# Raw judge verdict strings seen in fibs-v1, compared case-insensitively.
DECEPTIVE_VERDICTS = frozenset({"lie", "untruthful", "deceptive", "dishonest"})
HONEST_VERDICTS = frozenset({"honest", "truthful"})

# Columns copied into metadata. Free-text judge justifications and the original
# pre-paraphrase completion are deliberately omitted: they restate the label.
_METADATA_COLUMNS = (
    "judge_verdict", "judge_confidence", "judge_model_informed", "blind_judge_verdict_primary",
    "blind_judge_model_primary", "human_verified", "quadrant", "deception_type", "follow_up_category",
    "pair_id", "pair_type", "language", "an_generation_method", "an_generation_model", "an_paraphrased",
    "reasoning_in_context", "_v32_source_content_md5", "_v32_lane_attributed",
)


class FibsSource:
    """Iterate one or more FIBS quadrant splits as normalized examples."""

    def __init__(self, split: str | list[str] | tuple[str, ...] = "all", name: str = FIBS_DATASET,
                 config_name: str | None = "default", chunk_chars: int = 0, chunk_overlap_chars: int = 0,
                 max_examples_per_split: int | None = None, **load_kwargs) -> None:
        self.splits = resolve_fibs_splits(split)
        self.name = name
        self.config_name = config_name
        self.chunk_chars = int(chunk_chars or 0)
        self.chunk_overlap_chars = int(chunk_overlap_chars or 0)
        if self.chunk_chars < 0 or self.chunk_overlap_chars < 0:
            raise ValueError("FIBS chunk sizes must not be negative")
        if self.chunk_chars and self.chunk_overlap_chars >= self.chunk_chars:
            raise ValueError("dataset.chunk_overlap_chars must be smaller than dataset.chunk_chars")
        if max_examples_per_split is not None and (isinstance(max_examples_per_split, bool)
                                                   or int(max_examples_per_split) <= 0):
            raise ValueError("dataset.max_examples_per_split must be a positive integer")
        self.max_examples_per_split = max_examples_per_split
        self.load_kwargs = load_kwargs

    def iter_examples(self) -> Iterable[DatasetExample]:
        for split in self.splits:
            normalizer = lambda row, split=split: normalize_fibs_row(
                row, split=split, chunk_chars=self.chunk_chars, chunk_overlap_chars=self.chunk_overlap_chars)
            source = HuggingFaceSource(self.name, split=split, normalizer=normalizer,
                                       config_name=self.config_name, **self.load_kwargs)
            rows = 0
            last_row = None
            examples = source.iter_examples()
            try:
                for example in examples:
                    row_id = example.metadata["fibs_row_id"]
                    if row_id != last_row:
                        if self.max_examples_per_split and rows >= self.max_examples_per_split:
                            break
                        rows, last_row = rows + 1, row_id
                    yield example
            finally:
                # Stopping a streaming HF iterator early and leaving it to interpreter
                # shutdown aborts the process (SIGABRT, exit 134), so close it now.
                _close_iterator(examples)

    def __iter__(self):
        return iter(self.iter_examples())


def _close_iterator(iterator) -> None:
    close = getattr(iterator, "close", None)
    if close is not None:
        close()
    gc.collect()


def resolve_fibs_splits(split) -> list[str]:
    if split in (None, "", "all"):
        return list(FIBS_SPLITS)
    splits = [split] if isinstance(split, str) else list(split)
    unknown = [item for item in splits if item not in FIBS_SPLITS]
    if unknown or not splits:
        raise ValueError(f"Unknown FIBS split(s) {unknown}; choose from {', '.join(FIBS_SPLITS)} or 'all'")
    return splits


def fibs_verdict_label(verdict) -> int | None:
    """Map a raw ``judge_verdict`` to 1 (deceptive), 0 (honest) or None (missing/other)."""
    value = str(verdict or "").strip().lower()
    if value in DECEPTIVE_VERDICTS:
        return 1
    if value in HONEST_VERDICTS:
        return 0
    return None


def normalize_fibs_row(row: dict, split: str | None = None, chunk_chars: int = 0,
                       chunk_overlap_chars: int = 0) -> list[DatasetExample]:
    messages = _messages(row.get("messages_json"))
    history_lines = _render_lines(messages)
    output_text = f"assistant: {str(row.get('completion') or '').strip()}".strip()
    if output_text == "assistant:" or not history_lines:
        return []
    split = split or row.get("quadrant")
    content_md5 = str(row.get("_v32_source_content_md5") or "")
    row_id = _row_id(split, content_md5, row)
    metadata = {key: row.get(key) for key in _METADATA_COLUMNS if key in row}
    history = "\n".join(history_lines)
    metadata.update({
        "source": FIBS_DATASET,
        "fibs_split": split,
        "family": row.get("source"),
        "fibs_row_id": row_id,
        "group_id": f"{FIBS_DATASET}:{content_md5 or row_id}",
        "group_keys": [key for key in (f"md5:{content_md5}" if content_md5 else "",
                                       f"pair:{row.get('pair_id')}" if row.get("pair_id") else "") if key],
        "label_scope": "final_response",
        "label_origin": "ai_judge",
        "deceptive": fibs_verdict_label(row.get("judge_verdict")),
        "message_count": len(messages),
        "step_position": len(messages) + 1,
        "transcript_chars": len(history) + len(output_text),
        "has_tools": bool(row.get("tools")),
    })
    chunks = chunk_history(history_lines, chunk_chars, chunk_overlap_chars) if chunk_chars else [history]
    examples = []
    for index, chunk in enumerate(chunks):
        example_id = row_id if len(chunks) == 1 else f"{row_id}#c{index}"
        examples.append(DatasetExample(
            id=example_id, input=chunk, output=output_text, labels=[],
            metadata={**metadata, "chunk_index": index, "chunk_count": len(chunks)}))
    return examples


def chunk_history(lines: list[str], chunk_chars: int, overlap_chars: int) -> list[str]:
    """Split ordered message lines into overlapping windows of at most ``chunk_chars``.

    Windows break on message boundaries; a single message longer than the window is
    split by characters. The last window always ends with the newest message.
    """
    pieces = []
    for line in lines:
        while len(line) > chunk_chars:
            pieces.append(line[:chunk_chars])
            line = line[chunk_chars - overlap_chars:] if overlap_chars else line[chunk_chars:]
        if line:
            pieces.append(line)
    chunks, start = [], 0
    while start < len(pieces):
        end, size = start, 0
        while end < len(pieces) and size + len(pieces[end]) + (1 if end > start else 0) <= chunk_chars:
            size += len(pieces[end]) + (1 if end > start else 0)
            end += 1
        end = max(end, start + 1)
        chunks.append("\n".join(pieces[start:end]))
        if end >= len(pieces):
            break
        back, kept = end, 0
        while back > start + 1 and kept + len(pieces[back - 1]) <= overlap_chars:
            kept += len(pieces[back - 1]) + 1
            back -= 1
        start = back
    return chunks


def _messages(value) -> list[dict]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _render_lines(messages: list[dict]) -> list[str]:
    lines = []
    for message in messages:
        role = str(message.get("role") or "unknown")
        name = message.get("name")
        prefix = f"{role} ({name})" if name else role
        content = _content(message)
        if content:
            lines.append(f"{prefix}: {content}")
    return lines


def _content(message: dict) -> str:
    value = message.get("content")
    if isinstance(value, list):
        text = "\n".join(str(item.get("text") or item.get("content") or "") if isinstance(item, dict) else str(item)
                         for item in value).strip()
    elif isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value or "").strip()
    calls = message.get("tool_calls")
    if calls:
        text = "\n".join(part for part in (text, f"[tool calls] {json.dumps(calls, ensure_ascii=False)}") if part)
    return text


def _row_id(split, content_md5: str, row: dict) -> str:
    key = "\n".join(str(part) for part in (split, content_md5, row.get("follow_up_question"), row.get("completion"),
                                           row.get("messages_json") if not content_md5 else ""))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
