from __future__ import annotations

from typing import Any

from neuralsignal.datasets.sources.hf import HuggingFaceSource
from neuralsignal.datasets.sources.jsonl import JsonlSource
from neuralsignal.datasets.sources.malt import MaltTranscriptSource
from neuralsignal.datasets.sources.malt_samples import MaltSampleSource


def source_from_config(config: dict[str, Any]):
    dataset = config.get("dataset") or config
    source = dataset.get("source")
    if source == "jsonl" and dataset.get("path"):
        return JsonlSource(dataset["path"])
    if source in {"malt", "malt_samples", "malt_transcripts"}:
        name = str(dataset.get("hf_dataset") or dataset.get("name") or (
            "metr-evals/malt-transcripts-public" if source == "malt_transcripts" else "metr-evals/malt-public"
        ))
        transcripts = name == "metr-evals/malt-transcripts-public" or source == "malt_transcripts"
        source_class = MaltTranscriptSource if transcripts else MaltSampleSource
        return source_class(
            split=str(dataset.get("split") or ("transcripts" if transcripts else "public")),
            name=name,
            config_name=dataset.get("config_name"),
            **dict(dataset.get("load_kwargs") or {}),
        )
    if source in {"hf", "huggingface"}:
        return HuggingFaceSource(
            name=str(dataset["name"]),
            split=str(dataset.get("split") or "train"),
            config_name=dataset.get("config_name"),
            **dict(dataset.get("load_kwargs") or {}),
        )
    raise RuntimeError("dataset.source must be jsonl, malt, or huggingface")
