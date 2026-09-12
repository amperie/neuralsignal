from __future__ import annotations

import os
from typing import Iterable

from neuralsignal.datasets.v2 import DatasetExample


class HuggingFaceSource:
    def __init__(self, name: str, split: str, normalizer=None, config_name: str | None = None, **load_kwargs) -> None:
        self.name = name
        self.split = split
        self.config_name = config_name
        self.normalizer = normalizer or _default_normalizer
        self.load_kwargs = load_kwargs

    def iter_examples(self) -> Iterable[DatasetExample]:
        load_kwargs = dict(self.load_kwargs)
        if self.config_name:
            load_kwargs["name"] = self.config_name
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token and "token" not in load_kwargs and "use_auth_token" not in load_kwargs:
            load_kwargs["token"] = token
        dataset = _load_dataset(self.name, split=self.split, **load_kwargs)
        for row in dataset:
            yield from self.normalizer(dict(row))


def _load_dataset(*args, **kwargs):
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def _default_normalizer(row: dict) -> Iterable[DatasetExample]:
    yield DatasetExample(
        id=str(row.get("id") or row.get("example_id")),
        input=str(row["input"]),
        output=str(row["output"]),
        labels=list(row.get("labels") or []),
        metadata={k: v for k, v in row.items() if k not in {"id", "example_id", "input", "output", "labels"}},
    )
