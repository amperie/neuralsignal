from __future__ import annotations

from typing import Iterable

from neuralsignal.datasets.v2 import DatasetExample


class HuggingFaceSource:
    def __init__(self, name: str, split: str, normalizer=None, **load_kwargs) -> None:
        self.name = name
        self.split = split
        self.normalizer = normalizer or _default_normalizer
        self.load_kwargs = load_kwargs

    def iter_examples(self) -> Iterable[DatasetExample]:
        from datasets import load_dataset

        dataset = load_dataset(self.name, split=self.split, **self.load_kwargs)
        for row in dataset:
            yield from self.normalizer(dict(row))


def _default_normalizer(row: dict) -> Iterable[DatasetExample]:
    yield DatasetExample(
        id=str(row.get("id") or row.get("example_id")),
        input=str(row["input"]),
        output=str(row["output"]),
        labels=list(row.get("labels") or []),
        metadata={k: v for k, v in row.items() if k not in {"id", "example_id", "input", "output", "labels"}},
    )

