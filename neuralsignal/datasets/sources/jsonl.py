from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from neuralsignal.datasets.v2 import DatasetExample


class JsonlSource:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def iter_examples(self) -> Iterable[DatasetExample]:
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield _example_from_row(json.loads(line))


def _example_from_row(row: dict) -> DatasetExample:
    return DatasetExample(
        id=str(row.get("id") or row.get("example_id")),
        input=str(row["input"]),
        output=str(row["output"]),
        labels=list(row.get("labels") or []),
        metadata={**{key: value for key, value in row.items()
                     if key not in {"id", "example_id", "input", "output", "labels", "metadata"}},
                  **dict(row.get("metadata") or {})},
    )

