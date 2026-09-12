from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DatasetExample:
    id: str
    input: str
    output: str
    labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "input": self.input,
            "output": self.output,
            "labels": self.labels,
            "metadata": self.metadata,
        }

