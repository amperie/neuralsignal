from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DetectionResult:
    scores: dict[str, float]
    flags: dict[str, bool]
    metadata: dict = field(default_factory=dict)

    @property
    def flagged(self) -> bool:
        return any(self.flags.values())

    def to_dict(self) -> dict:
        return {
            "scores": self.scores,
            "flags": self.flags,
            "flagged": self.flagged,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass(frozen=True)
class BatchDetectionResult:
    results: list[DetectionResult]

    def to_dict(self) -> dict:
        return {"results": [result.to_dict() for result in self.results]}

