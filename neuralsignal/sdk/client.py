from __future__ import annotations

from collections.abc import Callable
from typing import Any

from neuralsignal.sdk.results import BatchDetectionResult, DetectionResult

Evaluator = Callable[[list[dict[str, str]], list[str]], list[dict[str, Any]]]


class NeuralSignal:
    def __init__(
        self,
        detectors: list[str] | None = None,
        judge_model: str = "google/flan-t5-large",
        device: str = "auto",
        threshold: float = 0.5,
        evaluator: Evaluator | None = None,
    ) -> None:
        self.detectors = detectors or []
        self.judge_model = judge_model
        self.device = device
        self.threshold = threshold
        self._evaluator = evaluator

    def evaluate(self, input_text: str, output_text: str) -> DetectionResult:
        return self.evaluate_batch([{"input": input_text, "output": output_text}]).results[0]

    def evaluate_batch(self, examples: list[dict[str, str]]) -> BatchDetectionResult:
        if not examples:
            return BatchDetectionResult([])
        rows = self._run_evaluator(examples)
        return BatchDetectionResult([self._result_from_row(row) for row in rows])

    def _run_evaluator(self, examples: list[dict[str, str]]) -> list[dict[str, Any]]:
        if self._evaluator is None:
            raise RuntimeError("No evaluator is configured yet")
        return self._evaluator(examples, self.detectors)

    def _result_from_row(self, row: dict[str, Any]) -> DetectionResult:
        scores = {name: float(score) for name, score in dict(row.get("scores") or {}).items()}
        flags = row.get("flags")
        if flags is None:
            flags = {name: score >= self.threshold for name, score in scores.items()}
        return DetectionResult(scores=scores, flags={name: bool(value) for name, value in dict(flags).items()}, metadata=dict(row.get("metadata") or {}))

