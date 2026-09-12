from __future__ import annotations

from typing import Any

from neuralsignal.core.modules.feature_sets.feature_processor import FeatureProcessor
from neuralsignal.core.modules.model_instrumentation import generate_from_batch, load_model
from neuralsignal.core.modules.prompting import wrap_with_prompt
from neuralsignal.datasets.v2 import DatasetExample
from neuralsignal.features.selection import FeatureSetSpec


class ModelFeatureExtractor:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.tokenizer = None
        self.model = None

    def extract_batch(self, examples: list[DatasetExample], specs: list[FeatureSetSpec]) -> list[dict[str, float]]:
        if not examples:
            return []
        self._load()
        prompts = [self._prompt(example) for example in examples]
        generation = self.config.get("generation") or {}
        scans = generate_from_batch(
            prompts,
            self.model,
            self.tokenizer,
            self.config.get("instrumentation") or self.config.get("indirect_instrumentation_config") or {},
            truncation_length=int(generation.get("truncation_length") or 0),
            max_new_tokens=int(generation.get("max_new_tokens") or 16),
        )
        if len(scans) != len(examples):
            raise RuntimeError(f"generation returned {len(scans)} scans for {len(examples)} examples")
        return [self._featurize(scan.data, example, specs) for scan, example in zip(scans, examples)]

    def _load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        model_config = self.config.get("model") or self.config.get("indirect_config")
        if not model_config:
            raise RuntimeError("model extraction requires a model or indirect_config section")
        self.tokenizer, self.model = load_model(model_config)

    def _prompt(self, example: DatasetExample) -> str:
        template = ((self.config.get("prompt") or {}).get("template")) or "{input}\n\n{output}"
        values = {**example.metadata, "input": example.input, "output": example.output}
        return wrap_with_prompt(template, values)

    def _featurize(self, scan: dict[str, Any], example: DatasetExample, specs: list[FeatureSetSpec]) -> dict[str, float]:
        scan = {**scan, "input": example.input, "output": example.output, "metadata": example.metadata}
        row: dict[str, float] = {}
        for spec in specs:
            cfg = {"name": spec.name, **spec.config}
            columns, values = FeatureProcessor(feature_set_configs=[cfg]).featurize_scan(scan)
            row.update({str(column): float(value) for column, value in zip(columns, values)})
        return row

