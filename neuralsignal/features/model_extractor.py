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
        hc = config.get("health_check") or {}
        self._hc_enabled = bool(hc.get("enabled", True))
        self._hc_rows = int(hc.get("rows", 100))
        self._hc_max_constant = float(hc.get("max_constant_fraction", 0.5))
        self._hc_rows_seen: list[dict[str, float]] = []
        self._hc_texts: list[str] = []
        self._hc_done = not self._hc_enabled

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
        rows = [self._featurize(scan.data, example, specs) for scan, example in zip(scans, examples)]
        if not self._hc_done:
            self._hc_rows_seen.extend(rows)
            self._hc_texts.extend(str(scan.data.get("decoded_output") or "") for scan in scans)
            if len(self._hc_rows_seen) >= self._hc_rows:
                check_feature_health(self._hc_rows_seen, self._hc_texts, self._hc_max_constant)
                self._hc_done = True
                self._hc_rows_seen, self._hc_texts = [], []
        return rows

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
            if not columns or len(columns) != len(values):
                raise ValueError(f"Feature set {spec.name} returned empty or misaligned columns and values")
            names = [str(column) for column in columns]
            if len(set(names)) != len(names) or set(names).intersection(row):
                raise ValueError(f"Feature set {spec.name} returned duplicate columns")
            row.update({column: float(value) for column, value in zip(names, values)})
        return row



class FeatureHealthError(RuntimeError):
    pass


def check_feature_health(rows: list[dict[str, float]], texts: list[str], max_constant_fraction: float = 0.5) -> dict[str, float]:
    """Fail fast when collected features look broken (e.g. zeroed activations).

    Raises FeatureHealthError if more than ``max_constant_fraction`` of feature
    columns are constant across ``rows`` or every generated text is empty.
    """
    import logging

    if not rows:
        return {}
    columns = sorted(set().union(*rows))
    constant = 0
    for column in columns:
        values = {row.get(column) for row in rows}
        if len(values) <= 1:
            constant += 1
    fraction = constant / len(columns) if columns else 1.0
    empty = all(not text.strip() for text in texts) if texts else False
    logging.getLogger(__name__).info(
        "feature health check rows=%s columns=%s constant_fraction=%.3f all_generated_empty=%s",
        len(rows), len(columns), fraction, empty)
    problems = []
    if len(rows) > 1 and fraction > max_constant_fraction:
        problems.append(f"{fraction:.1%} of {len(columns)} feature columns are constant across the first {len(rows)} rows")
    if empty:
        problems.append(f"all {len(texts)} generated texts are empty")
    if problems:
        raise FeatureHealthError("Feature health check failed: " + "; ".join(problems) + ". Aborting collection.")
    return {"constant_fraction": fraction}
