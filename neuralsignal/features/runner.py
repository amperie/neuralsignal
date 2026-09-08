from __future__ import annotations

from collections.abc import Callable, Iterable

import json
from typing import Any

from neuralsignal.datasets.v2 import DatasetExample
from neuralsignal.features.selection import FeatureSetSpec, materialized_feature_sets
from neuralsignal.storage.local import LocalFeatureShardWriter
from neuralsignal.storage.manifests import RunManifest

FeatureExtractor = Callable[[DatasetExample, FeatureSetSpec], dict[str, float]]


def collect_features(
    examples: Iterable[DatasetExample],
    config: dict[str, Any],
    writer: LocalFeatureShardWriter,
    extractor: FeatureExtractor,
) -> RunManifest:
    shard_size = int(((config.get("storage") or {}).get("shard_size_rows")) or 10000)
    feature_sets = materialized_feature_sets(config)
    buffer: list[dict[str, Any]] = []

    for index, example in enumerate(examples):
        row = _base_row(writer.manifest.run_id, index, example)
        for spec in feature_sets:
            row.update(_prefixed_features(spec, extractor(example, spec)))
        buffer.append(row)
        if len(buffer) >= shard_size:
            writer.write_shard(buffer)
            buffer = []

    if buffer:
        writer.write_shard(buffer)
    return writer.manifest


def _base_row(run_id: str, row_index: int, example: DatasetExample) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "example_id": example.id,
        "row_index": row_index,
        "input": example.input,
        "output": example.output,
        "labels_json": json.dumps(example.labels, sort_keys=True),
        "metadata_json": json.dumps(example.metadata, sort_keys=True),
    }


def _prefixed_features(spec: FeatureSetSpec, features: dict[str, float]) -> dict[str, float]:
    prefix = spec.column_prefix()
    return {
        (name if name.startswith(prefix) else f"{prefix}{name}"): float(value)
        for name, value in features.items()
    }

