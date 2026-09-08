from __future__ import annotations

from collections.abc import Callable, Iterable

import json
from typing import Any

from neuralsignal.datasets.v2 import DatasetExample
from neuralsignal.features.selection import FeatureSetSpec, materialized_feature_sets
from neuralsignal.storage.local import LocalFeatureShardWriter
from neuralsignal.storage.manifests import RunManifest

FeatureExtractor = Callable[[DatasetExample, FeatureSetSpec], dict[str, float]]
BatchFeatureExtractor = Callable[[list[DatasetExample], list[FeatureSetSpec]], list[dict[str, float]]]


def collect_features(
    examples: Iterable[DatasetExample],
    config: dict[str, Any],
    writer: LocalFeatureShardWriter,
    extractor: FeatureExtractor,
) -> RunManifest:
    return collect_features_batched(
        examples,
        config,
        writer,
        lambda batch, specs: [_extract_one(extractor, example, specs) for example in batch],
        batch_size=1,
    )


def collect_features_batched(
    examples: Iterable[DatasetExample],
    config: dict[str, Any],
    writer: LocalFeatureShardWriter,
    extractor: BatchFeatureExtractor,
    batch_size: int | None = None,
) -> RunManifest:
    shard_size = int(((config.get("storage") or {}).get("shard_size_rows")) or 10000)
    feature_sets = materialized_feature_sets(config)
    resolved_batch_size = batch_size or int(((config.get("generation") or {}).get("batch_size")) or 1)
    buffer: list[dict[str, Any]] = []
    batch: list[DatasetExample] = []
    row_index = 0

    for example in examples:
        batch.append(example)
        if len(batch) >= resolved_batch_size:
            row_index = _collect_batch(batch, feature_sets, writer, extractor, buffer, shard_size, row_index)
            batch = []

    if batch:
        _collect_batch(batch, feature_sets, writer, extractor, buffer, shard_size, row_index)
    if buffer:
        writer.write_shard(buffer)
    return writer.manifest


def _collect_batch(
    batch: list[DatasetExample],
    feature_sets: list[FeatureSetSpec],
    writer: LocalFeatureShardWriter,
    extractor: BatchFeatureExtractor,
    buffer: list[dict[str, Any]],
    shard_size: int,
    row_index: int,
) -> int:
    extracted = extractor(batch, feature_sets)
    if len(extracted) != len(batch):
        raise RuntimeError(f"Extractor returned {len(extracted)} rows for a batch of {len(batch)}")
    for example, features in zip(batch, extracted):
        row = _base_row(writer.manifest.run_id, row_index, example)
        row.update(features)
        buffer.append(row)
        row_index += 1
        if len(buffer) >= shard_size:
            writer.write_shard(buffer)
            buffer.clear()
    return row_index


def _extract_one(extractor: FeatureExtractor, example: DatasetExample, specs: list[FeatureSetSpec]) -> dict[str, float]:
    row: dict[str, float] = {}
    for spec in specs:
        row.update(_prefixed_features(spec, extractor(example, spec)))
    return row


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
