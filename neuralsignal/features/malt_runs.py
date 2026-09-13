"""Pool MALT sample snapshots before applying transcript-level supervision."""
from __future__ import annotations

import json
import logging
import pandas as pd
import numpy as np


def aggregate_malt_runs(data: pd.DataFrame, features: list[str], label: str, targets: list | None = None) -> pd.DataFrame:
    if not np.isfinite(data[features].to_numpy(dtype=float)).all():
        raise ValueError("MALT features must be finite before run aggregation")
    metadata = [json.loads(value) for value in data["metadata_json"]]
    if targets is None and not any(label in meta.get("run_labels", []) for meta in metadata):
        raise ValueError(f"No MALT runs have the requested label: {label}")
    groups: dict[str, list[int]] = {}
    for index, meta in enumerate(metadata):
        if meta.get("source") != "metr-evals/malt-public" or meta.get("label_scope") != "transcript":
            raise ValueError("Run aggregation requires only MALT sample snapshots with transcript labels")
        groups.setdefault(meta["group_id"], []).append(index)
    rows = []
    for group_id, indexes in groups.items():
        meta = metadata[indexes[0]]
        sample_ids = {metadata[index]["sample_index"] for index in indexes}
        if "sample_count" in meta:
            expected_samples = set(range(meta["sample_count"]))
            if sample_ids - expected_samples:
                raise ValueError(f"Invalid sample indexes for MALT run {group_id}")
            if expected_samples - sample_ids:
                logging.getLogger(__name__).warning(
                    "Incomplete MALT run %s: %s/%s samples available; continuing with available samples.",
                    group_id, len(sample_ids), len(expected_samples))
        for sample_id in sample_ids:
            completions = [metadata[index] for index in indexes if metadata[index]["sample_index"] == sample_id]
            if "completion_count" in completions[0]:
                expected = completions[0]["completion_count"]
                actual = [completion["completion_index"] for completion in completions]
                if len(actual) != len(set(actual)):
                    raise ValueError(f"Duplicate completions for {group_id} sample {sample_id}")
                if set(actual) - set(range(expected)):
                    raise ValueError(f"Invalid completion indexes for {group_id} sample {sample_id}")
                if set(actual) != set(range(expected)):
                    logging.getLogger(__name__).warning(
                        "Incomplete completions for %s sample %s: %s/%s available; continuing with available completions.",
                        group_id, sample_id, len(actual), expected)
        for index in indexes[1:]:
            other = metadata[index]
            if set(other["run_labels"]) != set(meta["run_labels"]) or other.get("run_source") != meta.get("run_source"):
                raise ValueError(f"Conflicting transcript metadata for {group_id}")
        target = int(label in meta["run_labels"]) if targets is None else targets[indexes[0]]
        if targets is not None and any(targets[index] != target for index in indexes):
            raise ValueError(f"Conflicting targets within MALT run {group_id}")
        # Average alternative completions within each call, then calls within a run.
        values = data.iloc[indexes][features].astype(float).copy()
        values["_sample_index"] = [metadata[index]["sample_index"] for index in indexes]
        pooled = values.groupby("_sample_index")[features].mean().mean().to_dict()
        rows.append({**pooled, label: target,
                     "group_id": group_id, "run_source": meta.get("run_source", "unknown")})
    return pd.DataFrame(rows, columns=[*features, label, "group_id", "run_source"])
