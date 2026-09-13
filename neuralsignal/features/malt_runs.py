"""Prepare one row-level MALT public run per training unit."""
from __future__ import annotations

import json
import pandas as pd
import numpy as np


def aggregate_malt_runs(data: pd.DataFrame, features: list[str], label: str, targets: list | None = None) -> pd.DataFrame:
    if not np.isfinite(data[features].to_numpy(dtype=float)).all():
        raise ValueError("MALT features must be finite before run preparation")
    metadata = [json.loads(value) for value in data["metadata_json"]]
    if targets is None and not any(label in meta.get("run_labels", []) for meta in metadata):
        raise ValueError(f"No MALT runs have the requested label: {label}")
    rows = []
    seen = set()
    for index, meta in enumerate(metadata):
        if meta.get("source") != "metr-evals/malt-public" or meta.get("label_scope") != "transcript":
            raise ValueError("MALT training requires row-level MALT public transcript labels")
        group_id = meta.get("group_id")
        if not group_id:
            raise ValueError("MALT row is missing group_id")
        if group_id in seen:
            raise ValueError(f"Duplicate MALT run {group_id}; collect one row per Hugging Face dataset row")
        seen.add(group_id)
        target = int(label in meta.get("run_labels", [])) if targets is None else targets[index]
        rows.append({**data.iloc[index][features].astype(float).to_dict(),
                     label: target, "group_id": group_id, "run_source": meta.get("run_source", "unknown")})
    return pd.DataFrame(rows, columns=[*features, label, "group_id", "run_source"])
