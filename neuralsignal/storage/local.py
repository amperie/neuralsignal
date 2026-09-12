from __future__ import annotations

import json

from pathlib import Path
from typing import Any

import pandas as pd

from neuralsignal.storage.manifests import RunManifest, ShardManifest, sha256_file


class LocalFeatureShardWriter:
    def __init__(self, root: str | Path, manifest: RunManifest, compression: str = "zstd") -> None:
        self.root = Path(root)
        self.manifest = manifest
        self.compression = compression
        self.features_dir = self.root / "features"
        if any(self.features_dir.glob("part-*.parquet")):
            raise FileExistsError(f"Feature shards already exist under {self.features_dir}; use a new run directory")
        self.features_dir.mkdir(parents=True, exist_ok=True)

    def write_shard(self, rows: list[dict[str, Any]]) -> ShardManifest:
        index = len(self.manifest.shards)
        path = self.features_dir / f"part-{index:05d}.parquet"
        pd.DataFrame([_parquet_safe(row) for row in rows]).to_parquet(path, index=False, compression=self.compression)
        shard = ShardManifest(
            index=index,
            path=str(path.relative_to(self.root)).replace("\\", "/"),
            rows=len(rows),
            bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
        self.manifest.add_shard(shard)
        self.manifest.write_json(self.root / "manifest.json")
        return shard



def _parquet_safe(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in row.items()
    }
