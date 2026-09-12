from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ShardManifest:
    index: int
    path: str
    rows: int
    sha256: str
    state: str = "written"
    bytes: int = 0

    def validate(self) -> None:
        if self.index < 0:
            raise ValueError("Shard index must be non-negative")
        if self.rows < 0:
            raise ValueError("Shard rows must be non-negative")
        if not self.path:
            raise ValueError("Shard path is required")
        if not self.sha256:
            raise ValueError("Shard sha256 is required")


@dataclass
class RunManifest:
    run_id: str
    dataset: dict[str, Any]
    features: dict[str, Any]
    shards: list[ShardManifest] = field(default_factory=list)
    schema_version: str = "run_manifest.v1"
    state: str = "created"
    rows: dict[str, int | None] = field(default_factory=lambda: {"expected": None, "written": 0, "uploaded": 0})

    def add_shard(self, shard: ShardManifest) -> None:
        shard.validate()
        if any(existing.index == shard.index for existing in self.shards):
            raise ValueError(f"Duplicate shard index: {shard.index}")
        self.shards.append(shard)
        self.rows["written"] = sum(item.rows for item in self.shards)

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required")
        if not self.dataset:
            raise ValueError("dataset is required")
        if not self.features:
            raise ValueError("features is required")
        for shard in self.shards:
            shard.validate()
        if len({shard.index for shard in self.shards}) != len(self.shards):
            raise ValueError("Shard indexes must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "state": self.state,
            "dataset": self.dataset,
            "features": self.features,
            "rows": self.rows,
            "shards": [shard.__dict__ for shard in self.shards],
        }

    def write_json(self, path: str | Path) -> None:
        self.validate()
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def local_shard_path(root: str | Path, shard_path: str) -> Path:
    """Resolve manifest paths without allowing writes or reads outside the run."""
    root = Path(root).resolve()
    relative = Path(shard_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid shard path: {shard_path}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError(f"Invalid shard path: {shard_path}")
    return resolved
