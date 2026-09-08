from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FeatureSetSpec:
    name: str
    enabled: bool = True
    version: str = "v1"
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FeatureSetSpec":
        if "name" not in config:
            raise ValueError("Feature set config requires a name")
        return cls(
            name=str(config["name"]),
            enabled=bool(config.get("enabled", True)),
            version=str(config.get("version", "v1")),
            config=dict(config.get("config") or {}),
        )

    def column_prefix(self) -> str:
        return f"{self.name}__"

    def to_manifest(self, columns: list[str] | None = None) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "version": self.version,
            "config": self.config,
            "columns": columns or [],
        }


def materialized_feature_sets(config: dict[str, Any]) -> list[FeatureSetSpec]:
    entries = ((config.get("features") or {}).get("materialize") or [])
    specs = [FeatureSetSpec.from_config(entry) for entry in entries]
    return [spec for spec in specs if spec.enabled]

