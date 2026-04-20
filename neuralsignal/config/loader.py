from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_ENV_VAR = "NEURALSIGNAL_CONFIG_PATH"
OVERRIDES_ENV_VAR = "NEURALSIGNAL_CONFIG_OVERRIDES"


@dataclass(frozen=True)
class ResolvedConfig:
    data: dict[str, Any]
    detector_registry: dict[str, dict[str, Any]]

    def get_detector_config(
            self,
            detector_name: str,
            overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        if detector_name not in self.detector_registry:
            raise ValueError(f"Could not find detector {detector_name}")
        detector_cfg = deepcopy(self.detector_registry[detector_name])
        if overrides:
            detector_cfg = deep_merge_dicts(detector_cfg, overrides)
        return detector_cfg

    def get_backend_config(
            self,
            overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        backend_cfg = deepcopy(self.data.get("backend_config", {}))
        if overrides:
            backend_cfg = deep_merge_dicts(backend_cfg, overrides)
        return backend_cfg

    def with_overrides(self, overrides: dict[str, Any] | None) -> "ResolvedConfig":
        return resolve_config(base_config=self.data, runtime_overrides=overrides)


def default_sdk_config_path() -> str:
    return str(Path(__file__).resolve().parents[1] / "sdk" / "neuralsignal_sdk.yaml")


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    merged = deepcopy(base)
    if not override:
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_sdk_config(
        config_path: str | None = None,
        overrides: dict[str, Any] | None = None) -> ResolvedConfig:
    return resolve_config(config_path=config_path, runtime_overrides=overrides)


def resolve_config(
        config_path: str | None = None,
        base_config: dict[str, Any] | None = None,
        runtime_overrides: dict[str, Any] | None = None) -> ResolvedConfig:
    if base_config is None:
        loaded_base = _load_yaml_config(config_path)
    else:
        loaded_base = deepcopy(base_config)

    env_overrides = _load_env_overrides()
    merged = deep_merge_dicts(loaded_base, env_overrides)
    merged = deep_merge_dicts(merged, runtime_overrides)

    detector_registry = _normalize_detectors(merged.get("detectors", []))
    merged["detectors"] = list(detector_registry.values())
    _ensure_home_dirs(merged)
    return ResolvedConfig(data=merged, detector_registry=detector_registry)


def _load_yaml_config(config_path: str | None) -> dict[str, Any]:
    resolved_path = config_path or os.getenv(DEFAULT_CONFIG_ENV_VAR) or default_sdk_config_path()
    with open(resolved_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_env_overrides() -> dict[str, Any]:
    raw = os.getenv(OVERRIDES_ENV_VAR)
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {OVERRIDES_ENV_VAR}"
        ) from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{OVERRIDES_ENV_VAR} must contain a JSON object")
    return loaded


def _normalize_detectors(detectors: Any) -> dict[str, dict[str, Any]]:
    if detectors is None:
        return {}
    if isinstance(detectors, dict):
        items = detectors.items()
        registry = {}
        for name, cfg in items:
            detector_cfg = deepcopy(cfg)
            detector_cfg.setdefault("behavior_name", name)
            registry[detector_cfg["behavior_name"]] = detector_cfg
        return registry

    registry = {}
    for detector_cfg in detectors:
        cfg = deepcopy(detector_cfg)
        registry[cfg["behavior_name"]] = cfg
    return registry


def _ensure_home_dirs(config: dict[str, Any]) -> None:
    home = config.get("home")
    if not home:
        return
    home_path = Path(home)
    home_path.mkdir(parents=True, exist_ok=True)
    (home_path / "s1").mkdir(parents=True, exist_ok=True)
