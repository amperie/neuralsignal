from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path) -> dict[str, str]:
    values: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return values
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def apply_env(values: dict[str, str]) -> None:
    for key, value in values.items():
        os.environ.setdefault(key, value)

