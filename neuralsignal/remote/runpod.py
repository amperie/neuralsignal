from __future__ import annotations

import base64
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neuralsignal.config import load_config


@dataclass(frozen=True)
class RunPodJob:
    run_id: str
    config: dict[str, Any]
    manifest: dict[str, Any]


def build_job(config_path: str | Path, run_id: str, manifest_path: str | Path) -> RunPodJob:
    config = load_config(config_path)
    manifest = load_config(manifest_path)
    return RunPodJob(run_id=run_id, config=config, manifest=manifest)


def build_pod_payload(job: RunPodJob, secrets: dict[str, str] | None = None) -> dict[str, Any]:
    runpod = job.manifest.get("runpod") or {}
    env = dict(job.manifest.get("env") or {})
    env.update(secrets or {})
    env["NEURALSIGNAL_RUN_ID"] = job.run_id
    env["NEURALSIGNAL_FEATURE_CONFIG_B64"] = _encode_config(job.config)

    command = f"--run-id {job.run_id}"
    return {
        "name": f"neuralsignal-{job.run_id}",
        "imageName": runpod["image"],
        "gpuCount": int(runpod.get("gpu_count", 1)),
        "containerDiskInGb": int(runpod.get("volume_gb", 75)),
        "volumeMountPath": runpod.get("volume_mount_path", "/workspace"),
        "env": env,
        "dockerArgs": command,
    }


def load_secrets(path: str | Path) -> dict[str, str]:
    secrets: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, _, value = line.partition("=")
            secrets[key.strip()] = value.strip()
    return secrets


def redacted(payload: dict[str, Any]) -> dict[str, Any]:
    return _redact(payload)


def api(method: str, path: str, token: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        f"https://api.runpod.io/v2{path}",
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def launch(payload: dict[str, Any], token: str | None = None) -> dict[str, Any]:
    return api("POST", "/pods", token or _runpod_token(), payload)


def terminate(pod_id: str, token: str | None = None) -> dict[str, Any]:
    return api("DELETE", f"/pods/{pod_id}", token or _runpod_token())


def _runpod_token() -> str:
    token = os.environ.get("RUNPOD_API_KEY")
    if not token:
        raise RuntimeError("RUNPOD_API_KEY is required")
    return token


def _encode_config(config: dict[str, Any]) -> str:
    return base64.b64encode(json.dumps(config, sort_keys=True).encode("utf-8")).decode("ascii")


def _redact(value):
    if isinstance(value, dict):
        return {key: ("<redacted>" if _is_secret_key(key) else _redact(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in ("token", "secret", "password", "access_key"))


