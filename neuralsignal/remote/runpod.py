from __future__ import annotations

import base64
import json
import os
import urllib.error
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


@dataclass(frozen=True)
class RunPodGpuType:
    id: str
    display_name: str
    memory_gb: int
    stock_status: str
    price_per_hour: float | None
    available_gpu_counts: tuple[int, ...]


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

    for key in ("RUNPOD_API_KEY", "RUNPOD_KEY"):
        env.pop(key, None)
    payload = {
        "name": f"neuralsignal-{job.run_id}",
        "imageName": runpod["image"],
        "gpuCount": int(runpod.get("gpu_count", 1)),
        "containerDiskInGb": int(runpod.get("volume_gb", 75)),
        "volumeMountPath": runpod.get("volume_mount_path", "/workspace"),
        "env": env,
        "dockerStartCmd": ["--run-id", job.run_id],
        "computeType": "GPU",
        "cloudType": runpod.get("cloud_type", "SECURE"),
    }
    if runpod.get("cloud_type"):
        payload["cloudType"] = runpod["cloud_type"]
    if runpod.get("gpu_type_ids"):
        payload["gpuTypeIds"] = list(runpod["gpu_type_ids"])
        payload["gpuTypePriority"] = runpod.get("gpu_type_priority", "custom")
    if runpod.get("container_registry_auth_id"):
        payload["containerRegistryAuthId"] = runpod["container_registry_auth_id"]
    return payload


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
        f"https://rest.runpod.io/v1{path}",
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "User-Agent": "neuralsignal-runpod-client/0.1"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response_body = response.read().decode("utf-8")
            return json.loads(response_body) if response_body.strip() else {}
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"RunPod API {method} {path} failed with HTTP {error.code}: {body}") from error


def launch(payload: dict[str, Any], token: str | None = None) -> dict[str, Any]:
    return api("POST", "/pods", token or _runpod_token(), payload)


def terminate(pod_id: str, token: str | None = None) -> dict[str, Any]:
    return api("DELETE", f"/pods/{pod_id}", token or _runpod_token())


def list_gpu_types(gpu_count: int = 1, secure_cloud: bool = True, token: str | None = None) -> list[RunPodGpuType]:
    query = """
    query GpuTypes($gpuCount: Int!, $secureCloud: Boolean!) {
      gpuTypes {
        id
        displayName
        memoryInGb
        lowestPrice(input: {gpuCount: $gpuCount, secureCloud: $secureCloud}) {
          stockStatus
          uninterruptablePrice
          availableGpuCounts
        }
      }
    }
    """
    response = graphql(query, {"gpuCount": gpu_count, "secureCloud": secure_cloud}, token or _runpod_token())
    gpus = []
    for item in ((response.get("data") or {}).get("gpuTypes") or []):
        price = item.get("lowestPrice") or {}
        gpus.append(RunPodGpuType(
            id=str(item["id"]),
            display_name=str(item.get("displayName") or item["id"]),
            memory_gb=int(item.get("memoryInGb") or 0),
            stock_status=str(price.get("stockStatus") or "None"),
            price_per_hour=price.get("uninterruptablePrice"),
            available_gpu_counts=tuple(int(count) for count in (price.get("availableGpuCounts") or ())),
        ))
    return gpus


def graphql(query: str, variables: dict[str, Any], token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps({"query": query, "variables": variables}).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "User-Agent": "neuralsignal-runpod-client/0.1"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"RunPod GraphQL failed with HTTP {error.code}: {body}") from error
    if result.get("errors"):
        raise RuntimeError(f"RunPod GraphQL failed: {result['errors']}")
    return result


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
    return any(part in lowered for part in ("token", "secret", "password", "access_key", "api_key", "runpod_key"))


