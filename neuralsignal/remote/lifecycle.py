from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
from typing import Protocol

from neuralsignal.console import color
from neuralsignal.config import load_config
from neuralsignal.config.env import apply_env, load_env_file
from neuralsignal.remote.runpod import RunPodGpuType, RunPodJob, build_pod_payload, launch, list_gpu_types, load_secrets, redacted, terminate, get_pod, ssh_command
from neuralsignal.remote.terraform import s3_settings_from_terraform
from neuralsignal.storage.bundle import delete_bundle, download_bundle, unpack_bundle
from neuralsignal.storage.s3 import Boto3ObjectStore, ObjectStore, exists, s3_join, upload_directory
from neuralsignal.training import train_s1

logger = logging.getLogger(__name__)


class RemoteCollectCancelled(Exception):
    def __init__(self, pod_id: str, terminated: bool):
        self.pod_id = pod_id
        self.terminated = terminated
        message = (f"Cancelled. Pod {pod_id} terminated." if terminated else
                   f"Cancelled, but could not confirm termination of pod {pod_id}. "
                   "Terminate it in the RunPod console.")
        super().__init__(message)


class RunPodApi(Protocol):
    def launch(self, payload: dict) -> dict:
        ...

    def get_pod(self, pod_id: str) -> dict:
        ...

    def terminate(self, pod_id: str) -> dict:
        ...

    def list_gpu_types(self, gpu_count: int, secure_cloud: bool) -> list[RunPodGpuType]:
        ...


@dataclass
class DefaultRunPodApi:
    token: str | None = None

    def launch(self, payload: dict) -> dict:
        return launch(payload, self.token)

    def get_pod(self, pod_id: str) -> dict:
        return get_pod(pod_id, self.token)

    def terminate(self, pod_id: str) -> dict:
        return terminate(pod_id, self.token)

    def list_gpu_types(self, gpu_count: int, secure_cloud: bool) -> list[RunPodGpuType]:
        return list_gpu_types(gpu_count, secure_cloud, self.token)


@dataclass(frozen=True)
class RemoteCollectResult:
    run_id: str
    pod_id: str
    bundle_uri: str
    target_dir: str
    training_metrics: dict | None


def remote_collect_lifecycle(
    config_path: str | Path,
    runpod_manifest_path: str | Path,
    run_id: str,
    target_dir: str | Path,
    env_file: str | Path | None = ".env",
    secrets_file: str | Path | None = None,
    terraform_dir: str | Path | None = "infra/terraform/s3-handoff",
    train_config_path: str | Path | None = None,
    minio_uri: str | None = None,
    poll_seconds: float = 30,
    timeout_seconds: float | None = None,
    runpod_api: RunPodApi | None = None,
    store: ObjectStore | None = None,
    minio_store: ObjectStore | None = None,
    dry_run: bool = False,
    gpu_vram_gb: int | None = None,
    gpu_id: str | None = None,
    yes: bool = False,
) -> RemoteCollectResult | dict:
    if env_file:
        logger.info("loading environment file path=%s", env_file)
        apply_env(load_env_file(env_file))
    logger.info("loading feature config path=%s", config_path)
    config = load_config(config_path)
    logger.info("loading runpod manifest path=%s", runpod_manifest_path)
    manifest = load_config(runpod_manifest_path)
    secrets = _forwarded_hf_env()
    if terraform_dir and Path(terraform_dir).exists():
        logger.info("loading terraform S3 handoff outputs dir=%s", terraform_dir)
        secrets.update(s3_settings_from_terraform(terraform_dir))
    if secrets_file:
        logger.info("loading forwarded secrets file path=%s", secrets_file)
        secrets.update(load_secrets(secrets_file))
    _fill_s3_defaults(config, secrets)

    bundle_uri = _bundle_uri(config, run_id)
    logger.info("remote collect prepared run_id=%s bundle_uri=%s", run_id, bundle_uri)
    runpod_api = runpod_api or DefaultRunPodApi()
    manifest = _with_selected_gpu(manifest, runpod_api, gpu_vram_gb, gpu_id, yes=yes)
    payload = build_pod_payload(RunPodJob(run_id, config, manifest), secrets)
    logger.info("runpod payload ready name=%s image=%s gpu_count=%s gpu_type_ids=%s", payload.get("name"), payload.get("imageName"), payload.get("gpuCount"), payload.get("gpuTypeIds"))
    if dry_run:
        logger.info("dry run requested; not launching RunPod pod")
        return redacted({"payload": payload, "bundle_uri": bundle_uri})

    store = store or Boto3ObjectStore(endpoint_url=os.environ.get("NEURALSIGNAL_S3_ENDPOINT_URL"))
    logger.info("launching RunPod pod")
    pod_id = str(runpod_api.launch(payload)["id"])
    logger.info("RunPod pod launched pod_id=%s", pod_id)
    training_metrics = None
    interrupted = False
    try:
        _wait_for_bundle(store, bundle_uri, poll_seconds, timeout_seconds,
                         monitor=PodProgress(runpod_api, pod_id))
    except KeyboardInterrupt:
        logger.warning("Interrupted; cleaning up pod %s", pod_id)
        interrupted = True
    except Exception:
        logger.error("Remote collection failed for pod %s; attempting termination", pod_id)
        raise
    finally:
        logger.info("terminating RunPod pod pod_id=%s", pod_id)
        try:
            runpod_api.terminate(pod_id)
        except (Exception, KeyboardInterrupt):
            if interrupted:
                raise RemoteCollectCancelled(pod_id, terminated=False) from None
            raise
        logger.info("RunPod pod terminated pod_id=%s", pod_id)

    if interrupted:
        raise RemoteCollectCancelled(pod_id, terminated=True)

    if not _bundle_available(store, bundle_uri):
        raise RuntimeError(f"Remote run finished without bundle; bundle is not available: {bundle_uri}")

    target = Path(target_dir) / run_id
    zip_path = Path(target_dir) / f"{run_id}.zip"
    logger.info("downloading bundle uri=%s path=%s", bundle_uri, zip_path)
    download_bundle(store, bundle_uri, zip_path)
    logger.info("deleting remote handoff bundle uri=%s", bundle_uri)
    delete_bundle(store, bundle_uri)
    logger.info("unpacking bundle path=%s target=%s", zip_path, target)
    unpack_bundle(zip_path, target)
    if minio_uri:
        (target / "minio-dataset-uri.txt").write_text(minio_uri, encoding="utf-8")
        minio_store = minio_store or _minio_store()
        logger.info("mirroring unpacked run to minio uri=%s", minio_uri)
        upload_directory(minio_store, target, minio_uri)
    if train_config_path:
        logger.info("starting local S1 training config=%s dataset=%s", train_config_path, target)
        training_metrics = _train_s1_from_config(train_config_path, target, minio_uri)
        logger.info("local S1 training completed metrics=%s", training_metrics)
    logger.info("remote collect lifecycle completed run_id=%s target=%s", run_id, target)
    return RemoteCollectResult(run_id, pod_id, bundle_uri, str(target), training_metrics)




def _with_selected_gpu(
    manifest: dict,
    runpod_api: RunPodApi,
    gpu_vram_gb: int | None,
    gpu_id: str | None,
    yes: bool = False,
) -> dict:
    if gpu_vram_gb is not None and gpu_vram_gb <= 0:
        raise ValueError("GPU VRAM must be positive")
    if not gpu_vram_gb and not gpu_id:
        return manifest
    updated = deepcopy(manifest)
    runpod = dict(updated.get("runpod") or {})
    if gpu_id:
        logger.info("using explicit RunPod GPU id=%s", gpu_id)
        runpod["gpu_type_ids"] = [gpu_id]
        runpod["gpu_type_priority"] = "custom"
        updated["runpod"] = runpod
        return updated

    gpu_count = int(runpod.get("gpu_count", 1))
    cloud_type = str(runpod.get("cloud_type", "SECURE")).upper()
    secure_cloud = cloud_type != "COMMUNITY"
    logger.info("querying RunPod GPUs requested_vram_gb=%s gpu_count=%s secure_cloud=%s", gpu_vram_gb, gpu_count, secure_cloud)
    choices = _available_gpus(runpod_api.list_gpu_types(gpu_count, secure_cloud), gpu_vram_gb or 0, gpu_count)
    if not choices:
        low, high = _vram_window(gpu_vram_gb or 0)
        raise RuntimeError(f"No available RunPod GPUs found within {low:.1f}-{high:.1f}GB VRAM")
    selected = _prompt_gpu_choice(choices, yes=yes)
    logger.info("selected RunPod GPU id=%s name=%s memory_gb=%s", selected.id, selected.display_name, selected.memory_gb)
    runpod["gpu_type_ids"] = [selected.id]
    runpod["gpu_type_priority"] = "custom"
    updated["runpod"] = runpod
    return updated


def _available_gpus(gpus: list[RunPodGpuType], requested_vram_gb: int, gpu_count: int) -> list[RunPodGpuType]:
    low, high = _vram_window(requested_vram_gb)

    def is_available(gpu: RunPodGpuType) -> bool:
        if not low <= gpu.memory_gb <= high:
            return False
        if gpu.stock_status.lower() in {"", "none", "unavailable"}:
            return False
        return not gpu.available_gpu_counts or gpu_count in gpu.available_gpu_counts

    return sorted(
        [gpu for gpu in gpus if is_available(gpu)],
        key=lambda gpu: (gpu.price_per_hour is None, gpu.price_per_hour if gpu.price_per_hour is not None else float("inf"), gpu.memory_gb, gpu.display_name),
    )


def _vram_window(requested_vram_gb: int) -> tuple[float, float]:
    spread = requested_vram_gb * 0.25
    return requested_vram_gb - spread, requested_vram_gb + spread


def _prompt_gpu_choice(choices: list[RunPodGpuType], yes: bool = False) -> RunPodGpuType:
    print("Available RunPod GPUs within 25% of the requested VRAM:")
    for index, gpu in enumerate(choices, start=1):
        price = "unknown" if gpu.price_per_hour is None else f"${gpu.price_per_hour:.3f}/hr"
        counts = ",".join(str(count) for count in gpu.available_gpu_counts) or "unknown"
        model = color(gpu.display_name, "cyan")
        price = color(price, "green" if gpu.price_per_hour is not None else "dim")
        details = color(f"{gpu.memory_gb}GB, {gpu.stock_status}", "dim")
        identifier = color(f"[{gpu.id}]", "dim")
        print(f"  {index}. {model} ({details}, {price}, counts: {counts}) {identifier}")
    if yes:
        priced = [gpu for gpu in choices if gpu.price_per_hour is not None]
        if not priced:
            raise RuntimeError("No matching GPU has a known price; cannot choose the cheapest automatically")
        selected = min(priced, key=lambda gpu: gpu.price_per_hour)
        model = color(selected.display_name, "cyan")
        price = color(f"${selected.price_per_hour:.3f}/hr", "green")
        print(f"Selected cheapest GPU: {model} ({price})")
        return selected
    try:
        raw = input(f"Select GPU [1-{len(choices)}] (default 1): ").strip()
    except EOFError as error:
        raise RuntimeError("GPU selection requires input; use --yes to choose the cheapest GPU") from error
    if not raw:
        return choices[0]
    try:
        index = int(raw)
        if not 1 <= index <= len(choices):
            raise ValueError("choice out of range")
        return choices[index - 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"Invalid GPU selection: {raw}") from error


def _forwarded_hf_env() -> dict[str, str]:
    keys = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    return {key: os.environ[key] for key in keys if os.environ.get(key)}

def _train_s1_from_config(config_path: str | Path, dataset_dir: str | Path, minio_uri: str | None) -> dict:
    config = load_config(config_path)
    mlflow = dict(config.get("mlflow") or {})
    extra = dict(mlflow.get("extra_params") or {})
    if minio_uri:
        extra["feature_dataset_uri"] = minio_uri
    mlflow["extra_params"] = extra
    result = train_s1(
        dataset_dir,
        label_column=(config.get("dataset") or {}).get("label_column", "label"),
        feature_config=config.get("features") or {},
        mlflow_config=mlflow,
    )
    return result.metrics


def _bundle_available(store: ObjectStore, bundle_uri: str) -> bool:
    try:
        return exists(store, bundle_uri) and exists(store, bundle_uri + ".sha256")
    except KeyboardInterrupt:
        return False


@dataclass
class PodProgress:
    api: RunPodApi
    pod_id: str
    last_status: str | None = None
    last_command: str | None = None
    lookup_failed: bool = False

    def __call__(self) -> None:
        lookup = getattr(self.api, "get_pod", None)
        if lookup is None:
            return
        try:
            pod = lookup(self.pod_id)
        except Exception as error:
            if not self.lookup_failed:
                logger.warning("Pod status lookup failed (%s); retrying while waiting for results", type(error).__name__)
            self.lookup_failed = True
            return
        if self.lookup_failed:
            logger.info("Pod status lookup recovered")
            self.lookup_failed = False
        status = str(pod.get("desiredStatus") or "initializing")
        if status != self.last_status:
            logger.info("Pod %s status=%s", self.pod_id, status)
            self.last_status = status
        command = ssh_command(pod)
        if command and command != self.last_command:
            print(f"\nPod SSH connection: {command}\nInside the pod: tmux attach -t ns\n", flush=True)
            logger.info("SSH endpoint assigned; the SSH service may take a moment to finish starting")
            self.last_command = command
        elif not command and self.last_command is None and status == "RUNNING":
            logger.debug("Pod is running; waiting for its public SSH port mapping")


def _wait_for_bundle(store: ObjectStore, bundle_uri: str, poll_seconds: float, timeout_seconds: float | None, monitor=None) -> None:
    started = time.monotonic()
    last_log = None
    logger.info("waiting for remote bundle uri=%s poll_seconds=%s timeout_seconds=%s", bundle_uri, poll_seconds, timeout_seconds)
    while True:
        if monitor is not None:
            monitor()
        if exists(store, bundle_uri) and exists(store, bundle_uri + ".sha256"):
            logger.info("remote bundle available uri=%s elapsed_seconds=%.1f", bundle_uri, time.monotonic() - started)
            return
        elapsed = time.monotonic() - started
        if last_log is None or elapsed - last_log >= 60:
            logger.info("still waiting for remote bundle uri=%s elapsed_seconds=%.1f", bundle_uri, elapsed)
            last_log = elapsed
        if timeout_seconds is not None and elapsed > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for {bundle_uri}")
        time.sleep(poll_seconds)


def _bundle_uri(config: dict, run_id: str) -> str:
    run = config.get("run") or {}
    if run.get("bundle_uri"):
        return str(run["bundle_uri"])
    base = run.get("s3_output_uri")
    if not base:
        bucket = os.environ.get("NEURALSIGNAL_S3_BUCKET")
        if not bucket:
            raise RuntimeError("NEURALSIGNAL_S3_BUCKET or run.s3_output_uri is required")
        base = f"s3://{bucket}/feature-runs"
    return s3_join(str(base), run_id, "bundle.zip")


def _fill_s3_defaults(config: dict, secrets: dict[str, str]) -> None:
    bucket = secrets.get("NEURALSIGNAL_S3_BUCKET") or os.environ.get("NEURALSIGNAL_S3_BUCKET")
    if bucket and not (config.get("run") or {}).get("s3_output_uri"):
        config.setdefault("run", {})["s3_output_uri"] = f"s3://{bucket}/feature-runs"


def _minio_store() -> Boto3ObjectStore:
    return Boto3ObjectStore(
        endpoint_url=os.environ.get("NEURALSIGNAL_MINIO_ENDPOINT_URL") or os.environ.get("MINIO_ENDPOINT_URL"),
        access_key_id=os.environ.get("MINIO_ACCESS_KEY"),
        secret_access_key=os.environ.get("MINIO_SECRET_KEY"),
    )

