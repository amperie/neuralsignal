from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from neuralsignal.config import load_config
from neuralsignal.config.env import apply_env, load_env_file
from neuralsignal.remote.runpod import RunPodJob, build_pod_payload, launch, load_secrets, redacted, terminate
from neuralsignal.remote.terraform import s3_settings_from_terraform
from neuralsignal.storage.bundle import delete_bundle, download_bundle, unpack_bundle
from neuralsignal.storage.s3 import Boto3ObjectStore, ObjectStore, exists, s3_join, upload_directory
from neuralsignal.training import train_s1


class RunPodApi(Protocol):
    def launch(self, payload: dict) -> dict:
        ...

    def terminate(self, pod_id: str) -> dict:
        ...


@dataclass
class DefaultRunPodApi:
    token: str | None = None

    def launch(self, payload: dict) -> dict:
        return launch(payload, self.token)

    def terminate(self, pod_id: str) -> dict:
        return terminate(pod_id, self.token)


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
) -> RemoteCollectResult | dict:
    if env_file:
        apply_env(load_env_file(env_file))
    config = load_config(config_path)
    manifest = load_config(runpod_manifest_path)
    secrets = {}
    if terraform_dir and Path(terraform_dir).exists():
        secrets.update(s3_settings_from_terraform(terraform_dir))
    if secrets_file:
        secrets.update(load_secrets(secrets_file))
    _fill_s3_defaults(config, secrets)

    bundle_uri = _bundle_uri(config, run_id)
    payload = build_pod_payload(RunPodJob(run_id, config, manifest), secrets)
    if dry_run:
        return redacted({"payload": payload, "bundle_uri": bundle_uri})

    runpod_api = runpod_api or DefaultRunPodApi()
    store = store or Boto3ObjectStore(endpoint_url=os.environ.get("NEURALSIGNAL_S3_ENDPOINT_URL"))
    pod_id = str(runpod_api.launch(payload)["id"])
    training_metrics = None
    interrupted = False
    try:
        _wait_for_bundle(store, bundle_uri, poll_seconds, timeout_seconds)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        runpod_api.terminate(pod_id)

    if not _bundle_available(store, bundle_uri):
        reason = "interrupted" if interrupted else "finished without bundle"
        raise RuntimeError(f"Remote run {reason}; bundle is not available: {bundle_uri}")

    target = Path(target_dir) / run_id
    zip_path = Path(target_dir) / f"{run_id}.zip"
    download_bundle(store, bundle_uri, zip_path)
    delete_bundle(store, bundle_uri)
    unpack_bundle(zip_path, target)
    if minio_uri:
        (target / "minio-dataset-uri.txt").write_text(minio_uri, encoding="utf-8")
        minio_store = minio_store or _minio_store()
        upload_directory(minio_store, target, minio_uri)
    if train_config_path:
        training_metrics = _train_s1_from_config(train_config_path, target, minio_uri)
    return RemoteCollectResult(run_id, pod_id, bundle_uri, str(target), training_metrics)


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


def _wait_for_bundle(store: ObjectStore, bundle_uri: str, poll_seconds: float, timeout_seconds: float | None) -> None:
    started = time.monotonic()
    while True:
        if exists(store, bundle_uri) and exists(store, bundle_uri + ".sha256"):
            return
        if timeout_seconds is not None and time.monotonic() - started > timeout_seconds:
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