import json
from pathlib import Path

from neuralsignal.remote.lifecycle import PodExitedError, RemoteCollectCancelled, remote_collect_lifecycle
from neuralsignal.remote.runpod import RunPodGpuType
from neuralsignal.storage.bundle import create_bundle, upload_bundle


class FakeStore:
    def __init__(self):
        self.objects = {}
        self.deleted = []

    def download_file(self, bucket: str, key: str, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(self.objects[(bucket, key)])

    def upload_file(self, path: str, bucket: str, key: str) -> None:
        self.objects[(bucket, key)] = Path(path).read_bytes()

    def delete_file(self, bucket: str, key: str) -> None:
        self.deleted.append((bucket, key))
        self.objects.pop((bucket, key), None)

    def exists(self, bucket: str, key: str) -> bool:
        return (bucket, key) in self.objects


class FakeRunPod:
    def __init__(self, statuses=None):
        self.payload = None
        self.terminated = []
        self.gpu_queries = []
        self.statuses = list(statuses or ["RUNNING"])

    def launch(self, payload: dict) -> dict:
        self.payload = payload
        return {"id": "pod-1"}

    def get_pod(self, pod_id: str) -> dict:
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return {"id": pod_id, "desiredStatus": status}

    def terminate(self, pod_id: str) -> dict:
        self.terminated.append(pod_id)
        return {"id": pod_id}

    def list_gpu_types(self, gpu_count: int, secure_cloud: bool) -> list[RunPodGpuType]:
        self.gpu_queries.append((gpu_count, secure_cloud))
        return [
            RunPodGpuType("NVIDIA GeForce RTX 3080", "RTX 3080", 10, "High", 0.10, (1,)),
            RunPodGpuType("NVIDIA GeForce RTX 4090", "RTX 4090", 24, "High", 0.40, (1,)),
            RunPodGpuType("NVIDIA A40", "A40", 48, "High", 0.20, (1,)),
        ]


def test_remote_collect_lifecycle_downloads_deletes_unpacks_and_uploads_minio(tmp_path, monkeypatch):
    monkeypatch.setenv("NEURALSIGNAL_S3_BUCKET", "handoff")
    feature_config = tmp_path / "feature.yaml"
    feature_config.write_text("""
run:
  s3_output_uri: s3://${NEURALSIGNAL_S3_BUCKET}/feature-runs
dataset:
  source: jsonl
features:
  materialize: []
""", encoding="utf-8")
    runpod_manifest = tmp_path / "runpod.yaml"
    runpod_manifest.write_text("runpod:\n  image: image:latest\n", encoding="utf-8")

    source = tmp_path / "remote-result"
    (source / "features").mkdir(parents=True)
    (source / "manifest.json").write_text(json.dumps({"run_id": "run-1"}), encoding="utf-8")
    (source / "features" / "part-00000.parquet").write_bytes(b"feature-data")
    bundle = create_bundle(source, tmp_path / "bundle.zip")

    handoff = FakeStore()
    upload_bundle(handoff, bundle, "s3://handoff/feature-runs/run-1/bundle.zip")
    minio = FakeStore()
    runpod = FakeRunPod()

    result = remote_collect_lifecycle(
        feature_config,
        runpod_manifest,
        "run-1",
        tmp_path / "downloads",
        env_file=None,
        terraform_dir=None,
        minio_uri="s3://local-minio/feature-datasets/run-1",
        poll_seconds=0,
        runpod_api=runpod,
        store=handoff,
        minio_store=minio,
    )

    assert result.pod_id == "pod-1"
    assert runpod.terminated == ["pod-1"]
    assert ("handoff", "feature-runs/run-1/inputs/feature_config.yaml") in handoff.objects
    uploaded_config = handoff.objects[("handoff", "feature-runs/run-1/inputs/feature_config.yaml")].decode("utf-8")
    assert "${NEURALSIGNAL_S3_BUCKET}" not in uploaded_config
    assert "s3://handoff/feature-runs" in uploaded_config
    assert runpod.payload["env"]["NEURALSIGNAL_FEATURE_CONFIG_URI"] == "s3://handoff/feature-runs/run-1/inputs/feature_config.yaml"
    assert (tmp_path / "downloads" / "run-1" / "manifest.json").exists()
    assert ("handoff", "feature-runs/run-1/bundle.zip") in handoff.deleted
    assert ("local-minio", "feature-datasets/run-1/manifest.json") in minio.objects
    assert ("local-minio", "feature-datasets/run-1/minio-dataset-uri.txt") in minio.objects


def test_remote_collect_dry_run_redacts_payload(tmp_path):
    feature_config = tmp_path / "feature.yaml"
    feature_config.write_text("run:\n  s3_output_uri: s3://handoff/feature-runs\n", encoding="utf-8")
    runpod_manifest = tmp_path / "runpod.yaml"
    runpod_manifest.write_text("runpod:\n  image: image:latest\n", encoding="utf-8")
    secrets = tmp_path / "runpod.secrets"
    secrets.write_text("HF_TOKEN=hf_secret\n", encoding="utf-8")

    result = remote_collect_lifecycle(
        feature_config,
        runpod_manifest,
        "run-1",
        tmp_path,
        env_file=None,
        secrets_file=secrets,
        terraform_dir=None,
        dry_run=True,
    )

    assert result["bundle_uri"] == "s3://handoff/feature-runs/run-1/bundle.zip"
    assert result["payload"]["env"]["HF_TOKEN"] == "<redacted>"
    assert result["payload"]["env"]["NEURALSIGNAL_FEATURE_CONFIG_URI"] == "s3://handoff/feature-runs/run-1/inputs/feature_config.yaml"
    assert "NEURALSIGNAL_FEATURE_CONFIG_B64" not in result["payload"]["env"]



def test_remote_collect_dry_run_selects_gpu_within_requested_vram_window(tmp_path, monkeypatch):
    feature_config = tmp_path / "feature.yaml"
    feature_config.write_text("run:\n  s3_output_uri: s3://handoff/feature-runs\n", encoding="utf-8")
    runpod_manifest = tmp_path / "runpod.yaml"
    runpod_manifest.write_text("runpod:\n  image: image:latest\n  gpu_count: 1\n  cloud_type: SECURE\n", encoding="utf-8")
    runpod = FakeRunPod()
    monkeypatch.setattr("builtins.input", lambda _: "1")

    result = remote_collect_lifecycle(
        feature_config,
        runpod_manifest,
        "run-1",
        tmp_path,
        env_file=None,
        terraform_dir=None,
        dry_run=True,
        runpod_api=runpod,
        gpu_vram_gb=24,
    )

    assert runpod.gpu_queries == [(1, True)]
    assert result["payload"]["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090"]
    assert result["payload"]["gpuTypePriority"] == "custom"


def test_remote_collect_dry_run_accepts_explicit_gpu_id(tmp_path):
    feature_config = tmp_path / "feature.yaml"
    feature_config.write_text("run:\n  s3_output_uri: s3://handoff/feature-runs\n", encoding="utf-8")
    runpod_manifest = tmp_path / "runpod.yaml"
    runpod_manifest.write_text("runpod:\n  image: image:latest\n", encoding="utf-8")
    runpod = FakeRunPod()

    result = remote_collect_lifecycle(
        feature_config,
        runpod_manifest,
        "run-1",
        tmp_path,
        env_file=None,
        terraform_dir=None,
        dry_run=True,
        runpod_api=runpod,
        gpu_id="NVIDIA A40",
    )

    assert runpod.gpu_queries == []
    assert result["payload"]["gpuTypeIds"] == ["NVIDIA A40"]


def test_remote_collect_accepts_bundle_when_pod_exits_after_upload(tmp_path):
    feature_config = tmp_path / "feature.yaml"
    feature_config.write_text("run:\n  s3_output_uri: s3://handoff/feature-runs\n", encoding="utf-8")
    runpod_manifest = tmp_path / "runpod.yaml"
    runpod_manifest.write_text("runpod:\n  image: image:latest\n", encoding="utf-8")
    source = tmp_path / "remote-result"
    (source / "features").mkdir(parents=True)
    (source / "manifest.json").write_text(json.dumps({"run_id": "run-1"}), encoding="utf-8")
    (source / "features" / "part-00000.parquet").write_bytes(b"feature-data")
    bundle = create_bundle(source, tmp_path / "bundle.zip")
    handoff = FakeStore()
    upload_bundle(handoff, bundle, "s3://handoff/feature-runs/run-1/bundle.zip")
    runpod = FakeRunPod(statuses=["EXITED"])

    result = remote_collect_lifecycle(
        feature_config,
        runpod_manifest,
        "run-1",
        tmp_path / "downloads",
        env_file=None,
        terraform_dir=None,
        poll_seconds=0,
        runpod_api=runpod,
        store=handoff,
    )

    assert result.pod_id == "pod-1"
    assert (tmp_path / "downloads" / "run-1" / "manifest.json").exists()
    assert runpod.terminated == ["pod-1"]


def test_remote_collect_stops_when_pod_exits_before_bundle(tmp_path):
    import pytest

    feature_config = tmp_path / "feature.yaml"
    feature_config.write_text("run:\n  s3_output_uri: s3://handoff/feature-runs\n", encoding="utf-8")
    runpod_manifest = tmp_path / "runpod.yaml"
    runpod_manifest.write_text("runpod:\n  image: image:latest\n", encoding="utf-8")
    runpod = FakeRunPod(statuses=["RUNNING", "EXITED"])

    with pytest.raises(PodExitedError, match="pod-1.*EXITED"):
        remote_collect_lifecycle(
            feature_config,
            runpod_manifest,
            "run-1",
            tmp_path / "downloads",
            env_file=None,
            terraform_dir=None,
            poll_seconds=0,
            runpod_api=runpod,
            store=FakeStore(),
        )

    assert runpod.terminated == ["pod-1"]


def test_remote_collect_interrupt_terminates_and_cancels(tmp_path):
    class InterruptingStore(FakeStore):
        def exists(self, bucket: str, key: str) -> bool:
            raise KeyboardInterrupt

    feature_config = tmp_path / "feature.yaml"
    feature_config.write_text("run:\n  s3_output_uri: s3://handoff/feature-runs\n", encoding="utf-8")
    runpod_manifest = tmp_path / "runpod.yaml"
    runpod_manifest.write_text("runpod:\n  image: image:latest\n", encoding="utf-8")
    runpod = FakeRunPod()

    import pytest

    with pytest.raises(RemoteCollectCancelled, match="Pod pod-1 terminated"):
        remote_collect_lifecycle(
            feature_config,
            runpod_manifest,
            "run-1",
            tmp_path / "downloads",
            env_file=None,
            terraform_dir=None,
            poll_seconds=0,
            runpod_api=runpod,
            store=InterruptingStore(),
        )

    assert runpod.terminated == ["pod-1"]


def test_remote_collect_dry_run_forwards_worker_env_from_env_file(tmp_path, monkeypatch):
    for key in (
        "HF_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_DEFAULT_REGION",
        "NEURALSIGNAL_S3_BUCKET",
        "NEURALSIGNAL_S3_ENDPOINT_URL",
        "RUNPOD_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    feature_config = tmp_path / "feature.yaml"
    feature_config.write_text("run:\n  s3_output_uri: s3://handoff/feature-runs\n", encoding="utf-8")
    runpod_manifest = tmp_path / "runpod.yaml"
    runpod_manifest.write_text("runpod:\n  image: image:latest\n", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "HF_TOKEN=hf_from_env\n"
        "AWS_ACCESS_KEY_ID=aws_key\n"
        "AWS_SECRET_ACCESS_KEY=aws_secret\n"
        "AWS_SESSION_TOKEN=aws_session\n"
        "AWS_DEFAULT_REGION=us-east-1\n"
        "NEURALSIGNAL_S3_BUCKET=handoff\n"
        "NEURALSIGNAL_S3_ENDPOINT_URL=https://s3.test\n"
        "RUNPOD_API_KEY=rp_local_only\n",
        encoding="utf-8",
    )

    result = remote_collect_lifecycle(
        feature_config,
        runpod_manifest,
        "run-1",
        tmp_path,
        env_file=env_file,
        terraform_dir=None,
        dry_run=True,
    )

    env = result["payload"]["env"]
    assert env["HF_TOKEN"] == "<redacted>"
    assert env["AWS_ACCESS_KEY_ID"] == "<redacted>"
    assert env["AWS_SECRET_ACCESS_KEY"] == "<redacted>"
    assert env["AWS_SESSION_TOKEN"] == "<redacted>"
    assert env["AWS_DEFAULT_REGION"] == "us-east-1"
    assert env["NEURALSIGNAL_S3_BUCKET"] == "handoff"
    assert env["NEURALSIGNAL_S3_ENDPOINT_URL"] == "https://s3.test"
    assert "RUNPOD_API_KEY" not in env
