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


def test_remote_collect_ignores_expanded_shards_until_bundle_exists(tmp_path):
    import pytest

    feature_config = tmp_path / "feature.yaml"
    feature_config.write_text("run:\n  s3_output_uri: s3://handoff/feature-runs\n", encoding="utf-8")
    runpod_manifest = tmp_path / "runpod.yaml"
    runpod_manifest.write_text("runpod:\n  image: image:latest\n", encoding="utf-8")
    handoff = FakeStore()
    handoff.objects[("handoff", "feature-runs/run-1/manifest.json")] = json.dumps({"state": "running"}).encode("utf-8")
    handoff.objects[("handoff", "feature-runs/run-1/features/part-00000.parquet")] = b"partial"
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
            store=handoff,
        )

    assert runpod.terminated == ["pod-1"]


def _partial_remote_run(store, run_uri_prefix="feature-runs/run-1", shard_rows=(3, 2), state="running",
                        extra_unuploaded=True, corrupt=()):
    """Put a manifest plus uploaded shards in the fake store, like RemoteShardUploader does."""
    import pandas as pd
    from neuralsignal.storage.manifests import RunManifest, ShardManifest, sha256_file

    work = Path(store.workdir)
    manifest = RunManifest(run_id="run-1", dataset={"source": "fibs"}, features={"schema_version": "features.v1"},
                           state=state)
    for index, rows in enumerate(shard_rows):
        rel = f"features/part-{index:05d}.parquet"
        path = work / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": [f"{index}-{i}" for i in range(rows)], "x": [float(i) for i in range(rows)]}).to_parquet(path)
        manifest.add_shard(ShardManifest(index, rel, rows, sha256_file(path), state="uploaded"))
        data = path.read_bytes() + (b"corrupt" if index in corrupt else b"")
        store.objects[("handoff", f"{run_uri_prefix}/{rel}")] = data
    if extra_unuploaded:  # written locally on the pod but never uploaded
        manifest.add_shard(ShardManifest(len(shard_rows), "features/part-99999.parquet", 7, "ab" * 32, state="written"))
    manifest.write_json(work / "manifest.json")
    store.objects[("handoff", f"{run_uri_prefix}/manifest.json")] = (work / "manifest.json").read_bytes()


def _partial_setup(tmp_path):
    feature_config = tmp_path / "feature.yaml"
    feature_config.write_text("run:\n  s3_output_uri: s3://handoff/feature-runs\n", encoding="utf-8")
    runpod_manifest = tmp_path / "runpod.yaml"
    runpod_manifest.write_text("runpod:\n  image: image:latest\n", encoding="utf-8")
    store = FakeStore()
    store.workdir = tmp_path / "pod"
    store.workdir.mkdir()
    return feature_config, runpod_manifest, store


def test_timeout_recovers_uploaded_shards_trains_and_marks_partial(tmp_path, monkeypatch, caplog):
    import logging
    import pandas as pd

    feature_config, runpod_manifest, store = _partial_setup(tmp_path)
    _partial_remote_run(store)
    trained = []

    def fake_train(config_path, dataset_dir, minio_uri):
        trained.append(Path(dataset_dir))
        from types import SimpleNamespace
        return SimpleNamespace(metrics={"auroc": 0.7}, output_dir="runs/s1/x")

    monkeypatch.setattr("neuralsignal.remote.lifecycle._train_s1_from_config", fake_train)
    train_config = tmp_path / "train.yaml"
    train_config.write_text("model: {}\n", encoding="utf-8")
    runpod = FakeRunPod()

    with caplog.at_level(logging.WARNING):
        result = remote_collect_lifecycle(
            feature_config, runpod_manifest, "run-1", tmp_path / "downloads", env_file=None, terraform_dir=None,
            train_config_path=train_config, poll_seconds=0, timeout_seconds=-1, runpod_api=runpod, store=store)

    assert runpod.terminated == ["pod-1"]
    assert result.partial is True and result.partial_rows == 5
    assert result.training_metrics == {"auroc": 0.7}
    assert "partial run: 5 rows" in caplog.text
    target = tmp_path / "downloads" / "run-1"
    assert trained == [target]
    assert result.target_dir == str(target)
    local = json.loads((target / "manifest.json").read_text())
    assert [s["path"] for s in local["shards"]] == ["features/part-00000.parquet", "features/part-00001.parquet"]
    assert local["state"] == "completed" and local["partial"] is True and local["source_state"] == "running"
    assert not (target / "features" / "part-99999.parquet").exists()
    # Training's own reader accepts the recovered layout.
    from neuralsignal.training.s1 import _read_features
    assert len(_read_features(target)) == 5


def test_pod_exit_without_bundle_recovers_partial_run_without_training(tmp_path):
    feature_config, runpod_manifest, store = _partial_setup(tmp_path)
    _partial_remote_run(store, shard_rows=(4,), state="failed", extra_unuploaded=False)
    runpod = FakeRunPod(statuses=["EXITED"])

    result = remote_collect_lifecycle(
        feature_config, runpod_manifest, "run-1", tmp_path / "downloads", env_file=None, terraform_dir=None,
        poll_seconds=0, runpod_api=runpod, store=store)

    assert result.partial is True and result.partial_rows == 4 and result.training_metrics is None
    assert runpod.terminated == ["pod-1"]
    assert (tmp_path / "downloads" / "run-1" / "features" / "part-00000.parquet").exists()


def test_timeout_with_zero_uploaded_rows_reraises_and_terminates_pod(tmp_path):
    import pytest

    feature_config, runpod_manifest, store = _partial_setup(tmp_path)
    runpod = FakeRunPod()
    with pytest.raises(TimeoutError, match="Timed out waiting"):
        remote_collect_lifecycle(
            feature_config, runpod_manifest, "run-1", tmp_path / "downloads", env_file=None, terraform_dir=None,
            poll_seconds=0, timeout_seconds=-1, runpod_api=runpod, store=store)
    assert runpod.terminated == ["pod-1"]
    assert not (tmp_path / "downloads" / "run-1").exists()

    # A manifest that lists only never-uploaded shards also counts as zero rows.
    _partial_remote_run(store, shard_rows=(), extra_unuploaded=True)
    runpod = FakeRunPod()
    with pytest.raises(TimeoutError):
        remote_collect_lifecycle(
            feature_config, runpod_manifest, "run-1", tmp_path / "downloads", env_file=None, terraform_dir=None,
            poll_seconds=0, timeout_seconds=-1, runpod_api=runpod, store=store)
    assert runpod.terminated == ["pod-1"]


def test_recovery_skips_shards_with_bad_checksums(tmp_path):
    feature_config, runpod_manifest, store = _partial_setup(tmp_path)
    _partial_remote_run(store, shard_rows=(3, 2), extra_unuploaded=False, corrupt=(0,))
    result = remote_collect_lifecycle(
        feature_config, runpod_manifest, "run-1", tmp_path / "downloads", env_file=None, terraform_dir=None,
        poll_seconds=0, timeout_seconds=-1, runpod_api=FakeRunPod(), store=store)
    assert result.partial_rows == 2
    local = json.loads((tmp_path / "downloads" / "run-1" / "manifest.json").read_text())
    assert [s["path"] for s in local["shards"]] == ["features/part-00001.parquet"]


def test_completed_bundle_is_not_marked_partial(tmp_path):
    feature_config, runpod_manifest, store = _partial_setup(tmp_path)
    source = tmp_path / "remote-result"
    (source / "features").mkdir(parents=True)
    (source / "manifest.json").write_text(json.dumps({"run_id": "run-1"}), encoding="utf-8")
    (source / "features" / "part-00000.parquet").write_bytes(b"feature-data")
    upload_bundle(store, create_bundle(source, tmp_path / "bundle.zip"), "s3://handoff/feature-runs/run-1/bundle.zip")
    result = remote_collect_lifecycle(
        feature_config, runpod_manifest, "run-1", tmp_path / "downloads", env_file=None, terraform_dir=None,
        poll_seconds=0, runpod_api=FakeRunPod(), store=store)
    assert result.partial is False and result.partial_rows is None
