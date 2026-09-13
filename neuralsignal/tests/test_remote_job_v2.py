import base64
import json
import sys

import pytest
from pathlib import Path

from neuralsignal.remote import job


class FakeStore:
    def __init__(self):
        self.objects = {}

    def download_file(self, bucket: str, key: str, path: str) -> None:
        Path(path).write_bytes(self.objects[(bucket, key)])

    def upload_file(self, path: str, bucket: str, key: str) -> None:
        self.objects[(bucket, key)] = Path(path).read_bytes()

    def delete_file(self, bucket: str, key: str) -> None:
        self.objects.pop((bucket, key), None)

    def exists(self, bucket: str, key: str) -> bool:
        return (bucket, key) in self.objects


@pytest.mark.parametrize("use_env", [False, True])
def test_remote_job_collects_features_and_uploads_bundle(tmp_path, monkeypatch, capsys, use_env):
    input_path = tmp_path / "examples.jsonl"
    input_path.write_text('{"id":"a","input":"abc","output":"xy"}\n' * 3, encoding="utf-8")
    config = {
        "run": {"s3_output_uri": "s3://handoff/feature-runs"},
        "dataset": {"source": "jsonl", "path": str(input_path), "max_examples": 1},
        "features": {"materialize": [{"name": "zones"}]},
        "storage": {"shard_size_rows": 10},
    }
    encoded = base64.b64encode(json.dumps(config).encode("utf-8")).decode("ascii")
    store = FakeStore()
    monkeypatch.setenv("NEURALSIGNAL_FEATURE_CONFIG_B64", encoded)
    monkeypatch.setenv("NEURALSIGNAL_RUN_WORKDIR", str(tmp_path / "work"))
    monkeypatch.setattr(job, "_s3_store", lambda: store)
    monkeypatch.setenv("NEURALSIGNAL_RUN_ID", "run-1" if use_env else "env-overridden")
    monkeypatch.setattr(sys, "argv", ["job"] if use_env else ["job", "--run-id", "run-1"])

    job.main()

    output = json.loads(capsys.readouterr().out)
    assert output["rows"] == 1
    assert ("handoff", "feature-runs/run-1/bundle.zip") in store.objects
    assert ("handoff", "feature-runs/run-1/bundle.zip.sha256") in store.objects



def test_remote_job_loads_feature_config_from_s3_uri(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "examples.jsonl"
    input_path.write_text('{"id":"a","input":"abc","output":"xy"}\n', encoding="utf-8")
    config = {
        "run": {"s3_output_uri": "s3://handoff/feature-runs"},
        "dataset": {"source": "jsonl", "path": str(input_path)},
        "features": {"materialize": [{"name": "zones"}]},
        "storage": {"shard_size_rows": 10},
    }
    store = FakeStore()
    config_path = tmp_path / "uploaded.yaml"
    import yaml
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    store.upload_file(config_path, "handoff", "feature-runs/run-1/inputs/feature_config.yaml")
    monkeypatch.setenv("NEURALSIGNAL_FEATURE_CONFIG_URI", "s3://handoff/feature-runs/run-1/inputs/feature_config.yaml")
    monkeypatch.setenv("NEURALSIGNAL_RUN_WORKDIR", str(tmp_path / "work"))
    monkeypatch.setattr(job, "_s3_store", lambda: store)
    monkeypatch.setenv("NEURALSIGNAL_RUN_ID", "run-1")
    monkeypatch.setattr(sys, "argv", ["job"])

    job.main()

    output = json.loads(capsys.readouterr().out)
    assert output["rows"] == 1
    assert ("handoff", "feature-runs/run-1/bundle.zip") in store.objects


@pytest.mark.parametrize("value", [None, "", "   "])
def test_remote_job_requires_run_identity(monkeypatch, capsys, value):
    monkeypatch.delenv("NEURALSIGNAL_RUN_ID", raising=False)
    if value is not None:
        monkeypatch.setenv("NEURALSIGNAL_RUN_ID", value)
    monkeypatch.setattr(sys, "argv", ["job"])
    with pytest.raises(SystemExit) as error:
        job.main()
    assert error.value.code == 2
    assert "--run-id or NEURALSIGNAL_RUN_ID is required" in capsys.readouterr().err
