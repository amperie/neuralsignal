import json

import pytest

from neuralsignal.cli.main import main


def test_top_level_help_lists_command_groups(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["-h"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "remote" in output
    assert "features" in output
    assert "train" in output


def test_remote_collect_help_lists_lifecycle_options(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["remote", "collect", "-h"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--run-id" in output
    assert "--launch-config" in output
    assert "--secrets-file" in output
    assert "--terraform-dir" in output
    assert "--minio-uri" in output
    assert "--train-config" in output
    assert "--gpu-vram-gb" in output
    assert "--gpu-id" in output


def test_dataset_import_jsonl_reports_rows(tmp_path, capsys):
    path = tmp_path / "examples.jsonl"
    path.write_text('{"id":"a","input":"i","output":"o"}\n', encoding="utf-8")

    assert main(["dataset", "import", "jsonl", str(path)]) == 0

    assert json.loads(capsys.readouterr().out)["rows"] == 1


def test_features_collect_local_writes_feature_shards(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("""
run:
  name: local-test
dataset:
  source: jsonl
features:
  materialize:
    - name: zones
storage:
  shard_size_rows: 1
""", encoding="utf-8")
    data = tmp_path / "examples.jsonl"
    data.write_text('{"id":"a","input":"abc","output":"xy"}\n', encoding="utf-8")

    assert main(["features", "collect-local", str(config), "--input-jsonl", str(data), "--out", str(tmp_path / "out")]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["rows"] == 1
    assert (tmp_path / "out" / "features" / "part-00000.parquet").exists()



def test_remote_collect_launch_config_dry_run(tmp_path, capsys):
    feature = tmp_path / "feature.yaml"
    feature.write_text("run:\n  s3_output_uri: s3://handoff/feature-runs\ndataset:\n  name: fixture\n", encoding="utf-8")
    manifest = tmp_path / "runpod.yaml"
    manifest.write_text("runpod:\n  image: image:latest\n", encoding="utf-8")
    launch = tmp_path / "launch.yaml"
    launch.write_text(f"""
remote_collect:
  config: {feature.as_posix()}
  manifest: {manifest.as_posix()}
  run_id: run-1
  terraform_dir: ""
  gpu_id: NVIDIA GeForce RTX 4090
""", encoding="utf-8")

    assert main(["remote", "collect", str(launch), "--dry-run"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["payload"]["name"] == "neuralsignal-run-1"
    assert output["bundle_uri"] == "s3://handoff/feature-runs/run-1/bundle.zip"
    assert output["payload"]["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090"]

def test_remote_collect_dry_run_redacts_secrets(tmp_path, capsys):
    config = tmp_path / "feature.yaml"
    config.write_text("run:\n  s3_output_uri: s3://handoff/feature-runs\ndataset:\n  name: fixture\n", encoding="utf-8")
    manifest = tmp_path / "runpod.yaml"
    manifest.write_text("runpod:\n  image: image:latest\n", encoding="utf-8")
    secrets = tmp_path / "runpod.secrets"
    secrets.write_text("HF_TOKEN=hf_x\nAWS_SECRET_ACCESS_KEY=secret\n", encoding="utf-8")

    assert main([
        "remote", "collect", str(config),
        "--manifest", str(manifest),
        "--run-id", "run-1",
        "--secrets-file", str(secrets),
        "--terraform-dir", "",
        "--dry-run",
    ]) == 0

    output = capsys.readouterr().out
    assert "hf_x" not in output
    assert "secret" not in output
    assert "<redacted>" in output


@pytest.mark.parametrize("configured, flags, expected", [
    ({}, [], (1800, 30)),
    ({"timeout_seconds": 7200, "poll_seconds": 7}, [], (7200, 7)),
    ({"timeout_seconds": 7200, "poll_seconds": 7},
     ["--timeout-seconds", "90", "--poll-seconds", "2"], (90, 2)),
])
def test_remote_launch_timing_precedence(tmp_path, monkeypatch, configured, flags, expected):
    import importlib
    import yaml

    cli = importlib.import_module("neuralsignal.cli.main")
    launch = tmp_path / "launch.yaml"
    launch.write_text(yaml.safe_dump({"remote_collect": {
        "config": "feature.yaml", "run_id": "timing-test", **configured,
    }}))
    captured = {}

    def collect(*args, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(cli, "remote_collect_lifecycle", collect)
    assert main(["remote", "collect", "--launch-config", str(launch), *flags]) == 0
    assert (captured["timeout_seconds"], captured["poll_seconds"]) == expected


@pytest.mark.parametrize("termination_fails", [False, True])
def test_ctrl_c_terminates_active_pod_without_traceback(tmp_path, monkeypatch, capsys, termination_fails):
    from neuralsignal.remote import lifecycle
    config = tmp_path / "feature.yaml"
    config.write_text("run:\n  s3_output_uri: s3://test/feature-runs\n")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("runpod:\n  image: test:latest\n")
    terminated = []
    monkeypatch.setattr(lifecycle.DefaultRunPodApi, "launch", lambda *args: {"id": "active-pod"})
    def terminate(self, pod_id):
        terminated.append(pod_id)
        if termination_fails:
            raise RuntimeError("private API response")
        return {}
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(lifecycle.DefaultRunPodApi, "terminate", terminate)
    monkeypatch.setattr(lifecycle, "Boto3ObjectStore", lambda **kwargs: object())
    monkeypatch.setattr(lifecycle, "_wait_for_bundle", interrupt)
    code = main(["remote", "collect", str(config), "--manifest", str(manifest),
                 "--run-id", "test", "--terraform-dir", "", "--env-file", ""])
    assert terminated == ["active-pod"]
    assert code == (1 if termination_fails else 130)
    output = capsys.readouterr().err
    assert "Cancelled" in output and "active-pod" in output
    assert "Traceback" not in output and "private API response" not in output
    if termination_fails:
        assert "could not confirm termination" in output
    else:
        assert "Pod active-pod terminated." in output


def test_ctrl_c_before_launch_has_no_traceback(monkeypatch, capsys):
    import importlib
    cli = importlib.import_module("neuralsignal.cli.main")
    def interrupt(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(cli, "_main", interrupt)
    assert cli.main([]) == 130
    assert capsys.readouterr().err == "Cancelled.\n"
