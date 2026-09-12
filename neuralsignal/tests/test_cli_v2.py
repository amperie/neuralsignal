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
