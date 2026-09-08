from pathlib import Path

import pandas as pd
import pytest

from neuralsignal.config import load_config
from neuralsignal.features import materialized_feature_sets
from neuralsignal.storage import LocalFeatureShardWriter, RunManifest, sha256_file


def test_load_config_deep_merges_overrides(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("a:\n  b: 1\n  c: 2\n", encoding="utf-8")

    config = load_config(path, {"a": {"b": 3}, "d": 4})

    assert config == {"a": {"b": 3, "c": 2}, "d": 4}


def test_materialized_feature_sets_filters_disabled_entries():
    config = {
        "features": {
            "materialize": [
                {"name": "zones", "enabled": True, "config": {"x": 1}},
                {"name": "logit-lens", "enabled": False},
                {"name": "layer_distribution"},
            ]
        }
    }

    specs = materialized_feature_sets(config)

    assert [spec.name for spec in specs] == ["zones", "layer_distribution"]
    assert specs[0].column_prefix() == "zones__"


def test_materialized_feature_sets_requires_name():
    with pytest.raises(ValueError, match="requires a name"):
        materialized_feature_sets({"features": {"materialize": [{"enabled": True}]}})


def test_local_feature_shard_writer_writes_manifest_and_parquet(tmp_path: Path):
    manifest = RunManifest(
        run_id="run-1",
        dataset={"name": "fixture", "split": "train"},
        features={"schema_version": "features.v1", "materialized_sets": [{"name": "zones"}]},
    )
    writer = LocalFeatureShardWriter(tmp_path, manifest)

    shard = writer.write_shard([
        {"example_id": "a", "zones__x": 1.0},
        {"example_id": "b", "zones__x": 2.0},
    ])

    shard_path = tmp_path / shard.path
    assert shard.rows == 2
    assert shard.sha256 == sha256_file(shard_path)
    assert (tmp_path / "manifest.json").exists()
    assert pd.read_parquet(shard_path)["zones__x"].tolist() == [1.0, 2.0]


def test_manifest_rejects_duplicate_shard_indexes():
    manifest = RunManifest(
        run_id="run-1",
        dataset={"name": "fixture"},
        features={"schema_version": "features.v1"},
    )
    writer = LocalFeatureShardWriter(Path.cwd(), manifest)
    shard = writer.write_shard([{"example_id": "a"}])

    with pytest.raises(ValueError, match="Duplicate shard index"):
        manifest.add_shard(shard)


def test_load_config_expands_environment_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("NEURALSIGNAL_S3_BUCKET", "bucket-1")
    path = tmp_path / "config.yaml"
    path.write_text("run:\n  s3_output_uri: s3://${NEURALSIGNAL_S3_BUCKET}/feature-runs\n", encoding="utf-8")

    config = load_config(path)

    assert config["run"]["s3_output_uri"] == "s3://bucket-1/feature-runs"
