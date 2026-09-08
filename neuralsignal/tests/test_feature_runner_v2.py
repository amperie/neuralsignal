import pandas as pd

from neuralsignal.datasets.v2 import DatasetExample
from neuralsignal.features.runner import collect_features
from neuralsignal.storage.local import LocalFeatureShardWriter
from neuralsignal.storage.manifests import RunManifest


def test_collect_features_materializes_enabled_sets_and_shards(tmp_path):
    config = {
        "features": {
            "materialize": [
                {"name": "zones", "enabled": True},
                {"name": "logit-lens", "enabled": False},
            ]
        },
        "storage": {"shard_size_rows": 2},
    }
    manifest = RunManifest(
        run_id="run-1",
        dataset={"name": "fixture"},
        features={"schema_version": "features.v1"},
    )
    writer = LocalFeatureShardWriter(tmp_path, manifest)
    examples = [
        DatasetExample(id="a", input="i1", output="o1", labels=["0"]),
        DatasetExample(id="b", input="i2", output="o2", labels=["1"]),
        DatasetExample(id="c", input="i3", output="o3", labels=["0"]),
    ]

    def extractor(example, spec):
        return {"length": len(example.input) + len(example.output)}

    result = collect_features(examples, config, writer, extractor)

    assert len(result.shards) == 2
    first = pd.read_parquet(tmp_path / "features" / "part-00000.parquet")
    assert first["zones__length"].tolist() == [4.0, 4.0]
    assert "logit-lens__length" not in first.columns


def test_collect_features_does_not_double_prefix_columns(tmp_path):
    config = {"features": {"materialize": [{"name": "zones"}]}, "storage": {"shard_size_rows": 10}}
    manifest = RunManifest(run_id="run-1", dataset={"name": "fixture"}, features={"schema_version": "features.v1"})
    writer = LocalFeatureShardWriter(tmp_path, manifest)

    collect_features(
        [DatasetExample(id="a", input="i", output="o")],
        config,
        writer,
        lambda example, spec: {"zones__x": 1},
    )

    data = pd.read_parquet(tmp_path / "features" / "part-00000.parquet")
    assert "zones__x" in data.columns
    assert "zones__zones__x" not in data.columns

