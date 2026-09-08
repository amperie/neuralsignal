import json
from pathlib import Path

from neuralsignal.remote import sync_feature_run
from neuralsignal.storage.manifests import sha256_file
from neuralsignal.storage.s3 import parse_s3_uri, s3_join, upload_file


class FakeStore:
    def __init__(self):
        self.objects = {}

    def download_file(self, bucket: str, key: str, path: str) -> None:
        Path(path).write_bytes(self.objects[(bucket, key)])

    def upload_file(self, path: str, bucket: str, key: str) -> None:
        self.objects[(bucket, key)] = Path(path).read_bytes()


def test_parse_s3_uri_and_join():
    assert parse_s3_uri("s3://bucket/a/b") == ("bucket", "a/b")
    assert s3_join("s3://bucket/a", "b", "/c/") == "s3://bucket/a/b/c"


def test_upload_file_uses_parsed_bucket_and_key(tmp_path):
    store = FakeStore()
    path = tmp_path / "x.txt"
    path.write_text("hello", encoding="utf-8")

    upload_file(store, path, "s3://bucket/prefix/x.txt")

    assert store.objects[("bucket", "prefix/x.txt")] == b"hello"


def test_sync_feature_run_downloads_and_verifies_shards(tmp_path):
    store = FakeStore()
    shard_bytes = b"feature-bytes"
    shard_path = tmp_path / "source.parquet"
    shard_path.write_bytes(shard_bytes)
    checksum = sha256_file(shard_path)
    manifest = {
        "run_id": "run-1",
        "shards": [
            {"path": "features/part-00000.parquet", "sha256": checksum, "state": "uploaded"}
        ],
    }
    store.objects[("bucket", "runs/run-1/manifest.json")] = json.dumps(manifest).encode("utf-8")
    store.objects[("bucket", "runs/run-1/features/part-00000.parquet")] = shard_bytes

    result = sync_feature_run(store, "s3://bucket/runs/run-1", tmp_path / "local")

    assert result["run_id"] == "run-1"
    assert (tmp_path / "local" / "features" / "part-00000.parquet").read_bytes() == shard_bytes
    assert (tmp_path / "local" / ".sync-complete").read_text(encoding="utf-8") == "run-1"

