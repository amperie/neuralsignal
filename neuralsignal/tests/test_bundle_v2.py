from pathlib import Path

from neuralsignal.storage.bundle import create_bundle, delete_bundle, download_bundle, unpack_bundle, upload_bundle


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

    def exists(self, bucket: str, key: str) -> bool:
        return (bucket, key) in self.objects


def test_bundle_upload_download_unpack_delete_roundtrip(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    bundle = create_bundle(source, tmp_path / "bundle.zip")
    store = FakeStore()

    upload_bundle(store, bundle, "s3://bucket/runs/run-1/bundle.zip")
    downloaded = download_bundle(store, "s3://bucket/runs/run-1/bundle.zip", tmp_path / "downloaded.zip")
    unpack_bundle(downloaded, tmp_path / "unpacked")
    delete_bundle(store, "s3://bucket/runs/run-1/bundle.zip")

    assert (tmp_path / "unpacked" / "manifest.json").exists()
    assert ("bucket", "runs/run-1/bundle.zip") in store.deleted
    assert ("bucket", "runs/run-1/bundle.zip.sha256") in store.deleted

