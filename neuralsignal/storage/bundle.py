from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from neuralsignal.storage.manifests import sha256_file
from neuralsignal.storage.s3 import ObjectStore, delete_file, download_file, upload_file


def create_bundle(source_dir: str | Path, bundle_path: str | Path) -> Path:
    source = Path(source_dir)
    bundle = Path(bundle_path)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            if path.resolve() != bundle.resolve():
                archive.write(path, path.relative_to(source).as_posix())
    bundle.with_suffix(bundle.suffix + ".sha256").write_text(sha256_file(bundle), encoding="utf-8")
    return bundle


def upload_bundle(store: ObjectStore, bundle_path: str | Path, bundle_uri: str) -> None:
    bundle = Path(bundle_path)
    upload_file(store, bundle, bundle_uri)
    upload_file(store, bundle.with_suffix(bundle.suffix + ".sha256"), bundle_uri + ".sha256")


def download_bundle(store: ObjectStore, bundle_uri: str, target_zip: str | Path) -> Path:
    target = Path(target_zip)
    download_file(store, bundle_uri, target)
    download_file(store, bundle_uri + ".sha256", str(target) + ".sha256")
    expected = Path(str(target) + ".sha256").read_text(encoding="utf-8").strip()
    actual = sha256_file(target)
    if actual != expected:
        raise ValueError(f"Bundle checksum mismatch: expected {expected} got {actual}")
    return target


def unpack_bundle(bundle_path: str | Path, target_dir: str | Path) -> Path:
    target = Path(target_dir)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with zipfile.ZipFile(bundle_path, "r") as archive:
        archive.extractall(target)
    return target


def delete_bundle(store: ObjectStore, bundle_uri: str) -> None:
    delete_file(store, bundle_uri)
    delete_file(store, bundle_uri + ".sha256")

