from __future__ import annotations

import json
from pathlib import Path

from neuralsignal.storage.manifests import sha256_file
from neuralsignal.storage.s3 import ObjectStore, download_file, s3_join


def sync_feature_run(store: ObjectStore, remote_run_uri: str, local_dir: str | Path) -> dict:
    local_dir = Path(local_dir)
    manifest_path = local_dir / "manifest.json"
    download_file(store, s3_join(remote_run_uri, "manifest.json"), manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for shard in manifest.get("shards", []):
        if shard.get("state") not in {"uploaded", "completed", "written"}:
            continue
        local_path = local_dir / shard["path"]
        if not local_path.exists() or sha256_file(local_path) != shard["sha256"]:
            download_file(store, s3_join(remote_run_uri, shard["path"]), local_path)
        actual = sha256_file(local_path)
        if actual != shard["sha256"]:
            raise ValueError(f"Checksum mismatch for {shard['path']}: expected {shard['sha256']} got {actual}")

    (local_dir / ".sync-complete").write_text(manifest["run_id"], encoding="utf-8")
    return manifest

