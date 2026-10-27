from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path

from neuralsignal.storage.manifests import sha256_file, local_shard_path
from neuralsignal.storage.s3 import ObjectStore, download_file, exists, s3_join

logger = logging.getLogger(__name__)


def sync_feature_run(store: ObjectStore, remote_run_uri: str, local_dir: str | Path) -> dict:
    local_dir = Path(local_dir)
    (local_dir / ".sync-complete").unlink(missing_ok=True)
    manifest_path = local_dir / "manifest.json"
    download_file(store, s3_join(remote_run_uri, "manifest.json"), manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("state", "completed") != "completed":
        raise ValueError("Feature run is not completed")

    for shard in manifest.get("shards", []):
        if shard.get("state") not in {"uploaded", "completed", "written"}:
            raise ValueError(f"Feature shard is not ready: {shard.get('path')}")
        local_path = local_shard_path(local_dir, shard["path"])
        if not local_path.exists() or sha256_file(local_path) != shard["sha256"]:
            download_file(store, s3_join(remote_run_uri, shard["path"]), local_path)
        actual = sha256_file(local_path)
        if actual != shard["sha256"]:
            raise ValueError(f"Checksum mismatch for {shard['path']}: expected {shard['sha256']} got {actual}")

    (local_dir / ".sync-complete").write_text(manifest["run_id"], encoding="utf-8")
    return manifest



def recover_partial_run(store: ObjectStore, remote_run_uri: str, target_dir: str | Path) -> int:
    """Download the shards a pod uploaded before it stopped, without a bundle.

    Only shards the remote manifest marks ``uploaded`` and whose checksum matches are
    kept. The result is laid out like an unpacked bundle (``manifest.json`` plus
    ``features/part-*.parquet``) so training can read it. The local manifest lists just
    the recovered shards, is marked ``completed`` so readers accept it, and carries
    ``"partial": true``. Returns the recovered row count; 0 means nothing was written
    to ``target_dir``.
    """
    target = Path(target_dir)
    manifest_uri = s3_join(remote_run_uri, "manifest.json")
    if not exists(store, manifest_uri):
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as staging:
        root = Path(staging) / "contents"
        root.mkdir()
        download_file(store, manifest_uri, root / "manifest.json")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        kept = []
        for shard in manifest.get("shards", []):
            if shard.get("state") != "uploaded":
                continue
            local_path = local_shard_path(root, shard["path"])
            try:
                download_file(store, s3_join(remote_run_uri, shard["path"]), local_path)
                actual = sha256_file(local_path)
            except Exception as error:
                logger.warning("skipping shard %s: download failed (%s)", shard["path"], type(error).__name__)
                continue
            if actual != shard["sha256"]:
                logger.warning("skipping shard %s: checksum mismatch", shard["path"])
                local_path.unlink(missing_ok=True)
                continue
            kept.append(shard)
        rows = sum(int(shard["rows"]) for shard in kept)
        if not rows:
            return 0
        log_uri = s3_join(remote_run_uri, "runpod.log")
        try:
            if exists(store, log_uri):
                download_file(store, log_uri, root / "runpod.log")
        except Exception:
            logger.debug("could not download runpod.log", exc_info=True)
        manifest.update({
            "shards": kept,
            "rows": {**(manifest.get("rows") or {}), "written": rows, "uploaded": rows},
            "source_state": manifest.get("state"),
            "state": "completed",
            "partial": True,
        })
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        if target.exists():
            shutil.rmtree(target)
        root.rename(target)
    return rows
