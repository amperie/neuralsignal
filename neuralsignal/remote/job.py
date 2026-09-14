from __future__ import annotations

import argparse
import base64
import json
import logging
import os
from itertools import islice
from pathlib import Path

from neuralsignal.console import configure_logging
from neuralsignal.config import load_config
from neuralsignal.datasets.sources.factory import source_from_config
from neuralsignal.features.model_extractor import ModelFeatureExtractor
from neuralsignal.features.selection import extraction_mode
from neuralsignal.features.runner import collect_features, collect_features_batched
from neuralsignal.storage.bundle import create_bundle, upload_bundle
from neuralsignal.storage.local import LocalFeatureShardWriter
from neuralsignal.storage.manifests import RunManifest, ShardManifest, local_shard_path
from neuralsignal.storage.s3 import Boto3ObjectStore, ObjectStore, download_file, s3_join, upload_file

logger = logging.getLogger(__name__)


class RemoteShardUploader:
    def __init__(self, run_dir: Path, run_uri: str, store: ObjectStore, manifest: RunManifest) -> None:
        self.run_dir = run_dir
        self.run_uri = run_uri
        self.store = store
        self.manifest = manifest

    def upload_shard(self, shard: ShardManifest) -> None:
        local_path = local_shard_path(self.run_dir, shard.path)
        shard_uri = s3_join(self.run_uri, shard.path)
        upload_file(self.store, local_path, shard_uri)
        shard.state = "uploaded"
        self._upload_manifest()
        logger.info("uploaded feature shard uri=%s rows=%s", shard_uri, shard.rows)

    def upload_recoverable(self) -> None:
        for shard in self.manifest.shards:
            if shard.state != "uploaded":
                self.upload_shard(shard)
        self._upload_manifest()
        self.upload_log()

    def complete(self) -> None:
        self.upload_recoverable()
        self.manifest.state = "completed"
        self._upload_manifest()

    def upload_log(self) -> None:
        log_path = self.run_dir / "runpod.log"
        if log_path.exists():
            upload_file(self.store, log_path, s3_join(self.run_uri, "runpod.log"))

    def _upload_manifest(self) -> None:
        self.manifest.rows["uploaded"] = sum(shard.rows for shard in self.manifest.shards if shard.state == "uploaded")
        path = self.run_dir / "manifest.json"
        self.manifest.write_json(path)
        upload_file(self.store, path, s3_join(self.run_uri, "manifest.json"))

def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=os.environ.get("NEURALSIGNAL_RUN_ID"),
                        help="Run ID; defaults to NEURALSIGNAL_RUN_ID on RunPod.")
    parser.add_argument("--config", help="Optional config path. Env config is used when omitted.")
    args = parser.parse_args()
    if not args.run_id or not args.run_id.strip():
        parser.error("--run-id or NEURALSIGNAL_RUN_ID is required")
    logger.info("remote job starting run_id=%s", args.run_id)
    config = load_config(args.config) if args.config else _load_config_from_env()
    mode = extraction_mode(config)
    run_dir = Path(os.environ.get("NEURALSIGNAL_RUN_WORKDIR", "/workspace/neuralsignal-runs")) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _log_config_summary(config, run_dir)

    manifest = RunManifest(
        run_id=args.run_id,
        dataset=config.get("dataset") or {},
        features={"schema_version": "features.v1", "materialized_sets": (config.get("features") or {}).get("materialize", [])},
        state="running",
    )
    store = _s3_store()
    run_uri = _run_uri(config, args.run_id)
    uploader = RemoteShardUploader(run_dir, run_uri, store, manifest)
    writer = LocalFeatureShardWriter(run_dir, manifest, on_shard_written=uploader.upload_shard)
    try:
        source = _dataset_source(config).iter_examples()
        max_examples = (config.get("dataset") or {}).get("max_examples")
        if max_examples is not None:
            if isinstance(max_examples, bool) or not isinstance(max_examples, int) or max_examples <= 0:
                raise ValueError("dataset.max_examples must be a positive integer")
            source = islice(source, max_examples)
        logger.info("feature collection starting")
        if mode == "model":
            logger.info("model extractor initializing")
            extractor = ModelFeatureExtractor(config)
            logger.info("model extractor ready")
            collect_features_batched(source, config, writer, extractor.extract_batch)
        else:
            collect_features(source, config, writer, _placeholder_extractor)
    except Exception:
        logger.exception("remote job failed; uploading recoverable shards before exit")
        manifest.state = "failed"
        uploader.upload_recoverable()
        raise
    logger.info("feature collection completed rows=%s shards=%s", manifest.rows["written"], len(manifest.shards))
    uploader.complete()
    logger.info("manifest written path=%s", run_dir / "manifest.json")

    bundle_uri = _bundle_uri(config, args.run_id)
    logger.info("creating bundle uri=%s", bundle_uri)
    bundle = create_bundle(run_dir, run_dir.parent / f"{args.run_id}.zip")
    logger.info("bundle created path=%s bytes=%s", bundle, bundle.stat().st_size)
    logger.info("uploading bundle to s3")
    upload_bundle(store, bundle, bundle_uri)
    logger.info("bundle upload completed uri=%s", bundle_uri)
    print(json.dumps({"run_id": args.run_id, "bundle_uri": bundle_uri, "rows": manifest.rows["written"]}))


def _configure_logging() -> None:
    configure_logging()


def _log_config_summary(config: dict, run_dir: Path) -> None:
    model = config.get("model") or {}
    generation = config.get("generation") or {}
    dataset = config.get("dataset") or {}
    features = [entry.get("name") for entry in (config.get("features") or {}).get("materialize", []) if entry.get("enabled", True)]
    logger.info("run directory path=%s", run_dir)
    logger.info("dataset source=%s path=%s", dataset.get("source"), dataset.get("path"))
    logger.info("model name=%s device=%s quantization=%s", model.get("model_name"), model.get("device"), model.get("quantization"))
    logger.info("generation batch_size=%s max_new_tokens=%s truncation_length=%s", generation.get("batch_size"), generation.get("max_new_tokens"), generation.get("truncation_length"))
    logger.info("enabled feature_sets=%s", ",".join(str(name) for name in features) or "none")


def _load_config_from_env() -> dict:
    uri = os.environ.get("NEURALSIGNAL_FEATURE_CONFIG_URI")
    if uri:
        path = Path(os.environ.get("NEURALSIGNAL_RUN_WORKDIR", "/workspace/neuralsignal-runs")) / "inputs" / "feature_config.yaml"
        logger.info("downloading feature config uri=%s path=%s", uri, path)
        download_file(_s3_store(), uri, path)
        return load_config(path)
    encoded = os.environ.get("NEURALSIGNAL_FEATURE_CONFIG_B64")
    if not encoded:
        raise RuntimeError("NEURALSIGNAL_FEATURE_CONFIG_URI or NEURALSIGNAL_FEATURE_CONFIG_B64 is required")
    logger.info("loading feature config from NEURALSIGNAL_FEATURE_CONFIG_B64 fallback")
    return json.loads(base64.b64decode(encoded).decode("utf-8"))


def _dataset_source(config: dict):
    return source_from_config(config)


def _run_uri(config: dict, run_id: str) -> str:
    run = config.get("run") or {}
    if run.get("bundle_uri"):
        return str(run["bundle_uri"]).rsplit("/", 1)[0]
    base = run.get("s3_output_uri") or f"s3://{os.environ['NEURALSIGNAL_S3_BUCKET']}/feature-runs"
    return s3_join(str(base), run_id)


def _bundle_uri(config: dict, run_id: str) -> str:
    run = config.get("run") or {}
    if run.get("bundle_uri"):
        return str(run["bundle_uri"])
    return s3_join(_run_uri(config, run_id), "bundle.zip")


def _s3_store() -> Boto3ObjectStore:
    return Boto3ObjectStore(endpoint_url=os.environ.get("NEURALSIGNAL_S3_ENDPOINT_URL"))


def _placeholder_extractor(example, spec) -> dict[str, float]:
    return {"input_chars": float(len(example.input)), "output_chars": float(len(example.output))}


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("remote job failed")
        raise
