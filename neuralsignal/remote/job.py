from __future__ import annotations

import argparse
import base64
import json
import logging
import os
from pathlib import Path

from neuralsignal.config import load_config
from neuralsignal.datasets.sources.factory import source_from_config
from neuralsignal.features.model_extractor import ModelFeatureExtractor
from neuralsignal.features.runner import collect_features, collect_features_batched
from neuralsignal.storage.bundle import create_bundle, upload_bundle
from neuralsignal.storage.local import LocalFeatureShardWriter
from neuralsignal.storage.manifests import RunManifest
from neuralsignal.storage.s3 import Boto3ObjectStore, s3_join

logger = logging.getLogger(__name__)


def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", help="Optional config path. Env config is used when omitted.")
    args = parser.parse_args()
    logger.info("remote job starting run_id=%s", args.run_id)
    config = load_config(args.config) if args.config else _load_config_from_env()
    run_dir = Path(os.environ.get("NEURALSIGNAL_RUN_WORKDIR", "/workspace/neuralsignal-runs")) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _log_config_summary(config, run_dir)

    manifest = RunManifest(
        run_id=args.run_id,
        dataset=config.get("dataset") or {},
        features={"schema_version": "features.v1", "materialized_sets": (config.get("features") or {}).get("materialize", [])},
        state="running",
    )
    writer = LocalFeatureShardWriter(run_dir, manifest)
    source = _dataset_source(config).iter_examples()
    logger.info("feature collection starting")
    if ((config.get("extraction") or {}).get("mode") or "placeholder") == "model":
        logger.info("model extractor initializing")
        extractor = ModelFeatureExtractor(config)
        logger.info("model extractor ready")
        collect_features_batched(source, config, writer, extractor.extract_batch)
    else:
        collect_features(source, config, writer, _placeholder_extractor)
    logger.info("feature collection completed rows=%s shards=%s", manifest.rows["written"], len(manifest.shards))
    manifest.state = "completed"
    manifest.write_json(run_dir / "manifest.json")
    logger.info("manifest written path=%s", run_dir / "manifest.json")

    bundle_uri = _bundle_uri(config, args.run_id)
    logger.info("creating bundle uri=%s", bundle_uri)
    bundle = create_bundle(run_dir, run_dir.parent / f"{args.run_id}.zip")
    logger.info("bundle created path=%s bytes=%s", bundle, bundle.stat().st_size)
    logger.info("uploading bundle to s3")
    upload_bundle(_s3_store(), bundle, bundle_uri)
    logger.info("bundle upload completed uri=%s", bundle_uri)
    print(json.dumps({"run_id": args.run_id, "bundle_uri": bundle_uri, "rows": manifest.rows["written"]}))


def _configure_logging() -> None:
    level = os.environ.get("NEURALSIGNAL_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


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
    encoded = os.environ.get("NEURALSIGNAL_FEATURE_CONFIG_B64")
    if not encoded:
        raise RuntimeError("NEURALSIGNAL_FEATURE_CONFIG_B64 is required")
    return json.loads(base64.b64decode(encoded).decode("utf-8"))


def _dataset_source(config: dict):
    return source_from_config(config)


def _bundle_uri(config: dict, run_id: str) -> str:
    run = config.get("run") or {}
    if run.get("bundle_uri"):
        return str(run["bundle_uri"])
    base = run.get("s3_output_uri") or f"s3://{os.environ['NEURALSIGNAL_S3_BUCKET']}/feature-runs"
    return s3_join(str(base), run_id, "bundle.zip")


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

