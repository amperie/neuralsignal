from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

from neuralsignal.config import load_config
from neuralsignal.datasets.sources.jsonl import JsonlSource
from neuralsignal.features.runner import collect_features
from neuralsignal.storage.bundle import create_bundle, upload_bundle
from neuralsignal.storage.local import LocalFeatureShardWriter
from neuralsignal.storage.manifests import RunManifest
from neuralsignal.storage.s3 import Boto3ObjectStore, s3_join


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", help="Optional config path. Env config is used when omitted.")
    args = parser.parse_args()
    config = load_config(args.config) if args.config else _load_config_from_env()
    run_dir = Path(os.environ.get("NEURALSIGNAL_RUN_WORKDIR", "/workspace/neuralsignal-runs")) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = RunManifest(
        run_id=args.run_id,
        dataset=config.get("dataset") or {},
        features={"schema_version": "features.v1", "materialized_sets": (config.get("features") or {}).get("materialize", [])},
        state="running",
    )
    writer = LocalFeatureShardWriter(run_dir, manifest)
    source = _dataset_source(config)
    collect_features(source.iter_examples(), config, writer, _placeholder_extractor)
    manifest.state = "completed"
    manifest.write_json(run_dir / "manifest.json")

    bundle = create_bundle(run_dir, run_dir.parent / f"{args.run_id}.zip")
    upload_bundle(_s3_store(), bundle, _bundle_uri(config, args.run_id))
    print(json.dumps({"run_id": args.run_id, "bundle_uri": _bundle_uri(config, args.run_id), "rows": manifest.rows["written"]}))


def _load_config_from_env() -> dict:
    encoded = os.environ.get("NEURALSIGNAL_FEATURE_CONFIG_B64")
    if not encoded:
        raise RuntimeError("NEURALSIGNAL_FEATURE_CONFIG_B64 is required")
    return json.loads(base64.b64decode(encoded).decode("utf-8"))


def _dataset_source(config: dict):
    dataset = config.get("dataset") or {}
    if dataset.get("source") == "jsonl" and dataset.get("path"):
        return JsonlSource(dataset["path"])
    raise RuntimeError("remote job currently requires dataset.source=jsonl and dataset.path")


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
    main()