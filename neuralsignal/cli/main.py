from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuralsignal.config import load_config
from neuralsignal.datasets.sources.jsonl import JsonlSource
from neuralsignal.features.model_extractor import ModelFeatureExtractor
from neuralsignal.features.runner import collect_features, collect_features_batched
from neuralsignal.remote.lifecycle import remote_collect_lifecycle
from neuralsignal.remote.sync import sync_feature_run
from neuralsignal.storage.local import LocalFeatureShardWriter
from neuralsignal.storage.manifests import RunManifest
from neuralsignal.storage.s3 import Boto3ObjectStore
from neuralsignal.training import train_s1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ns",
        description="NeuralSignal v2 tools for dataset import, feature collection, remote RunPod jobs, and local S1 training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    dataset = sub.add_parser(
        "dataset",
        help="Inspect or import dataset sources.",
        description="Dataset utilities. The v2 importer currently supports local JSONL files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_import = dataset_sub.add_parser(
        "import",
        help="Validate a dataset source and report row count.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    dataset_import.add_argument("source", choices=["jsonl"], help="Dataset source type.")
    dataset_import.add_argument("path", help="Path to the source file.")

    features = sub.add_parser(
        "features",
        help="Collect materialized feature datasets.",
        description="Feature collection commands. These write feature shards and manifests without saving raw scans.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    features_sub = features.add_subparsers(dest="features_command", required=True)
    collect_local = features_sub.add_parser(
        "collect-local",
        help="Run local feature collection from JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    collect_local.add_argument("config", help="Feature collection config YAML.")
    collect_local.add_argument("--input-jsonl", required=True, help="Input JSONL dataset with id/input/output/label fields.")
    collect_local.add_argument("--out", required=True, help="Directory where feature shards and manifest files are written.")

    train = sub.add_parser(
        "train",
        help="Train local S1 models.",
        description="Local S1 training commands. Training logs metrics, params, and artifacts to MLflow when configured.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_sub = train.add_subparsers(dest="train_command", required=True)
    train_s1_cmd = train_sub.add_parser(
        "s1",
        help="Train an S1 classifier from a feature dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_s1_cmd.add_argument("config", help="S1 training config YAML.")

    remote = sub.add_parser(
        "remote",
        help="Run and synchronize remote feature jobs.",
        description="RunPod lifecycle commands for feature collection and S3 handoff bundles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    remote_sub = remote.add_subparsers(dest="remote_command", required=True)
    remote_collect = remote_sub.add_parser(
        "collect",
        help="Launch RunPod, wait for the S3 bundle, download it, and optionally train S1.",
        description=(
            "Full v2 lifecycle: launch a RunPod feature job, wait for its S3 bundle, terminate the pod, "
            "unpack locally, mirror to MinIO, and optionally train/log an S1 model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    remote_collect.add_argument("config", help="Feature collection config YAML sent to the RunPod worker.")
    remote_collect.add_argument("--manifest", default="configs/runpod_manifest.yaml", help="RunPod pod manifest YAML.")
    remote_collect.add_argument("--run-id", required=True, help="Stable id for this feature run and bundle.")
    remote_collect.add_argument("--secrets-file", help="Optional KEY=VALUE file forwarded to the RunPod worker environment.")
    remote_collect.add_argument("--env-file", default=".env", help="Local .env file for RunPod, AWS/S3, MinIO, and MLflow settings.")
    remote_collect.add_argument("--terraform-dir", default="infra/terraform/s3-handoff", help="Terraform folder used to read S3 handoff outputs.")
    remote_collect.add_argument("--target-dir", default="runs/remote", help="Local directory where the downloaded bundle is unpacked.")
    remote_collect.add_argument("--train-config", help="Optional S1 training config to run after bundle download.")
    remote_collect.add_argument("--minio-uri", help="Optional MinIO/S3 URI where unpacked datasets and artifacts are mirrored.")
    remote_collect.add_argument("--poll-seconds", type=float, default=30, help="Seconds between S3 completion checks.")
    remote_collect.add_argument("--timeout-seconds", type=float, help="Maximum seconds to wait for the S3 bundle.")
    remote_collect.add_argument("--dry-run", action="store_true", help="Build and print a redacted RunPod payload without launching.")
    remote_sync = remote_sub.add_parser(
        "sync",
        help="Download an existing remote feature run from S3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    remote_sync.add_argument("remote_run_uri", help="S3 URI for the remote run prefix or bundle.")
    remote_sync.add_argument("local_dir", help="Local directory where synced files are written.")

    args = parser.parse_args(argv)
    if args.command == "dataset":
        return _dataset_import(args)
    if args.command == "features":
        return _collect_local(args)
    if args.command == "train":
        return _train_s1(args)
    if args.command == "remote":
        return _remote(args)
    return 1


def _dataset_import(args) -> int:
    rows = [example.to_row() for example in JsonlSource(args.path).iter_examples()]
    print(json.dumps({"source": args.source, "rows": len(rows)}))
    return 0


def _collect_local(args) -> int:
    config = load_config(args.config)
    out = Path(args.out)
    manifest = RunManifest(
        run_id=str((config.get("run") or {}).get("name") or "local-feature-run"),
        dataset=config.get("dataset") or {"source": "jsonl"},
        features={"schema_version": "features.v1"},
    )
    writer = LocalFeatureShardWriter(out, manifest)
    source = JsonlSource(args.input_jsonl).iter_examples()
    if ((config.get("extraction") or {}).get("mode") or "placeholder") == "model":
        extractor = ModelFeatureExtractor(config)
        collect_features_batched(source, config, writer, extractor.extract_batch)
    else:
        collect_features(source, config, writer, _placeholder_extractor)
    print(json.dumps({"run_id": manifest.run_id, "rows": manifest.rows["written"], "out": str(out)}))
    return 0


def _train_s1(args) -> int:
    config = load_config(args.config)
    result = train_s1(
        (config.get("dataset") or {})["path"],
        label_column=(config.get("dataset") or {}).get("label_column", "label"),
        feature_config=config.get("features") or {},
        mlflow_config=config.get("mlflow") or {},
    )
    print(json.dumps({"metrics": result.metrics, "features": result.feature_columns}))
    return 0


def _remote(args) -> int:
    if args.remote_command == "sync":
        sync_feature_run(Boto3ObjectStore(), args.remote_run_uri, args.local_dir)
        return 0
    result = remote_collect_lifecycle(
        args.config,
        args.manifest,
        args.run_id,
        args.target_dir,
        env_file=args.env_file,
        secrets_file=args.secrets_file,
        terraform_dir=args.terraform_dir or None,
        train_config_path=args.train_config,
        minio_uri=args.minio_uri,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
        dry_run=args.dry_run,
    )
    print(json.dumps(result if isinstance(result, dict) else result.__dict__, indent=2, sort_keys=True))
    return 0


def _placeholder_extractor(example, spec) -> dict[str, float]:
    return {"input_chars": float(len(example.input)), "output_chars": float(len(example.output))}


if __name__ == "__main__":
    raise SystemExit(main())
