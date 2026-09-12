from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from neuralsignal.console import configure_logging
from neuralsignal.config import load_config
from neuralsignal.datasets.sources.jsonl import JsonlSource
from neuralsignal.features.model_extractor import ModelFeatureExtractor
from neuralsignal.features.runner import collect_features, collect_features_batched
from neuralsignal.remote.lifecycle import RemoteCollectCancelled, remote_collect_lifecycle
from neuralsignal.remote.sync import sync_feature_run
from neuralsignal.storage.local import LocalFeatureShardWriter
from neuralsignal.storage.manifests import RunManifest
from neuralsignal.storage.s3 import Boto3ObjectStore
from neuralsignal.training import train_s1


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except RemoteCollectCancelled as error:
        print(str(error), file=sys.stderr)
        return 130 if error.terminated else 1
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130


def _main(argv: list[str] | None = None) -> int:
    _configure_logging()
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
    remote_collect.add_argument("config", nargs="?", help="Feature collection config YAML, or remote collect launch config YAML.")
    remote_collect.add_argument("--launch-config", help="Remote collect launch config YAML.")
    remote_collect.add_argument("--manifest", help="RunPod pod manifest YAML.")
    remote_collect.add_argument("--run-id", help="Stable id for this feature run and bundle.")
    remote_collect.add_argument("--secrets-file", help="Optional KEY=VALUE file forwarded to the RunPod worker environment.")
    remote_collect.add_argument("--env-file", help="Local .env file for RunPod, AWS/S3, MinIO, and MLflow settings.")
    remote_collect.add_argument("--terraform-dir", help="Terraform folder used to read S3 handoff outputs.")
    remote_collect.add_argument("--target-dir", help="Local directory where the downloaded bundle is unpacked.")
    remote_collect.add_argument("--train-config", help="Optional S1 training config to run after bundle download.")
    remote_collect.add_argument("--minio-uri", help="Optional MinIO/S3 URI where unpacked datasets and artifacts are mirrored.")
    gpu_selection = remote_collect.add_mutually_exclusive_group()
    gpu_selection.add_argument("-gb", "--gpu-vram-gb", type=int, help="Target GPU VRAM in GB; list available GPUs within 25%% with prices.")
    gpu_selection.add_argument("--gpu-id", help="Exact RunPod GPU id to request, bypassing interactive GPU selection.")
    remote_collect.add_argument("--yes", action="store_true", default=None, help="Choose the cheapest priced GPU matching the VRAM range without prompting.")
    remote_collect.add_argument("--poll-seconds", type=float, help="Seconds between S3 completion checks.")
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


def _configure_logging() -> None:
    configure_logging()


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
    options = _remote_collect_options(args)
    result = remote_collect_lifecycle(
        options["config"],
        options["manifest"],
        options["run_id"],
        options["target_dir"],
        env_file=options["env_file"],
        secrets_file=options["secrets_file"],
        terraform_dir=options["terraform_dir"] or None,
        train_config_path=options["train_config"],
        minio_uri=options["minio_uri"],
        poll_seconds=options["poll_seconds"],
        timeout_seconds=options["timeout_seconds"],
        dry_run=options["dry_run"],
        gpu_vram_gb=options["gpu_vram_gb"],
        gpu_id=options["gpu_id"],
        yes=options["yes"],
    )
    print(json.dumps(result if isinstance(result, dict) else result.__dict__, indent=2, sort_keys=True))
    return 0


def _remote_collect_options(args) -> dict:
    defaults = {
        "manifest": "configs/runpod_manifest.yaml",
        "env_file": ".env",
        "terraform_dir": "infra/terraform/s3-handoff",
        "target_dir": "runs/remote",
        "poll_seconds": 30,
        "dry_run": False,
        "secrets_file": None,
        "train_config": None,
        "minio_uri": None,
        "timeout_seconds": 1800,
        "gpu_vram_gb": None,
        "gpu_id": None,
        "yes": False,
    }
    launch = {}
    config = args.config
    if args.launch_config:
        loaded = load_config(args.launch_config)
        launch = loaded.get("remote_collect") or loaded
    elif config:
        loaded = load_config(config)
        if "remote_collect" in loaded:
            launch = loaded["remote_collect"]
            config = launch.get("config")

    options = {**defaults, **launch}
    for key in (*defaults.keys(), "config", "run_id"):
        value = getattr(args, key, None)
        if value is not None and not (key == "dry_run" and value is False):
            options[key] = value
    if args.gpu_vram_gb is not None:
        options["gpu_id"] = None
    elif args.gpu_id is not None:
        options["gpu_vram_gb"] = None
    if options["gpu_vram_gb"] is not None and options["gpu_vram_gb"] <= 0:
        raise SystemExit("-gb must be a positive number of GB")
    options["config"] = config or options.get("config")
    missing = [key for key in ("config", "run_id") if not options.get(key)]
    if missing:
        raise SystemExit("missing remote collect option(s): " + ", ".join(missing))
    return options


def _placeholder_extractor(example, spec) -> dict[str, float]:
    return {"input_chars": float(len(example.input)), "output_chars": float(len(example.output))}


if __name__ == "__main__":
    raise SystemExit(main())
