from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

from neuralsignal.cli.targets import select_target
from neuralsignal.training.targets import TargetError
from neuralsignal.cli.selection import choose_config, choose_run

from neuralsignal.console import color, configure_logging
from neuralsignal.config import load_config
from neuralsignal.datasets.sources.jsonl import JsonlSource
from neuralsignal.features.model_extractor import ModelFeatureExtractor
from neuralsignal.features.selection import extraction_mode
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
    except TargetError as error:
        print(f"Cannot train: {error}", file=sys.stderr)
        return 2
    except RemoteCollectCancelled as error:
        print(str(error), file=sys.stderr)
        return 130 if error.terminated else 1
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Preserve examples, then color the finished layout to retain alignment."""

    def _fill_text(self, text, width, indent):
        if "\n" in text:
            return super()._fill_text(text, width, indent)
        return argparse.HelpFormatter._fill_text(self, text, width, indent)

    def format_help(self):
        rendered = super().format_help()
        lines = []
        for line in rendered.splitlines():
            if line and not line.startswith(" ") and ":" in line:
                heading, rest = line.split(":", 1)
                line = color(heading + ":", "cyan") + rest
            elif line.lstrip().startswith(("ns ", "uv run ns ")):
                line = color(line, "green")
            else:
                line = re.sub(r"(?<![\w-])--?[a-zA-Z][\w-]*", lambda m: color(m[0], "yellow"), line)
            lines.append(line)
        return "\n".join(lines) + "\n"


def _add_remote_options(parser, title="remote options"):
    remote_options = parser.add_argument_group(title)
    remote_options.add_argument("--launch-config", help="Remote collect launch config YAML.")
    remote_options.add_argument("--manifest", help="RunPod pod manifest YAML.")
    remote_options.add_argument("--run-id", help="Stable id for this feature run and bundle.")
    remote_options.add_argument("--secrets-file", help="Optional KEY=VALUE file forwarded to the RunPod worker environment.")
    remote_options.add_argument("--env-file", help="Local .env file for RunPod, AWS/S3, MinIO, and MLflow settings.")
    remote_options.add_argument("--terraform-dir", help="Terraform folder used to read S3 handoff outputs.")
    remote_options.add_argument("--train-config", help="Optional S1 training config to run after bundle download.")
    remote_options.add_argument("--minio-uri", help="Optional MinIO/S3 URI where unpacked datasets and artifacts are mirrored.")
    gpu_selection = remote_options.add_mutually_exclusive_group()
    gpu_selection.add_argument("--gpu-vram-gb", type=int, help="Target GPU VRAM in GB; list available GPUs within 25%% with prices.")
    gpu_selection.add_argument("--gpu-id", help="Exact RunPod GPU id to request, bypassing interactive GPU selection.")
    remote_options.add_argument("--yes", action="store_true", default=None, help="Choose the cheapest priced GPU matching the VRAM range without prompting.")
    remote_options.add_argument("--poll-seconds", type=float, help="Seconds between S3 completion checks.")
    remote_options.add_argument("--timeout-seconds", type=float, help="Maximum seconds to wait for the S3 bundle.")
    remote_options.add_argument("--ssh-key-path", help="Private SSH key path to include in the printed pod login command.")
    remote_options.add_argument("--dry-run", action="store_true", help="Build and print a redacted RunPod payload without launching.")
    return remote_options


def _main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = argparse.ArgumentParser(
        prog="ns", formatter_class=HelpFormatter,
        description="Collect judge-model features and train local S1 classifiers.",
        epilog="""Workflow:
  ns run handles dataset -> RunPod features -> download -> local S1 training.
  Omit config paths to select YAML files from configs/ interactively.

  1. Validate your JSONL input (optional).
  2. Collect features locally or on RunPod using a YAML config path.
  3. Train from a local feature dataset using a training YAML config.

Examples:
  ns run
  ns validate data/examples.jsonl
  ns collect configs/feature_collection/example_runpod_jsonl.yaml --input data/examples.jsonl --out runs/local/example
  ns collect configs/remote/malt_smoke.yaml --remote --run-id smoke-001 --dry-run
  ns train configs/training/s1.yaml --run runs/remote/complete-run
  ns download s3://my-bucket/feature-runs/example --out runs/downloaded/example

Help:
  ns collect -h     Local and remote collection options, defaults, and examples.
  ns train -h       Training data requirements and configuration.
  Run from the repository root; config paths are explicit, with no named presets.
  Colors appear in terminals; NO_COLOR=1 disables them. Pipes stay plain text.
""",
    )
    sub = parser.add_subparsers(dest="command", required=True, title="commands")
    validate = sub.add_parser(
        "validate", help="Check a local JSONL dataset and report its row count.",
        formatter_class=HelpFormatter,
        description="Read a JSONL file through the dataset adapter and report the normalized row count as JSON. No dataset is imported or written.",
        epilog="""Input:
  Use one JSON object per line with id, input, output, labels, and metadata:
  {"id":"example-1","input":"question","output":"answer","labels":["sabotage"],"metadata":{}}
  This checks whether the adapter can read the rows. It does not assess label
  quality or prepare numeric training targets.

Examples:
  ns validate data/examples.jsonl
  uv run ns validate data/examples.jsonl

Next step:
  ns collect configs/feature_collection/example_runpod_jsonl.yaml --input data/examples.jsonl --out runs/local/example
""",
    )
    validate.add_argument("path", metavar="FILE", help="Path to a local JSONL file; no source-type argument is needed.")
    validate.set_defaults(source="jsonl")
    collect = sub.add_parser(
        "collect", help="Collect features locally or on RunPod with --remote.",
        formatter_class=HelpFormatter,
        description="Write feature shards and a manifest from a feature collection YAML. Local execution is the default; --remote launches and manages a RunPod job.",
        epilog="""Local collection:
  Supply a feature config, --input, and --out. Use a fresh output directory;
  existing shards are rejected. Local collection currently reads JSONL only.
  Set extraction.mode: model to extract model activations. Omitted mode uses
  placeholder character counts. Match model device/quantization to your machine;
  the checked-in RunPod example uses CUDA and int8.
  Labels are preserved as labels_json. During training, choose a label or
  map an existing column/metadata field to a binary target.

Remote collection:
  CONFIG may be a feature YAML or a launch YAML containing remote_collect.
  --launch-config explicitly loads launch settings; a supplied feature CONFIG
  overrides its config field. --remote is always required for remote execution.
  Keep this command running: it launches, waits for a checksum-protected bundle,
  terminates the pod, downloads/verifies/extracts results, and optionally mirrors
  to MinIO and trains S1. Local polling does not stream worker logs.
  Publish an image containing your current code and configure credentials first.
  --dry-run prints a redacted payload without launching, but may query Terraform
  outputs and GPU availability or prompt for GPU selection.

Configuration and defaults:
  CLI overrides launch YAML, which overrides defaults. Paths are relative to
  the working directory, not the YAML file. Use a fresh --run-id for each launch;
  it is required unless supplied in launch YAML. No named presets are resolved.
  Remote defaults: manifest=configs/runpod_manifest.yaml, env-file=.env,
  terraform-dir=infra/terraform/s3-handoff, out=runs/remote,
  poll-seconds=30, timeout-seconds=1800. Results live under OUT/RUN_ID.
  --out overrides launch YAML target_dir. Local --out is required.
  GPU selection can prompt; --yes picks the cheapest priced VRAM match.
  --gpu-id requests an exact GPU. Both GPU selectors override the YAML choice.

Examples:
  ns collect configs/feature_collection/example_runpod_jsonl.yaml --input data/examples.jsonl --out runs/local/example
  ns collect configs/remote/malt_smoke.yaml --remote --run-id smoke-001 --dry-run
  ns collect configs/remote/malt_smoke.yaml --remote --run-id smoke-002 --gpu-vram-gb 24 --yes
  ns collect configs/remote/malt_smoke.yaml --remote --run-id smoke-003 --out runs/experiments --timeout-seconds 3600
  ns collect configs/feature_collection/smoke_runpod_malt.yaml --remote --launch-config configs/remote/malt_smoke.yaml --run-id smoke-004

Results:
  Prints JSON with collection results; progress logs and GPU prompts may also
  appear. Remote training_metrics is null when no --train-config is requested.
  See specs/remote-feature-collection-workflow.md for setup and recovery.
""",
    )
    collect.add_argument("config", nargs="?", metavar="CONFIG", help="Feature YAML path, or remote launch YAML path with --remote; omit to choose from configs/ unless --launch-config supplies it.")
    collect.add_argument("--remote", action="store_true", help="Execute collection on RunPod instead of this machine.")
    collect.add_argument("--input", dest="input_jsonl", metavar="FILE", help="Local JSONL input; required for local collection and unavailable with --remote.")
    collect.add_argument("--out", metavar="DIR", help="Local output directory (required locally); remote output root containing RUN_ID (default: runs/remote).")
    remote_options = _add_remote_options(collect, "remote options (require --remote)")
    workflow = sub.add_parser(
        "run", formatter_class=HelpFormatter,
        help="Run the complete dataset -> RunPod features -> download -> S1 workflow.",
        description="Collect the configured dataset on RunPod, download verified features, and train S1 locally. Remote execution and training are automatic.",
        epilog="""Workflow:
  CONFIG is a feature or remote launch YAML. Omit it to choose from configs/.
  The dataset source and limits come from the feature YAML; the worker reads
  the dataset on RunPod. Local files must be available inside the worker image
  or through the configured dataset source. --input is for local collect only.
  --train-config supplies the training YAML; otherwise use launch YAML's
  train_config or choose a training config before launching.
  The downloaded run overrides dataset.path in the training config.
  A unique run ID is generated unless --run-id or launch YAML supplies one.
  Keep the command running through collection, download, and local training.
  Pod termination and bundle verification use the same lifecycle as collect.

Results:
  Features go to OUT/RUN_ID (OUT defaults to runs/remote). Training saves the
  model, metrics, predictions, reports, and selected columns under runs/s1/.
  MLflow reporting is enabled by default. Reporting failures are logged and
  do not discard training results. Training/data failures still stop the run.
  Use a complete labeled dataset; the eight-example smoke YAML is too small
  for meaningful S1 training. --dry-run previews collection without training.

Examples:
  ns run
  ns run configs/feature_collection/example_runpod_malt.yaml --train-config configs/training/s1.yaml --gpu-vram-gb 24 --yes
  ns run configs/remote/malt_smoke.yaml --train-config configs/training/s1.yaml --run-id preview-001 --dry-run

Options:
  Remote execution is automatic. Options above control collection and training.
  --yes selects a GPU only; supply config paths for non-interactive runs.
""",
    )
    workflow.add_argument("config", nargs="?", metavar="CONFIG", help="Feature or remote launch YAML; omit to choose from configs/.")
    workflow.add_argument("--out", metavar="DIR", help="Downloaded feature output root (default: runs/remote); S1 outputs always go under runs/s1/.")
    _add_remote_options(workflow)
    train = sub.add_parser(
        "train", help="Train an S1 classifier from a local feature dataset.",
        formatter_class=HelpFormatter,
        description="Train a local binary S1 classifier using a training YAML file. Prints metrics, selected features, and output directory as JSON. Saves local artifacts and reports to MLflow by default.",
        epilog="""Configuration:
  CONFIG is a training YAML; omit it to choose from configs/.
  --run PATH selects a feature directory or Parquet file and overrides
  dataset.path. Without --run, choose a run from runs/ interactively.
  In scripts, supply both CONFIG and --run explicitly.
  Without target settings, choose an existing column, metadata field, or label
  list, then choose how values become 0 and 1. --target-column and
  --positive-label override YAML settings. Existing dataset.label_column configs
  are still supported. In scripts, conventional numeric label/target columns
  are detected automatically; other mappings require target settings in YAML.
  MALT features are pooled to complete runs before splitting. Too few runs,
  missing labels or conflicting targets produce clear errors. Incomplete MALT
  samples/completions produce warnings; training uses the available features.
  The resolved mapping and class counts are saved in training.json; split.json
  records the training/test assignments.
  features selects included sets/columns and excluded columns.
  model configures the classifier; mlflow configures tracking and artifacts.
  The checked-in s1.yaml uses XGBoost and http://z440.lan:5000 for
  MLflow. Reporting is enabled even without an mlflow section.
  Tracking URI: YAML tracking_uri, then MLFLOW_TRACKING_URI, then
  http://z440.lan:5000. Set mlflow.enabled: false to disable reporting.
  Reporting errors are logged without failing training. Local outputs are
  saved first under runs/s1/<timestamp>-<unique-id>/, including model.ubj,
  metrics.json, selected_features.json, training.json, predictions.csv,
  confusion_matrix.json, and classification_report.json.

Data requirements:
  Label lists require an explicit positive class (interactive or configured).
  MALT training pools completions into samples and then runs before splitting;
  partial smoke-test datasets are unsuitable for training.

Examples:
  ns train configs/training/s1.yaml --run runs/remote/complete-run
  ns train
  uv run ns train configs/training/s1.yaml --run runs/remote/complete-run
  ns download s3://my-bucket/feature-runs/complete-run --out runs/downloaded/complete-run
  ns train configs/training/s1.yaml --run runs/downloaded/complete-run

Details:
  See specs/local-s1-mlflow.md for feature selection, pooling, and MLflow.
""",
    )
    train.add_argument("config", nargs="?", metavar="CONFIG", help="Training YAML path containing dataset, features, model, and optional mlflow settings.")
    train.add_argument("--run", dest="dataset_path", metavar="PATH", help="Feature run directory or Parquet file; omit to choose from runs/. Overrides dataset.path in YAML.")
    target_options = train.add_mutually_exclusive_group()
    target_options.add_argument("--target-column", help="Existing binary column, or metadata.FIELD; overrides target settings in YAML.")
    target_options.add_argument("--positive-label", help="Create 1 for this label and 0 for other labels; uses MALT run labels or generic example labels.")
    download = sub.add_parser(
        "download", help="Download and verify an expanded S3 feature dataset.",
        formatter_class=HelpFormatter,
        description="Download a completed feature run's manifest and shards from an S3 prefix into a local directory. Verify shard checksums before marking the download complete.",
        epilog="""Source layout:
  URI must point to an expanded run prefix containing manifest.json and the
  shard paths it references. A bundle.zip URI is not supported by this command;
  remote collection handles its own bundle download and extraction.
  Uses credentials available to the AWS SDK (environment or configured profile).

Repeated downloads:
  Existing shards with matching checksums are reused. Missing or mismatched
  shards are downloaded again. Success writes .sync-complete to the output.
  Use a dedicated output directory for each run.

Examples:
  ns download s3://my-bucket/feature-runs/example --out runs/downloaded/example
  AWS_PROFILE=research ns download s3://my-bucket/feature-runs/example --out runs/downloaded/example

Next step:
  Pass the downloaded feature run with --run:
  ns train configs/training/s1.yaml --run runs/downloaded/example
""",
    )
    download.add_argument("remote_run_uri", metavar="URI", help="S3 prefix containing manifest.json and expanded shards (not bundle.zip).")
    download.add_argument("--out", dest="local_dir", required=True, metavar="DIR", help="Local directory for the manifest and verified feature shards.")
    args = parser.parse_args(argv)
    if args.command == "validate":
        return _dataset_import(args)
    if args.command == "train":
        return _train_s1(args)
    if args.command == "download":
        args.remote_command = "sync"
        return _remote(args)
    if args.command == "run":
        args.remote = True
    if getattr(args, "remote", False):
        if getattr(args, "input_jsonl", None):
            collect.error("--input is only available for local collection")
        if not args.config and not args.launch_config:
            args.config = choose_config("remote")
        args.target_dir = args.out
        args.remote_command = "collect"
        return _remote(args)
    remote_flags = [action.option_strings[0] for action in remote_options._group_actions
                    if getattr(args, action.dest, None) is not None and getattr(args, action.dest, None) is not False]
    if remote_flags:
        collect.error("these options require --remote: " + ", ".join(remote_flags))
    if not args.config:
        args.config = choose_config("local")
    if not args.input_jsonl or not args.out:
        collect.error("local collection requires CONFIG, --input FILE, and --out DIR")
    return _collect_local(args)


def _configure_logging() -> None:
    configure_logging()


def _dataset_import(args) -> int:
    rows = [example.to_row() for example in JsonlSource(args.path).iter_examples()]
    print(json.dumps({"source": args.source, "rows": len(rows)}))
    return 0


def _collect_local(args) -> int:
    config = load_config(args.config)
    mode = extraction_mode(config)
    out = Path(args.out)
    manifest = RunManifest(
        run_id=str((config.get("run") or {}).get("name") or "local-feature-run"),
        dataset=config.get("dataset") or {"source": "jsonl"},
        features={"schema_version": "features.v1"},
    )
    writer = LocalFeatureShardWriter(out, manifest)
    source = JsonlSource(args.input_jsonl).iter_examples()
    if mode == "model":
        extractor = ModelFeatureExtractor(config)
        collect_features_batched(source, config, writer, extractor.extract_batch)
    else:
        collect_features(source, config, writer, _placeholder_extractor)
    manifest.state = "completed"
    manifest.write_json(out / "manifest.json")
    print(json.dumps({"run_id": manifest.run_id, "rows": manifest.rows["written"], "out": str(out)}))
    return 0


def _train_s1(args) -> int:
    config = load_config(args.config or choose_config("training"))
    dataset_path = args.dataset_path or choose_run()
    target = select_target(dataset_path, config, args.target_column, args.positive_label)
    result = train_s1(
        dataset_path,
        label_column=(config.get("dataset") or {}).get("label_column"),
        target_config=target,
        feature_config=config.get("features") or {},
        mlflow_config=config.get("mlflow") or {},
        model_config=config.get("model") or {},
    )
    print(json.dumps({"metrics": result.metrics, "features": result.feature_columns, "out": result.output_dir}, indent=2))
    return 0


def _remote(args) -> int:
    if args.remote_command == "sync":
        sync_feature_run(Boto3ObjectStore(), args.remote_run_uri, args.local_dir)
        return 0
    options = _remote_collect_options(args)
    if args.command == "run":
        options["train_config"] = options["train_config"] or choose_config("training")
    if options["train_config"]:
        # Catch an unreadable training YAML before launching a paid worker.
        load_config(options["train_config"])
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
        ssh_key_path=options["ssh_key_path"],
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
        "ssh_key_path": None,
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
        raise SystemExit("--gpu-vram-gb must be a positive number of GB")
    options["config"] = config or options.get("config")
    if args.command == "run" and not options.get("run_id"):
        options["run_id"] = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S-") + uuid4().hex[:8]
    missing = [key for key in ("config", "run_id") if not options.get(key)]
    if missing:
        raise SystemExit("missing remote collection option(s): " + ", ".join(missing))
    return options


def _placeholder_extractor(example, spec) -> dict[str, float]:
    return {"input_chars": float(len(example.input)), "output_chars": float(len(example.output))}


if __name__ == "__main__":
    raise SystemExit(main())
