"""Collect complete agent trajectories with the existing remote extraction pipeline."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from neuralsignal.remote.lifecycle import remote_collect_lifecycle
from neuralsignal.remote.terraform import s3_settings_from_terraform
from neuralsignal.storage.s3 import Boto3ObjectStore


class BenchmarkStore(Boto3ObjectStore):
    def exists(self, bucket, key):
        present = super().exists(bucket, key)
        if not present and key.endswith('/bundle.zip'):
            manifest = key.rsplit('/', 1)[0] + '/manifest.json'
            if super().exists(bucket, manifest):
                with self.client.get_object(Bucket=bucket, Key=manifest)['Body'] as body:
                    if json.load(body).get('state') == 'failed':
                        raise RuntimeError('Remote feature extraction failed; terminating pod without waiting for a bundle')
        return present


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/remote/agent_full_context_bs1.yaml"))
    parser.add_argument("--gpu-id")
    parser.add_argument("--timeout", type=int, default=43200)
    parser.add_argument("--ssh-key", type=Path, default=Path.home()/".ssh/codex_z440_ed25519")
    args = parser.parse_args()
    if "h200" in (args.gpu_id or "").lower():
        parser.error("H200 provisioning is prohibited by the user")
    load_dotenv(".env")
    settings = s3_settings_from_terraform("infra/terraform/s3-handoff")
    for key, value in settings.items():
        os.environ.setdefault(key, value)
    target = Path("runs/remote") / args.run_id
    if target.exists():
        raise FileExistsError(target)
    config = yaml.safe_load(args.config.read_text())
    config = config.get("feature_config", config)
    if config["generation"].get("truncation_length", 0):
        raise ValueError("Agent benchmark collection requires truncation_length=0; content must not be discarded")
    audit = json.loads(args.input.with_suffix(".audit.json").read_text())
    if audit["sha256"] != hashlib.sha256(args.input.read_bytes()).hexdigest():
        raise ValueError("Input checksum differs from source audit")
    bucket = os.environ["NEURALSIGNAL_S3_BUCKET"]
    store = BenchmarkStore(endpoint_url=os.environ.get("NEURALSIGNAL_S3_ENDPOINT_URL") or None)
    key = f"feature-runs/{args.run_id}/inputs/normalized.jsonl"
    store.upload_file(str(args.input), bucket, key)
    # Private, short-lived handoff; revoked by deleting this object after extraction.
    url = store.client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=86400)
    config["dataset"] = {"source": "huggingface", "name": "json", "split": "train",
                         "load_kwargs": {"data_files": {"train": url}}}
    config["run"] = {"name": args.run_id, "s3_output_uri": f"s3://{bucket}/feature-runs"}
    path = Path("runs/datasets") / f"{args.run_id}.collection.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    manifest = yaml.safe_load(Path("configs/runpod_manifest.yaml").read_text())
    public_key = Path(str(args.ssh_key) + ".pub")
    if public_key.exists():
        manifest.setdefault("env", {})["SSH_PUBLIC_KEY"] = public_key.read_text().strip()
    manifest_path = path.with_suffix(".manifest.yaml")
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    print(f"Starting existing remote collector: {args.run_id}", flush=True)
    try:
        result = remote_collect_lifecycle(path, manifest_path, args.run_id,
                                         "runs/remote", store=store, timeout_seconds=args.timeout,
                                         gpu_vram_gb=24 if not args.gpu_id else None, gpu_id=args.gpu_id, yes=True, ssh_key_path=args.ssh_key)
        audit.update(model=config["model"], generation=config["generation"], prompt=config["prompt"],
                     full_trajectory_single_pass=True, pod_id=result.pod_id)
        (target / "source_audit.json").write_text(json.dumps(audit, indent=2))
        print(f"Completed {result.run_id}: {result.target_dir}", flush=True)
    finally:
        store.delete_file(bucket, key)
        config["dataset"]["load_kwargs"]["data_files"] = {"train": str(args.input)}
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        resolved = Path("runs/remote/.launch-configs") / args.run_id / "feature_config.yaml"
        if resolved.exists():
            resolved.write_text(yaml.safe_dump(config), encoding="utf-8")


if __name__ == "__main__":
    main()
