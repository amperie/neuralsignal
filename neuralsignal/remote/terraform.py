from __future__ import annotations

import json
import subprocess
from pathlib import Path


def terraform_outputs(terraform_dir: str | Path) -> dict[str, str]:
    proc = subprocess.run(
        ["terraform", f"-chdir={Path(terraform_dir)}", "output", "-json"],
        check=True,
        capture_output=True,
        text=True,
    )
    raw = json.loads(proc.stdout or "{}")
    return {
        key: str(value.get("value"))
        for key, value in raw.items()
        if isinstance(value, dict) and value.get("value") is not None
    }


def s3_settings_from_terraform(terraform_dir: str | Path | None) -> dict[str, str]:
    if terraform_dir is None:
        return {}
    try:
        outputs = terraform_outputs(terraform_dir)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    values = {}
    if "bucket_name" in outputs:
        values["NEURALSIGNAL_S3_BUCKET"] = outputs["bucket_name"]
    if "aws_region" in outputs:
        values["AWS_DEFAULT_REGION"] = outputs["aws_region"]
    if "runpod_access_key_id" in outputs:
        values["AWS_ACCESS_KEY_ID"] = outputs["runpod_access_key_id"]
    if "runpod_secret_access_key" in outputs:
        values["AWS_SECRET_ACCESS_KEY"] = outputs["runpod_secret_access_key"]
    return values

