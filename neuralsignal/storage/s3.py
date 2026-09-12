from __future__ import annotations

from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse


class ObjectStore(Protocol):
    def download_file(self, bucket: str, key: str, path: str) -> None:
        ...

    def upload_file(self, path: str, bucket: str, key: str) -> None:
        ...

    def delete_file(self, bucket: str, key: str) -> None:
        ...

    def exists(self, bucket: str, key: str) -> bool:
        ...


class Boto3ObjectStore:
    def __init__(
        self,
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
    ) -> None:
        import boto3

        kwargs = {"endpoint_url": endpoint_url}
        if access_key_id and secret_access_key:
            kwargs["aws_access_key_id"] = access_key_id
            kwargs["aws_secret_access_key"] = secret_access_key
        self.client = boto3.client("s3", **kwargs)

    def download_file(self, bucket: str, key: str, path: str) -> None:
        self.client.download_file(bucket, key, path)

    def upload_file(self, path: str, bucket: str, key: str) -> None:
        self.client.upload_file(path, bucket, key)

    def delete_file(self, bucket: str, key: str) -> None:
        self.client.delete_object(Bucket=bucket, Key=key)

    def exists(self, bucket: str, key: str) -> bool:
        try:
            self.client.head_object(Bucket=bucket, Key=key)
            return True
        except Exception as error:
            from botocore.exceptions import ClientError

            if isinstance(error, ClientError) and error.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def s3_join(base_uri: str, *parts: str) -> str:
    bucket, prefix = parse_s3_uri(base_uri)
    clean = [prefix.strip("/"), *(part.strip("/") for part in parts if part)]
    return f"s3://{bucket}/{'/'.join(part for part in clean if part)}"


def upload_file(store: ObjectStore, local_path: str | Path, s3_uri: str) -> None:
    bucket, key = parse_s3_uri(s3_uri)
    store.upload_file(str(local_path), bucket, key)


def download_file(store: ObjectStore, s3_uri: str, local_path: str | Path) -> None:
    bucket, key = parse_s3_uri(s3_uri)
    path = Path(local_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    store.download_file(bucket, key, str(path))


def delete_file(store: ObjectStore, s3_uri: str) -> None:
    bucket, key = parse_s3_uri(s3_uri)
    store.delete_file(bucket, key)


def exists(store: ObjectStore, s3_uri: str) -> bool:
    bucket, key = parse_s3_uri(s3_uri)
    return store.exists(bucket, key)


def upload_directory(store: ObjectStore, local_dir: str | Path, s3_uri: str) -> None:
    root = Path(local_dir)
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        upload_file(store, path, s3_join(s3_uri, path.relative_to(root).as_posix()))