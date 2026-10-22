"""S3-compatible StorageProvider (MinIO in docker-compose; any S3 API).

boto3 is imported lazily, so the default local backend (and the test suite)
never needs it. The client is injectable for tests.
"""

from __future__ import annotations

import io
from typing import Any, BinaryIO

from app.storage import StorageError, assert_key_for_tenant


_BUCKETS_READY: set[tuple[str | None, str]] = set()


class S3Storage:
    def __init__(self, client: Any, bucket: str) -> None:
        self.client = client
        self.bucket = bucket

    @classmethod
    def from_settings(cls, s) -> "S3Storage":
        import boto3
        from botocore.config import Config
        client = boto3.client(
            "s3", endpoint_url=s.s3_endpoint_url, region_name=s.s3_region,
            aws_access_key_id=s.s3_access_key, aws_secret_access_key=s.s3_secret_key,
            config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 3}))
        store = cls(client, s.s3_bucket)
        if (s.s3_endpoint_url, s.s3_bucket) not in _BUCKETS_READY:   # once per process
            store.ensure_bucket()
            _BUCKETS_READY.add((s.s3_endpoint_url, s.s3_bucket))
        return store

    def ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception:  # noqa: BLE001 - missing bucket surfaces as ClientError(404)
            try:
                self.client.create_bucket(Bucket=self.bucket)
            except Exception as exc:  # noqa: BLE001
                raise StorageError(f"cannot create bucket {self.bucket!r}: {exc}") from exc

    def put(self, key: str, fileobj: BinaryIO, *, tenant_id: str) -> None:
        assert_key_for_tenant(key, tenant_id)
        try:
            self.client.upload_fileobj(fileobj, self.bucket, key)
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"s3 put failed: {exc}") from exc

    def open(self, key: str, *, tenant_id: str) -> BinaryIO:
        assert_key_for_tenant(key, tenant_id)
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"]
            return io.BytesIO(body.read())      # uploads are capped (UPLOAD_MAX_BYTES), so bounded
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"s3 open failed: {exc}") from exc

    def exists(self, key: str, *, tenant_id: str) -> bool:
        assert_key_for_tenant(key, tenant_id)
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:  # noqa: BLE001 - 404 and transport errors both read as "not there"
            return False

    def delete(self, key: str, *, tenant_id: str) -> None:
        assert_key_for_tenant(key, tenant_id)
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"s3 delete failed: {exc}") from exc
