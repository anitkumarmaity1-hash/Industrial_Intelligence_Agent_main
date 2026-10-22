"""Google Cloud Storage ObjectStore (the one connector type implemented in Phase 5).

google-cloud-storage is imported lazily, so importing the app never needs it. The service
endpoint is NOT tenant-configurable (a tenant admin must not be able to point the worker at
an internal host): it is the library default, or STORAGE_EMULATOR_HOST from the deployment's
own environment (fake-gcs-server in docker-compose).

Credentials: the secret is a service-account key JSON. Without a secret the connector may only
run against an emulator: Application Default Credentials would give every tenant the worker's
own identity, i.e. access to buckets that are not theirs.
"""

from __future__ import annotations

import json
import os
from datetime import timezone
from typing import BinaryIO, Iterator

from app.connectors.base import (
    ConnectorAccessError,
    ConnectorAuthError,
    ConnectorError,
    ConnectorTransientError,
    ObjectGoneError,
    ObjectInfo,
    ObjectTooLargeError,
)

SCOPES = ["https://www.googleapis.com/auth/devstorage.read_only"]
TIMEOUT_SECONDS = 30
RETRY_DEADLINE_SECONDS = 30      # the library's own retries (5xx/429/connection resets) are bounded


def _retry():
    from google.cloud.storage.retry import DEFAULT_RETRY
    return DEFAULT_RETRY.with_deadline(RETRY_DEADLINE_SECONDS)


def emulator_configured() -> bool:
    return bool(os.environ.get("STORAGE_EMULATOR_HOST"))


def _translate(exc: Exception, *, during_download: bool = False) -> ConnectorError:
    """Map a google/requests exception to a fixed-message ConnectorError."""
    from google.api_core import exceptions as gexc
    from google.auth import exceptions as aexc

    if isinstance(exc, ConnectorError):
        return exc
    if isinstance(exc, aexc.TransportError):
        return ConnectorTransientError()
    if isinstance(exc, aexc.GoogleAuthError):
        return ConnectorAuthError()
    if isinstance(exc, gexc.Unauthorized):
        return ConnectorAuthError()
    if isinstance(exc, gexc.Forbidden):
        return ConnectorAccessError()
    if isinstance(exc, gexc.NotFound):
        return ObjectGoneError() if during_download else ConnectorAccessError()
    # 5xx, 429, timeouts, connection resets, any other API or transport error
    return ConnectorTransientError()


class GcsObjectStore:
    def __init__(self, client, bucket: str) -> None:
        self.client = client
        self.bucket_name = bucket

    @classmethod
    def from_secret(cls, bucket: str, secret: str | None) -> "GcsObjectStore":
        from google.cloud import storage
        if secret:
            try:
                from google.oauth2 import service_account
                info = json.loads(secret)
                creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
                client = storage.Client(project=info.get("project_id"), credentials=creds)
            except Exception:  # noqa: BLE001 - malformed key: never echo any of it
                raise ConnectorAuthError() from None
        elif emulator_configured():
            client = storage.Client()      # the library uses anonymous credentials for the emulator
        else:
            raise ConnectorAuthError()
        return cls(client, bucket)

    def check_access(self) -> None:
        try:
            list(self.client.list_blobs(self.bucket_name, max_results=1, timeout=TIMEOUT_SECONDS,
                                        retry=_retry()))
        except Exception as exc:  # noqa: BLE001
            raise _translate(exc) from None

    def list_objects(self, prefix: str) -> Iterator[ObjectInfo]:
        try:
            for blob in self.client.list_blobs(self.bucket_name, prefix=prefix or None,
                                               timeout=TIMEOUT_SECONDS, retry=_retry()):
                updated = blob.updated.astimezone(timezone.utc).replace(tzinfo=None)
                yield ObjectInfo(blob.name, str(blob.generation), int(blob.size or 0), updated)
        except Exception as exc:  # noqa: BLE001
            raise _translate(exc) from None

    def download(self, info: ObjectInfo, fileobj: BinaryIO, max_bytes: int) -> None:
        if info.size > max_bytes:
            raise ObjectTooLargeError()
        try:
            blob = self.client.bucket(self.bucket_name).blob(info.key, generation=int(info.version))
            blob.download_to_file(fileobj, timeout=TIMEOUT_SECONDS, retry=_retry())
        except Exception as exc:  # noqa: BLE001
            raise _translate(exc, during_download=True) from None
        if fileobj.tell() > max_bytes:
            raise ObjectTooLargeError()


def make_gcs_store(connector: dict, secret: str | None) -> GcsObjectStore:
    """The default StoreFactory."""
    if connector["type"] != "gcs":
        raise ConnectorAccessError()
    return GcsObjectStore.from_secret(connector["config"]["bucket"], secret)
