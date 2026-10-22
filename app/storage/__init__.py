"""Object storage for uploaded files (Phase 3, task 1).

Files live on the local filesystem (`LocalStorage`; in docker-compose a volume
shared by the api and worker containers) behind one small `StorageProvider`
interface, so another backend can be added later. Object keys are NEVER taken
from a client:

    tenants/<tenant_id>/uploads/<upload_id>/data.csv

is built by `upload_key()` from the authenticated tenant and a server-generated
upload id, and every provider call re-checks the key with `assert_key_for_tenant`
(prefix match, no `..`, no absolute paths), so a stored or forged key from one
tenant can't reach another tenant's objects through this layer.
"""

from __future__ import annotations

import re
import uuid
from typing import BinaryIO, Protocol

from app.core.tenancy import validate_tenant_id

_KEY_RE = re.compile(r"^tenants/[a-z0-9][a-z0-9_-]{0,47}/uploads/[0-9a-f-]{36}/[A-Za-z0-9._-]{1,64}$")


class StorageError(Exception):
    """The storage backend failed (transient: callers may retry)."""


class StorageKeyError(ValueError):
    """The key is not a well-formed key of the claimed tenant (never retry)."""


def new_upload_id() -> str:
    return str(uuid.uuid4())


def upload_key(tenant_id: str, upload_id: str, name: str = "data.csv") -> str:
    validate_tenant_id(tenant_id)
    uuid.UUID(upload_id)  # raises ValueError if not a UUID
    key = f"tenants/{tenant_id}/uploads/{upload_id}/{name}"
    assert_key_for_tenant(key, tenant_id)
    return key


def assert_key_for_tenant(key: str, tenant_id: str) -> str:
    """Return `key` if it is a well-formed upload key belonging to `tenant_id`."""
    if not isinstance(key, str) or not _KEY_RE.match(key) or ".." in key:
        raise StorageKeyError(f"malformed storage key {key!r}")
    if not key.startswith(f"tenants/{tenant_id}/uploads/"):
        raise StorageKeyError("storage key does not belong to this tenant")
    return key


class StorageProvider(Protocol):
    def put(self, key: str, fileobj: BinaryIO, *, tenant_id: str) -> None: ...
    def open(self, key: str, *, tenant_id: str) -> BinaryIO: ...
    def exists(self, key: str, *, tenant_id: str) -> bool: ...
    def delete(self, key: str, *, tenant_id: str) -> None: ...


def get_storage() -> StorageProvider:
    from app.core.ingest_settings import ingest_settings
    from app.storage.local import LocalStorage
    return LocalStorage(ingest_settings().storage_local_dir)