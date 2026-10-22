"""Storage providers and key safety (Phase 3, task 1). No database needed."""

from __future__ import annotations

import io
import uuid

import pytest

from app.storage import (StorageError, StorageKeyError, assert_key_for_tenant, new_upload_id,
                         upload_key)
from app.storage.local import LocalStorage


def test_keys_are_built_server_side_under_the_tenants_prefix():
    uid = new_upload_id()
    key = upload_key("acme", uid)
    assert key == f"tenants/acme/uploads/{uid}/data.csv"
    with pytest.raises(ValueError):
        upload_key("acme", "../../etc")            # not a UUID
    with pytest.raises(ValueError):
        upload_key("../evil", uid)                 # invalid tenant id


@pytest.mark.parametrize("key", [
    "tenants/acme/uploads/../../other/uploads/x/data.csv", "/etc/passwd", "data.csv",
    f"tenants/acme/uploads/{uuid.uuid4()}/../data.csv", f"tenants/acme/uploads/{uuid.uuid4()}/a/b.csv",
    f"tenants/other/uploads/{uuid.uuid4()}/data.csv", "",
])
def test_malformed_or_foreign_keys_are_refused(key):
    with pytest.raises(StorageKeyError):
        assert_key_for_tenant(key, "acme")


def test_local_round_trip_and_tenant_check(tmp_path):
    store = LocalStorage(tmp_path)
    key = upload_key("acme", new_upload_id())
    assert not store.exists(key, tenant_id="acme")
    store.put(key, io.BytesIO(b"a,b\n1,2\n"), tenant_id="acme")
    assert store.exists(key, tenant_id="acme")
    with store.open(key, tenant_id="acme") as f:
        assert f.read() == b"a,b\n1,2\n"
    assert not [p for p in (tmp_path / key).parent.iterdir() if p.name.startswith(".part")]
    with pytest.raises(StorageKeyError):
        store.open(key, tenant_id="other")
    store.delete(key, tenant_id="acme")
    assert not store.exists(key, tenant_id="acme")
    with pytest.raises(StorageError):
        store.open(key, tenant_id="acme")