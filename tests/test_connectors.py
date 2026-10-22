"""
Phase 5 acceptance tests: the scheduled pull connector, against a FAKE object store.

No cloud call is made. The GCS adapter itself is covered by tests/test_connectors_gcs.py (a local
fake of the GCS HTTP API). Real Postgres (as `iia_app`, `iia_worker`), real Spark for one end-to-end
job; needs DATABASE_URL, APP_DATABASE_URL, WORKER_DATABASE_URL and Java, skips cleanly without them.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from itertools import count
from pathlib import Path

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

from app.api.connectors import get_secrets, get_store_factory
from app.connectors import scheduler
from app.connectors.base import (
    ConnectorAuthError,
    ConnectorTransientError,
    ObjectGoneError,
    ObjectInfo,
)
from app.connectors.puller import run_pull
from app.core.secrets_provider import EnvFileSecretsProvider
from app.core.security import create_token, hash_password
from app.core.tenancy import generate_api_key, hash_api_key
from app.database.tenant_context import bind_tenant, install
from app.jobs import worker
from app.jobs.pipeline import tenant_conn
from app.main import app
from app.storage import get_storage
from tests.helpers import env
from tests.test_company_b import B_MAPPING, N, _b_export

load_dotenv()

TA, TB = "p5-a", "p5-b"
TENANTS = (TA, TB)
KEYS = {t: generate_api_key() for t in TENANTS}
JWT_SECRET = "connectors-test-secret-" + "0123456789abcdef" * 2
SECRET_MARK = "SENTINEL-PRIVATE-KEY-91d7"
SECRET = json.dumps({"type": "service_account", "client_email": "sa-91d7@x.iam.gserviceaccount.com",
                     "private_key": f"-----BEGIN PRIVATE KEY-----\n{SECRET_MARK}\n-----END PRIVATE KEY-----\n"})
SECRETS = EnvFileSecretsProvider(env={"CONNECTOR_SECRET__TEST_GCS": SECRET}, directory=Path("/nonexistent"))
RESPONSES: list[str] = []          # every API response body, scanned by the leak test


# ---------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------

class FakeStore:
    """In-memory ObjectStore with switchable failures."""

    _versions = count(1)

    def __init__(self) -> None:
        self.objs: dict[str, dict] = {}
        self.auth_error = False
        self.list_error: Exception | None = None
        self.fail_download: dict[str, Exception] = {}
        self.download_delay = 0.0
        self.downloads: list[str] = []

    def put(self, key: str, data: bytes, age_seconds: float = 86400, version: str | None = None) -> ObjectInfo:
        updated = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=age_seconds)
        version = version or str(next(self._versions))
        self.objs[key] = {"version": version, "data": data, "updated": updated}
        return ObjectInfo(key, version, len(data), updated)

    def check_access(self) -> None:
        if self.auth_error:
            raise ConnectorAuthError()

    def list_objects(self, prefix: str):
        if self.auth_error:
            raise ConnectorAuthError()
        if self.list_error:
            raise self.list_error
        for key, o in sorted(self.objs.items()):
            if key.startswith(prefix):
                yield ObjectInfo(key, o["version"], len(o["data"]), o["updated"])

    def download(self, info: ObjectInfo, fileobj, max_bytes: int) -> None:
        if info.key in self.fail_download:
            raise self.fail_download[info.key]
        time.sleep(self.download_delay)
        o = self.objs.get(info.key)
        if o is None or o["version"] != info.version:
            raise ObjectGoneError()
        self.downloads.append(info.key)
        fileobj.write(o["data"])


class Factory:
    """A StoreFactory that always returns one FakeStore and records the secret it was given."""

    def __init__(self, store: FakeStore) -> None:
        self.store, self.secrets_seen = store, []

    def __call__(self, connector: dict, secret: str | None) -> FakeStore:
        self.secrets_seen.append(secret)
        return self.store


def _cleanup(conn) -> None:
    for table in ("audit_log", "users", "connector_objects", "connector_runs", "connectors", "rejected_rows",
                  "jobs", "uploads", "ingestion_checkpoints", "sensor_readings", "machine_anomalies",
                  "sensor_summary", "machines", "sensor_registry", "tenant_settings"):
        conn.execute(text(f"DELETE FROM {table} WHERE tenant_id = ANY(:t)"), {"t": list(TENANTS)})
    conn.execute(text("DELETE FROM tenants WHERE tenant_id = ANY(:t)"), {"t": list(TENANTS)})


@pytest.fixture(scope="module")
def admin():
    missing = [v for v in ("DATABASE_URL", "APP_DATABASE_URL", "WORKER_DATABASE_URL") if not __import__("os").environ.get(v)]
    if missing:
        pytest.skip(f"{', '.join(missing)} not set")
    import os
    engine = create_engine(os.environ["DATABASE_URL"])
    with engine.begin() as conn:
        _cleanup(conn)
        for t in TENANTS:
            conn.execute(text("INSERT INTO tenants (tenant_id, name, api_key_hash) VALUES (:t, :n, :h)"),
                         {"t": t, "n": f"{t} (test)", "h": hash_api_key(KEYS[t])})
    yield engine
    with engine.begin() as conn:
        _cleanup(conn)
    engine.dispose()


@pytest.fixture(scope="module")
def wengine(admin):
    import os
    engine = install(create_engine(os.environ["WORKER_DATABASE_URL"]))
    yield engine
    engine.dispose()


@pytest.fixture(scope="module", autouse=True)
def stack(admin, tmp_path_factory):
    with env(STORAGE_LOCAL_DIR=str(tmp_path_factory.mktemp("storage")), STORAGE_BACKEND="local",
             JOB_BACKOFF_BASE_SECONDS="0", UPLOAD_RATE_LIMIT_PER_MINUTE="1000", API_KEY=None,
             AUTH_REQUIRED="false", JWT_SECRET=JWT_SECRET, JWT_TTL_MINUTES="60",
             STORAGE_EMULATOR_HOST=None, CONNECTOR_MIN_AGE_SECONDS="60",
             CONNECTOR_MAX_OBJECTS_PER_RUN="25", CONNECTOR_OBJECT_MAX_ATTEMPTS="5",
             CONNECTOR_MAX_PER_TENANT="100"):
        yield


@pytest.fixture(scope="module")
def factory_store():
    store = FakeStore()
    factory = Factory(store)
    app.dependency_overrides[get_store_factory] = lambda: factory
    app.dependency_overrides[get_secrets] = lambda: SECRETS
    yield store, factory
    app.dependency_overrides.pop(get_store_factory, None)
    app.dependency_overrides.pop(get_secrets, None)


@pytest.fixture(scope="module")
def client(stack, factory_store):
    with TestClient(app) as c:
        yield c


def _user_hdr(admin_engine, tenant: str, role: str) -> dict[str, str]:
    uid = str(uuid.uuid4())
    with admin_engine.begin() as conn:
        conn.execute(text("INSERT INTO users (id, tenant_id, email, password_hash, role) "
                          "VALUES (:i, :t, :e, :h, :r)"),
                     {"i": uid, "t": tenant, "e": f"{uid[:8]}@example.com", "h": hash_password("x" * 16), "r": role})
    return {"Authorization": "Bearer " + create_token(user_id=uid, tenant_id=tenant, secret=JWT_SECRET,
                                                       ttl_minutes=60)}


@pytest.fixture(scope="module")
def adm(admin):
    return {TA: _user_hdr(admin, TA, "admin"), TB: _user_hdr(admin, TB, "admin")}


def call(client, method: str, url: str, **kw):
    r = getattr(client, method)(url, **kw)
    RESPONSES.append(r.text)
    return r


def new_connector(client, adm, tenant=TA, name=None, **over) -> dict:
    body = dict(name=name or f"c-{uuid.uuid4().hex[:6]}", bucket="acme-bucket", prefix="in/",
                pattern="*.csv", schedule="*/15 * * * *", secret_ref="test-gcs", mapping=B_MAPPING)
    body.update(over)
    r = call(client, "post", "/connectors", json=body, headers=adm[tenant])
    assert r.status_code == 201, r.text
    return r.json()


def pull(wengine, factory_store, tenant: str, cid: str, **kw):
    store, factory = factory_store
    return run_pull(wengine, cid, tenant, storage=get_storage(), store_factory=factory,
                    secrets=kw.pop("secrets", SECRETS), **kw)


def q(admin, sql: str, **params):
    with admin.connect() as conn:
        return conn.execute(text(sql), params).all()


def scalar(admin, sql: str, **params):
    return q(admin, sql, **params)[0][0]


def ledger(admin, cid: str) -> dict[str, str]:
    return {r[0] + "@" + r[1]: r[2] for r in q(
        admin, "SELECT object_key, object_version, status FROM connector_objects WHERE connector_id = :c", c=cid)}


def jobs_for(admin, tenant: str) -> int:
    return scalar(admin, "SELECT COUNT(*) FROM jobs WHERE tenant_id = :t", t=tenant)


def fresh_store(factory_store) -> FakeStore:
    store, _ = factory_store
    store.objs.clear()
    store.auth_error, store.list_error = False, None
    store.fail_download.clear()
    store.download_delay = 0.0
    store.downloads.clear()
    return store


SMALL = b"AssetTag,ReadTime,Area,PressKPa,AmpsMilli,BrgTempF\nPUMP-A101,2026-03-01T00:00:00,Pump house 1,400,25000,150\n"


# ---------------------------------------------------------------------
# 1. End to end: pull -> upload + job -> worker -> data
# ---------------------------------------------------------------------

def test_happy_path_pull_queues_the_phase_3_job_and_the_worker_loads_the_data(admin, wengine, client, adm, factory_store):
    store = fresh_store(factory_store)
    export = _b_export().to_csv(index=False).encode()
    info = store.put("in/pumps.csv", export)
    c = new_connector(client, adm, name="happy")
    assert c["last_status"] == "never" and c["enabled"] and c["next_run_at"]

    result = pull(wengine, factory_store, TA, c["connector_id"])
    assert (result.status, result.seen, result.new, result.queued, result.failed) == ("ok", 1, 1, 1, 0)

    # one upload (mapping snapshot copied), one queued job, one ledger row tying them together
    up = q(admin, "SELECT u.status, u.size_bytes, u.mapping = c.mapping, u.filename FROM uploads u, connectors c "
                  "WHERE u.tenant_id = :t AND c.id = :c", t=TA, c=c["connector_id"])
    assert up == [("mapped", len(export), True, "pumps.csv")]
    assert ledger(admin, c["connector_id"]) == {f"in/pumps.csv@{info.version}": "queued"}
    assert scalar(admin, "SELECT status FROM jobs WHERE tenant_id = :t", t=TA) == "queued"

    assert worker.process_one(wengine) is True and worker.process_one(wengine) is False
    job = q(admin, "SELECT status, rows_in, rows_loaded, rows_rejected, error FROM jobs WHERE tenant_id = :t", t=TA)[0]
    assert job == ("succeeded", 3 * N, 3 * 3 * N, 0, None)
    assert scalar(admin, "SELECT COUNT(*) FROM sensor_readings WHERE tenant_id = :t", t=TA) == 3 * 3 * N
    assert scalar(admin, "SELECT COUNT(*) FROM sensor_readings WHERE tenant_id = :t", t=TB) == 0
    # the timestamp checkpoint is still only the pipeline's data high-water mark
    assert [r[0] for r in q(admin, "SELECT source FROM ingestion_checkpoints WHERE tenant_id = :t", t=TA)] \
        == ["sensor_summary"]

    # connector state and history
    row = call(client, "get", f"/connectors/{c['connector_id']}", headers=adm[TA]).json()
    assert row["last_status"] == "ok" and row["last_message"] == "1 file(s) queued"
    assert row["next_run_at"] > datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    runs = call(client, "get", f"/connectors/{c['connector_id']}/runs", headers=adm[TA]).json()
    assert len(runs) == 1 and runs[0]["objects_queued"] == 1 and runs[0]["trigger"] == "schedule"
    # run audited as the system principal
    assert q(admin, "SELECT principal_kind, actor, outcome FROM audit_log WHERE tenant_id = :t "
                    "AND action = 'connector.pull'", t=TA) == [("system", "scheduler", "success")]


# ---------------------------------------------------------------------
# 2. Configuration API
# ---------------------------------------------------------------------

def test_configuration_is_validated_and_never_accepts_credentials_or_endpoints(client, adm, admin):
    base = dict(name="v", bucket="acme-bucket", prefix="in/", pattern="*.csv", schedule="*/15 * * * *",
                secret_ref="test-gcs", mapping=B_MAPPING)
    bad = [
        dict(bucket="Not_A_Bucket!"), dict(bucket="a"), dict(prefix="/abs"), dict(prefix="a/../b"),
        dict(pattern="../*"), dict(pattern="*;rm"), dict(schedule="* * * *"), dict(schedule="nonsense a b c d"),
        dict(schedule="* * * * *"),                                # more often than every 5 minutes
        dict(secret_ref=None), dict(secret_ref="Has Space"), dict(secret_ref="../etc/passwd"),
        dict(mapping=None), dict(mapping={"nope": 1}), dict(mapping_from_upload_id=str(uuid.uuid4())),
        dict(endpoint_url="http://postgres:5432"), dict(credentials="x"), dict(private_key="x"),
        dict(type="s3"),
    ]
    for over in bad:
        r = call(client, "post", "/connectors", json={**base, **over}, headers=adm[TA])
        assert r.status_code == 422, (over, r.status_code, r.text)
    ok = new_connector(client, adm, name="dup")
    assert call(client, "post", "/connectors", json={**base, "name": "dup"}, headers=adm[TA]).status_code == 409
    assert "mapping" not in ok and "config" not in ok and "private_key" not in json.dumps(ok)
    # update: partial, validated, and a changed source resets the listing cursor
    cid = ok["connector_id"]
    with admin.begin() as conn:
        conn.execute(text("UPDATE connectors SET list_watermark = '2026-01-01' WHERE id = :c"), {"c": cid})
    assert call(client, "post", f"/connectors/{cid}", json={"prefix": "../x"}, headers=adm[TA]).status_code == 422
    assert call(client, "post", f"/connectors/{cid}", json={}, headers=adm[TA]).status_code == 422
    r = call(client, "post", f"/connectors/{cid}", json={"prefix": "other/", "schedule": "0 * * * *"}, headers=adm[TA])
    assert r.status_code == 200 and r.json()["prefix"] == "other/" and r.json()["schedule"] == "0 * * * *"
    assert scalar(admin, "SELECT list_watermark FROM connectors WHERE id = :c", c=cid) is None
    # mapping from a confirmed upload
    with admin.begin() as conn:
        conn.execute(text("INSERT INTO uploads (upload_id, tenant_id, filename, content_type, storage_key, status, mapping) "
                          "VALUES (:u, :t, 'a.csv', 'text/csv', :k, 'mapped', CAST(:m AS JSONB))"),
                     {"u": "00000000-0000-0000-0000-0000000000a1", "t": TA, "m": json.dumps(B_MAPPING),
                      "k": "tenants/p5-a/uploads/00000000-0000-0000-0000-0000000000a1/data.csv"})
    r = call(client, "post", "/connectors", headers=adm[TA], json={
        **base, "name": "from-upload", "mapping": None, "mapping_from_upload_id": "00000000-0000-0000-0000-0000000000a1"})
    assert r.status_code == 201, r.text
    # enable/disable/delete
    cid2 = r.json()["connector_id"]
    d = call(client, "post", f"/connectors/{cid2}/disable", headers=adm[TA]).json()
    assert d["enabled"] is False and d["next_run_at"] is None
    e = call(client, "post", f"/connectors/{cid2}/enable", headers=adm[TA]).json()
    assert e["enabled"] is True and e["next_run_at"]
    assert call(client, "post", f"/connectors/{cid2}/delete", headers=adm[TA]).status_code == 204
    assert call(client, "get", f"/connectors/{cid2}", headers=adm[TA]).status_code == 404
    # per-tenant cap
    with env(CONNECTOR_MAX_PER_TENANT="1"):
        assert call(client, "post", "/connectors", json={**base, "name": "over"}, headers=adm[TA]).status_code == 409
    with admin.begin() as conn:
        conn.execute(text("DELETE FROM uploads WHERE upload_id = '00000000-0000-0000-0000-0000000000a1'"))


# ---------------------------------------------------------------------
# 3. Idempotency, overlap, bad files, watermark
# ---------------------------------------------------------------------

def test_repeated_and_overlapping_pulls_create_no_duplicates_and_a_new_version_is_a_new_file(admin, wengine, client, adm, factory_store):
    store = fresh_store(factory_store)
    c = new_connector(client, adm, name="idem")
    cid = c["connector_id"]
    store.put("in/a.csv", SMALL, age_seconds=7200)
    store.put("in/b.csv", SMALL, age_seconds=3600)
    store.put("in/ignored.txt", SMALL)                      # not matching *.csv
    store.put("other/c.csv", SMALL)                         # outside the prefix
    first = pull(wengine, factory_store, TA, cid)
    assert (first.seen, first.queued) == (2, 2)
    before = (jobs_for(admin, TA), scalar(admin, "SELECT COUNT(*) FROM uploads WHERE tenant_id = :t", t=TA))
    for _ in range(3):                                      # the overlap window re-lists both files
        again = pull(wengine, factory_store, TA, cid)
        assert (again.new, again.queued, again.failed, again.message) == (0, 0, 0, "no new files")
    assert (jobs_for(admin, TA), scalar(admin, "SELECT COUNT(*) FROM uploads WHERE tenant_id = :t", t=TA)) == before
    assert len(ledger(admin, cid)) == 2
    store.put("in/b.csv", SMALL + b"x", age_seconds=1800)   # overwritten: new generation
    third = pull(wengine, factory_store, TA, cid)
    assert (third.new, third.queued) == (1, 1)
    assert jobs_for(admin, TA) == before[0] + 1 and len(ledger(admin, cid)) == 3
    wm = scalar(admin, "SELECT list_watermark FROM connectors WHERE id = :c", c=cid)
    assert wm == store.objs["in/b.csv"]["updated"]


def test_one_bad_file_never_blocks_the_others_and_the_watermark_waits_for_it(admin, wengine, client, adm, factory_store):
    store = fresh_store(factory_store)
    cid = new_connector(client, adm, name="bad-among-good")["connector_id"]
    g1 = store.put("in/1-good.csv", SMALL, age_seconds=9000)
    store.put("in/2-bad.csv", SMALL, age_seconds=8000)
    g3 = store.put("in/3-good.csv", SMALL, age_seconds=7000)
    store.fail_download["in/2-bad.csv"] = ConnectorTransientError()
    r1 = pull(wengine, factory_store, TA, cid)
    assert (r1.status, r1.queued, r1.failed) == ("partial", 2, 1)
    assert r1.message == "2 file(s) queued, 1 file(s) failed"
    states = ledger(admin, cid)
    assert sorted(states.values()) == ["queued", "queued", "retry"]
    # the cursor stops BEFORE the failed file (it will be seen again), but the good file after it was queued
    assert scalar(admin, "SELECT list_watermark FROM connectors WHERE id = :c", c=cid) == g1.updated
    assert q(admin, "SELECT last_error FROM connector_objects WHERE status = 'retry' AND connector_id = :c", c=cid) \
        == [("the storage service is temporarily unavailable",)]
    # fixed: the retry is queued, nothing else is repeated, the cursor moves past everything
    del store.fail_download["in/2-bad.csv"]
    before = jobs_for(admin, TA)
    r2 = pull(wengine, factory_store, TA, cid)
    assert (r2.status, r2.queued, r2.failed) == ("ok", 1, 0) and jobs_for(admin, TA) == before + 1
    assert sorted(ledger(admin, cid).values()) == ["queued"] * 3
    assert scalar(admin, "SELECT list_watermark FROM connectors WHERE id = :c", c=cid) == g3.updated


def test_permanent_failures_are_rejected_without_holding_the_cursor_and_retries_are_bounded(admin, wengine, client, adm, factory_store):
    store = fresh_store(factory_store)
    cid = new_connector(client, adm, name="permanent")["connector_id"]
    store.put("in/1-empty.csv", b"", age_seconds=9000)
    store.put("in/2-huge.csv", b"x" * 400, age_seconds=8000)
    store.put("in/3-flaky.csv", SMALL, age_seconds=7000)
    store.put("in/4-good.csv", SMALL, age_seconds=6000)
    store.fail_download["in/3-flaky.csv"] = ConnectorTransientError()
    with env(UPLOAD_MAX_BYTES="300", CONNECTOR_OBJECT_MAX_ATTEMPTS="2"):
        r1 = pull(wengine, factory_store, TA, cid)
        states = {k.split("@")[0]: v for k, v in ledger(admin, cid).items()}
        assert states == {"in/1-empty.csv": "rejected", "in/2-huge.csv": "rejected",
                          "in/3-flaky.csv": "retry", "in/4-good.csv": "queued"}
        assert (r1.queued, r1.failed) == (1, 3)
        # rejected files do not hold the cursor; the flaky one does
        assert scalar(admin, "SELECT list_watermark FROM connectors WHERE id = :c", c=cid) \
            == store.objs["in/2-huge.csv"]["updated"]
        r2 = pull(wengine, factory_store, TA, cid)            # second failed attempt -> rejected
        assert r2.failed == 1 and ledger(admin, cid)[f"in/3-flaky.csv@{store.objs['in/3-flaky.csv']['version']}"] == "rejected"
        assert q(admin, "SELECT attempts FROM connector_objects WHERE object_key = 'in/3-flaky.csv' AND connector_id = :c", c=cid) == [(2,)]
    assert scalar(admin, "SELECT list_watermark FROM connectors WHERE id = :c", c=cid) \
        == store.objs["in/4-good.csv"]["updated"]


def test_young_files_wait_and_a_per_run_cap_defers_the_rest(admin, wengine, client, adm, factory_store):
    store = fresh_store(factory_store)
    cid = new_connector(client, adm, name="young-cap")["connector_id"]
    store.put("in/young.csv", SMALL, age_seconds=5)           # may still be uploading
    r = pull(wengine, factory_store, TA, cid)
    assert (r.seen, r.new, r.queued) == (1, 0, 0) and scalar(admin, "SELECT list_watermark FROM connectors WHERE id = :c", c=cid) is None
    later = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=10)
    assert pull(wengine, factory_store, TA, cid, now=later).queued == 1
    for i in range(3):
        store.put(f"in/cap-{i}.csv", SMALL, age_seconds=400 - i)
    with env(CONNECTOR_MAX_OBJECTS_PER_RUN="2"):
        assert pull(wengine, factory_store, TA, cid).queued == 2
        assert pull(wengine, factory_store, TA, cid).queued == 1
    assert pull(wengine, factory_store, TA, cid).queued == 0


# ---------------------------------------------------------------------
# 4. Credentials, disabled connectors, run-now, test-connection
# ---------------------------------------------------------------------

def test_bad_credentials_and_missing_secret_fail_the_run_with_a_clear_secret_free_status(admin, wengine, client, adm, factory_store):
    store = fresh_store(factory_store)
    cid = new_connector(client, adm, name="creds")["connector_id"]
    store.put("in/a.csv", SMALL)
    store.auth_error = True
    r = pull(wengine, factory_store, TA, cid)
    assert (r.status, r.queued) == ("failed", 0) and r.message == "the storage service rejected the credentials"
    assert call(client, "post", f"/connectors/{cid}/test", headers=adm[TA]).json() == {
        "ok": False, "message": "the storage service rejected the credentials"}
    row = call(client, "get", f"/connectors/{cid}", headers=adm[TA]).json()
    assert row["last_status"] == "failed" and row["next_run_at"]       # still scheduled to try again
    assert len(ledger(admin, cid)) == 0
    store.auth_error = False
    assert call(client, "post", f"/connectors/{cid}/test", headers=adm[TA]).json() == {"ok": True, "message": "access OK"}
    nosecret = new_connector(client, adm, name="no-secret", secret_ref="missing-ref")["connector_id"]
    r = pull(wengine, factory_store, TA, nosecret)
    assert r.status == "failed" and r.message == "the configured secret is not available"
    assert call(client, "post", f"/connectors/{nosecret}/test", headers=adm[TA]).json()["ok"] is False
    # test-connection ingests nothing
    assert len(ledger(admin, cid)) == 0


def test_disabled_connectors_are_not_run_and_run_now_goes_through_the_scheduler(admin, wengine, client, adm, factory_store):
    store = fresh_store(factory_store)
    factory = factory_store[1]
    cid = new_connector(client, adm, name="sched")["connector_id"]
    store.put("in/a.csv", SMALL)
    with admin.begin() as conn:
        conn.execute(text("UPDATE connectors SET next_run_at = '2020-01-01' WHERE id = :c"), {"c": cid})
    call(client, "post", f"/connectors/{cid}/disable", headers=adm[TA])
    with admin.begin() as conn:                                        # even if a past time is left behind
        conn.execute(text("UPDATE connectors SET next_run_at = '2020-01-01' WHERE id = :c"), {"c": cid})
    scheduler.tick(wengine, storage=get_storage(), store_factory=factory, secrets=SECRETS)
    assert scalar(admin, "SELECT COUNT(*) FROM connector_runs WHERE connector_id = :c", c=cid) == 0
    assert call(client, "post", f"/connectors/{cid}/run", headers=adm[TA]).status_code == 409
    call(client, "post", f"/connectors/{cid}/enable", headers=adm[TA])
    r = call(client, "post", f"/connectors/{cid}/run", headers=adm[TA])
    assert r.status_code == 202
    assert scalar(admin, "SELECT run_requested FROM connectors WHERE id = :c", c=cid) is True
    scheduler.tick(wengine, storage=get_storage(), store_factory=factory, secrets=SECRETS)
    runs = call(client, "get", f"/connectors/{cid}/runs", headers=adm[TA]).json()
    assert [(x["trigger"], x["status"], x["objects_queued"]) for x in runs] == [("manual", "ok", 1)]
    assert scalar(admin, "SELECT run_requested FROM connectors WHERE id = :c", c=cid) is False
    nxt = scalar(admin, "SELECT next_run_at FROM connectors WHERE id = :c", c=cid)
    assert nxt > datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------
# 5. Two schedulers racing
# ---------------------------------------------------------------------

def test_two_schedulers_claim_disjoint_connectors(admin, wengine):
    ids = [str(uuid.uuid4()) for _ in range(8)]
    with admin.begin() as conn:
        for i, cid in enumerate(ids):
            conn.execute(text("INSERT INTO connectors (id, tenant_id, name, type, config, schedule, mapping, secret_ref, "
                              "next_run_at) VALUES (:i, :t, :n, 'gcs', '{}', '0 * * * *', '{}', 'x', '2020-01-01')"),
                         {"i": cid, "t": TB, "n": f"race-{i}"})
    got: list[list] = [[], []]
    barrier = threading.Barrier(2)

    def go(n: int) -> None:
        barrier.wait()
        for _ in range(10):                      # bounded: a broken claim fails the test, never hangs it
            claimed = scheduler.claim_due(wengine, 900, 2)
            if not claimed:
                return
            got[n].extend(c[0] for c in claimed)

    threads = [threading.Thread(target=go, args=(n,)) for n in (0, 1)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    claimed = got[0] + got[1]
    assert sorted(claimed) == sorted(ids), "every due connector is claimed exactly once"
    assert scheduler.claim_due(wengine, 900, 5) == [], "a leased connector is not claimed again"
    # a lease that expired (crashed scheduler) is claimable again
    with admin.begin() as conn:
        conn.execute(text("UPDATE connectors SET claimed_at = NOW() AT TIME ZONE 'UTC' - interval '1 hour' "
                          "WHERE id = :c"), {"c": ids[0]})
    assert [c[0] for c in scheduler.claim_due(wengine, 900, 5)] == [ids[0]]
    with admin.begin() as conn:
        conn.execute(text("DELETE FROM connectors WHERE id = ANY(CAST(:ids AS UUID[]))"), {"ids": ids})


def test_two_pulls_of_the_same_connector_at_once_queue_each_file_once(admin, wengine, client, adm, factory_store):
    store = fresh_store(factory_store)
    cid = new_connector(client, adm, name="race-pull")["connector_id"]
    store.put("in/a.csv", SMALL, age_seconds=5000)
    store.put("in/b.csv", SMALL, age_seconds=4000)
    store.download_delay = 0.3
    uploads_before = scalar(admin, "SELECT COUNT(*) FROM uploads WHERE tenant_id = :t", t=TA)
    jobs_before = jobs_for(admin, TA)
    results: list = []
    barrier = threading.Barrier(2)

    def go() -> None:
        barrier.wait()
        results.append(pull(wengine, factory_store, TA, cid))

    threads = [threading.Thread(target=go) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(r.queued for r in results) == 2, [r.__dict__ for r in results]
    assert scalar(admin, "SELECT COUNT(*) FROM uploads WHERE tenant_id = :t", t=TA) == uploads_before + 2
    assert jobs_for(admin, TA) == jobs_before + 2
    assert sorted(ledger(admin, cid).values()) == ["queued", "queued"]
    stored = list((Path(get_storage().root) / "tenants" / TA / "uploads").glob("*/data.csv"))
    assert len(stored) == scalar(admin, "SELECT COUNT(*) FROM uploads WHERE tenant_id = :t", t=TA), \
        "the losing pull removed its copies"


# ---------------------------------------------------------------------
# 6. Tenant isolation, roles, database grants
# ---------------------------------------------------------------------

def test_tenants_cannot_see_or_touch_each_others_connectors_or_ledgers(admin, wengine, client, adm, factory_store):
    store = fresh_store(factory_store)
    a = new_connector(client, adm, TA, name="iso-a")["connector_id"]
    b = new_connector(client, adm, TB, name="iso-b")["connector_id"]
    store.put("in/a.csv", SMALL)
    assert pull(wengine, factory_store, TA, a).queued == 1
    assert pull(wengine, factory_store, TB, b).queued == 1          # B pulls its own copy of the same key
    assert {c["connector_id"] for c in call(client, "get", "/connectors", headers=adm[TB]).json()} == {b}
    for method, suffix in (("get", ""), ("post", ""), ("post", "/enable"), ("post", "/disable"), ("post", "/run"),
                           ("post", "/test"), ("get", "/runs"), ("post", "/delete")):
        kw = {"json": {"name": "hijack"}} if (method, suffix) == ("post", "") else {}
        r = call(client, method, f"/connectors/{a}{suffix}", headers=adm[TB], **kw)
        assert r.status_code == 404, (method, suffix, r.status_code)
    assert call(client, "post", "/connectors", headers=adm[TB], json={
        "name": "x", "bucket": "acme-bucket", "schedule": "0 * * * *", "secret_ref": "test-gcs",
        "mapping_from_upload_id": scalar(admin, "SELECT upload_id FROM uploads WHERE tenant_id = :t LIMIT 1", t=TA).__str__(),
    }).status_code == 422                                          # cannot borrow A's mapping either
    # Postgres RLS, as the worker role
    with tenant_conn(wengine, TB) as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM connectors WHERE id = :a"), {"a": a}).scalar() == 0
        assert conn.execute(text("SELECT COUNT(*) FROM connector_objects WHERE connector_id = :a"), {"a": a}).scalar() == 0
        assert conn.execute(text("SELECT COUNT(*) FROM connector_runs WHERE connector_id = :a"), {"a": a}).scalar() == 0
        assert conn.execute(text("SELECT COUNT(DISTINCT tenant_id) FROM connector_objects")).scalar() == 1
    assert scalar(admin, "SELECT COUNT(*) FROM jobs j JOIN connector_objects o ON o.job_id = j.job_id "
                         "WHERE o.tenant_id <> j.tenant_id") == 0
    assert ledger(admin, a) and ledger(admin, b)


def test_only_admins_may_configure_connectors(client, adm, admin):
    viewer = _user_hdr(admin, TA, "viewer")
    cid = new_connector(client, adm, name="roles")["connector_id"]
    routes = [("get", "/connectors"), ("get", f"/connectors/{cid}"), ("get", f"/connectors/{cid}/runs"),
              ("post", "/connectors"), ("post", f"/connectors/{cid}"), ("post", f"/connectors/{cid}/run"),
              ("post", f"/connectors/{cid}/test"), ("post", f"/connectors/{cid}/enable"),
              ("post", f"/connectors/{cid}/disable"), ("post", f"/connectors/{cid}/delete")]
    for method, url in routes:
        kw = {"json": {}} if method == "post" else {}
        assert call(client, method, url, headers=viewer, **kw).status_code == 403, ("viewer", url)
        assert call(client, method, url, headers={"X-API-Key": KEYS[TA]}, **kw).status_code == 403, ("api key", url)
        assert call(client, method, url, **kw).status_code == 401, ("anonymous", url)
    assert call(client, "get", f"/connectors/{cid}", headers=adm[TA]).status_code == 200
    # writes are audited with the acting admin, denials too
    assert scalar(admin, "SELECT COUNT(*) FROM audit_log WHERE tenant_id = :t AND action = 'POST /connectors' "
                         "AND outcome = 'success'", t=TA) >= 1
    assert scalar(admin, "SELECT COUNT(*) FROM audit_log WHERE tenant_id = :t AND action = 'permission.denied' "
                         "AND resource LIKE '/connectors%'", t=TA) >= 10


def test_the_api_role_cannot_forge_run_state_and_the_worker_cannot_exceed_its_grants(admin, client, adm):
    import os
    cid = new_connector(client, adm, name="grants")["connector_id"]
    app_engine = install(create_engine(os.environ["APP_DATABASE_URL"]))
    try:
        with tenant_conn(app_engine, TA) as conn:
            assert conn.execute(text("UPDATE connectors SET name = 'renamed-by-api' WHERE id = :c"), {"c": cid}).rowcount == 1
        for sql in ("UPDATE connectors SET last_status = 'ok' WHERE id = :c",
                    "UPDATE connectors SET claimed_at = NULL WHERE id = :c",
                    "INSERT INTO connector_runs (id, tenant_id, connector_id, trigger) "
                    "VALUES (gen_random_uuid(), 'p5-a', :c, 'manual')",
                    "INSERT INTO connector_objects (tenant_id, connector_id, object_key, object_version, status) "
                    "VALUES ('p5-a', :c, 'k', '1', 'queued')"):
            with pytest.raises(ProgrammingError):
                with tenant_conn(app_engine, TA) as conn:
                    conn.execute(text(sql), {"c": cid})
    finally:
        app_engine.dispose()
    assert scalar(admin, "SELECT has_table_privilege('iia_worker', 'connectors', 'DELETE')") is False
    assert scalar(admin, "SELECT has_function_privilege('iia_app', 'iia_claim_due_connectors(integer, integer)', 'EXECUTE')") is False


# ---------------------------------------------------------------------
# 7. Secrets never leak
# ---------------------------------------------------------------------

def test_a_planted_secret_never_appears_in_logs_responses_errors_audit_or_any_table(admin, wengine, client, adm, factory_store, caplog):
    store = fresh_store(factory_store)
    factory = factory_store[1]
    factory.secrets_seen.clear()
    caplog.set_level(logging.DEBUG)
    cid = new_connector(client, adm, name="leak", secret_ref="test-gcs")["connector_id"]
    store.put("in/a.csv", SMALL, age_seconds=9000)
    store.put("in/b.csv", SMALL, age_seconds=8000)

    # 1. a store that raises an exception whose text contains the secret, while listing
    store.list_error = RuntimeError(f"EXC-TEXT-LIST {SECRET}")
    r = pull(wengine, factory_store, TA, cid)
    assert r.status == "failed" and SECRET_MARK not in (r.message or "")
    store.list_error = None
    # 2. the same while downloading one file, next to a good file
    store.fail_download["in/a.csv"] = RuntimeError(f"EXC-TEXT-DOWNLOAD {SECRET}")
    r = pull(wengine, factory_store, TA, cid)
    assert (r.status, r.queued, r.failed) == ("partial", 1, 1)
    store.fail_download.clear()
    # 3. bad credentials, test-connection, a job whose file is junk (job errors / rejected rows)
    store.auth_error = True
    pull(wengine, factory_store, TA, cid)
    call(client, "post", f"/connectors/{cid}/test", headers=adm[TA])
    store.auth_error = False
    store.put("in/c.csv", b"not,a,sensor,file\n1,2,3,4\n", age_seconds=7000)
    pull(wengine, factory_store, TA, cid)
    worker.process_one(wengine)                   # runs queued jobs (junk file -> permanent job failure)
    worker.process_one(wengine)
    # the secret really reached the connector (so the scan below means something)
    assert SECRET in factory.secrets_seen
    call(client, "get", "/connectors", headers=adm[TA])
    call(client, "get", f"/connectors/{cid}/runs", headers=adm[TA])

    haystack = [caplog.text, *RESPONSES]
    for table in ("audit_log", "connectors", "connector_runs", "connector_objects", "uploads", "jobs", "rejected_rows"):
        haystack += [r[0] for r in q(admin, f"SELECT row_to_json(t)::text FROM {table} t WHERE tenant_id = :t", t=TA)]
    blob = "\n".join(haystack)
    assert blob.strip()
    for needle in (SECRET_MARK, "sa-91d7", "BEGIN PRIVATE KEY", "EXC-TEXT-DOWNLOAD", "EXC-TEXT-LIST"):
        assert needle not in blob, f"{needle!r} leaked"
