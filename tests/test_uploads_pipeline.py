"""
Phase 3 acceptance tests: CSV upload -> mapping -> background job -> data, end to end.

Real stack: the FastAPI app (as the least-privilege `iia_app` role), a local-filesystem
StorageProvider in a temp dir, the worker's own code (as `iia_worker`), real Spark.

Needs DATABASE_URL (owner, fixtures only), APP_DATABASE_URL (iia_app),
WORKER_DATABASE_URL (iia_worker) and Java; skips cleanly without them.
Company B's export is the Phase 1 fixture (tests/test_company_b.py).
"""

from __future__ import annotations

import io
import os
import uuid
from datetime import datetime, timedelta

import pandas as pd
import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from app.api.rate_limit import limiter
from app.core.anomaly_rules import load_anomaly_rules
from app.core.security import create_token, hash_password
from app.core.tenancy import generate_api_key, hash_api_key
from app.database.tenant_context import bind_tenant, install
from app.jobs import pipeline, worker
from app.main import app
from app.onboarding.sensors import SensorMapping, normalize_sensor_readings, write_bundle
from app.storage import StorageKeyError, get_storage
from tests.helpers import env
from tests.test_company_b import B_MAPPING, FAULT, N, _b_export

load_dotenv()

TA, TB, TS1, TS2 = "p3-a", "p3-b", "p3-split-1", "p3-split-2"
TENANTS = (TA, TB, TS1, TS2)
KEYS = {t: generate_api_key() for t in TENANTS}
CSV = {"Content-Type": "text/csv"}


def _hdr(tenant: str, **extra) -> dict[str, str]:
    return {"X-API-Key": KEYS[tenant], **extra}


JWT_SECRET = "uploads-test-secret-" + "0123456789abcdef" * 2


def _admin_hdr(admin_engine, tenant: str) -> dict[str, str]:
    """Phase 4: retrying a job is an admin permission (a tenant API key has machine scope
    only), so the retry calls below act as a human admin of the tenant."""
    uid = str(uuid.uuid4())
    with admin_engine.begin() as conn:
        conn.execute(text("INSERT INTO users (id, tenant_id, email, password_hash, role) "
                          "VALUES (:i, :t, :e, :h, 'admin')"),
                     {"i": uid, "t": tenant, "e": f"{uid[:8]}@example.com", "h": hash_password("x" * 16)})
    return {"Authorization": "Bearer " + create_token(
        user_id=uid, tenant_id=tenant, secret=JWT_SECRET, ttl_minutes=60)}


def _cleanup(conn) -> None:
    for table in ("audit_log", "users", "rejected_rows", "jobs", "uploads", "ingestion_checkpoints", "sensor_readings",
                  "machine_anomalies", "sensor_summary", "machines", "sensor_registry", "tenant_settings"):
        conn.execute(text(f"DELETE FROM {table} WHERE tenant_id = ANY(:t)"), {"t": list(TENANTS)})
    conn.execute(text("DELETE FROM tenants WHERE tenant_id = ANY(:t)"), {"t": list(TENANTS)})


@pytest.fixture(scope="module")
def admin():
    missing = [v for v in ("DATABASE_URL", "APP_DATABASE_URL", "WORKER_DATABASE_URL") if not os.environ.get(v)]
    if missing:
        pytest.skip(f"{', '.join(missing)} not set")
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
    engine = install(create_engine(os.environ["WORKER_DATABASE_URL"]))
    yield engine
    engine.dispose()


@pytest.fixture(scope="module", autouse=True)
def stack(admin, tmp_path_factory):
    """Local storage in a temp dir, instant retry backoff, generous rate limit."""
    with env(STORAGE_LOCAL_DIR=str(tmp_path_factory.mktemp("storage")), STORAGE_BACKEND="local",
             JOB_BACKOFF_BASE_SECONDS="0", UPLOAD_RATE_LIMIT_PER_MINUTE="1000",
             API_KEY=None, AUTH_REQUIRED="false", JWT_SECRET=JWT_SECRET, JWT_TTL_MINUTES="60"):
        yield


@pytest.fixture(scope="module")
def client(stack):
    with TestClient(app) as c:
        yield c


def drain(wengine, limit: int = 10) -> int:
    n = 0
    while worker.process_one(wengine):
        n += 1
        assert n < limit, "worker did not drain the queue"
    return n


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def submit(client, tenant: str, df: pd.DataFrame | bytes, mapping=B_MAPPING, confirm=True) -> dict:
    """create -> content -> mapping(confirm). Returns the mapping response (with .job)."""
    body = df if isinstance(df, bytes) else _csv(df)
    up = client.post("/uploads", json={"filename": "export.csv"}, headers=_hdr(tenant))
    assert up.status_code == 201, up.text
    uid = up.json()["upload_id"]
    put = client.post(f"/uploads/{uid}/content", content=body, headers=_hdr(tenant, **CSV))
    assert put.status_code == 200, put.text
    r = client.post(f"/uploads/{uid}/mapping", json={"mapping": mapping, "confirm": confirm},
                    headers=_hdr(tenant))
    assert r.status_code == 200, r.text
    out = r.json()
    out["upload_id"] = uid
    return out


def job(client, tenant: str, job_id: str) -> dict:
    r = client.get(f"/jobs/{job_id}", headers=_hdr(tenant))
    assert r.status_code == 200, r.text
    return r.json()


def table(admin, sql: str, **params) -> pd.DataFrame:
    with admin.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def anomalies(admin, tenant: str) -> pd.DataFrame:
    return table(admin, "SELECT machine_id, detected_at, anomaly_score, severity, triggered_reasons "
                        "FROM machine_anomalies WHERE tenant_id = :t ORDER BY machine_id, detected_at", t=tenant)


def summary(admin, tenant: str) -> pd.DataFrame:
    return table(admin, "SELECT machine_id, window_start, window_end, reading_count, anomalous_reading_count, "
                        "max_anomaly_score, health_status FROM sensor_summary WHERE tenant_id = :t "
                        "ORDER BY machine_id, window_start", t=tenant)


def n_readings(admin, tenant: str) -> int:
    return int(table(admin, "SELECT COUNT(*) c FROM sensor_readings WHERE tenant_id = :t", t=tenant)["c"][0])


@pytest.fixture(scope="module")
def b_export() -> pd.DataFrame:
    return _b_export()


# ---------------------------------------------------------------------
# 1. Company B end to end, and parity with the Spark path
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def b_run(admin, wengine, client, b_export):
    out = submit(client, TB, b_export)
    assert out["job"]["status"] == "queued"
    assert drain(wengine) == 1
    return out


def test_company_b_job_succeeds_and_loads_only_its_tenant(admin, client, b_run):
    j = job(client, TB, b_run["job"]["job_id"])
    assert j["status"] == "succeeded", j
    assert (j["rows_in"], j["rows_loaded"], j["rows_rejected"]) == (3 * N, 3 * 3 * N, 0)
    assert j["finished_at"] and j["error"] is None
    assert n_readings(admin, TB) == 3 * 3 * N
    assert n_readings(admin, TA) == 0
    assert set(table(admin, "SELECT DISTINCT machine_id FROM machines WHERE tenant_id = :t", t=TB)["machine_id"]) \
        == {"PUMP-A101", "PUMP-A102", "PUMP-A103"}
    assert table(admin, "SELECT COUNT(*) c FROM machines WHERE tenant_id = :t", t=TA)["c"][0] == 0
    # registry came from the mapping, in the stored (target) units
    reg = table(admin, "SELECT sensor_name, unit FROM sensor_registry WHERE tenant_id = :t", t=TB)
    assert dict(zip(reg["sensor_name"], reg["unit"])) == {"discharge_pressure": "bar", "motor_current": "A",
                                                          "bearing_temp": "C"}


def test_the_injected_faults_are_found(admin, b_run):
    a = anomalies(admin, TB)
    start = datetime(2026, 3, 1) + timedelta(minutes=5 * FAULT.start)
    stop = datetime(2026, 3, 1) + timedelta(minutes=5 * FAULT.stop)
    a101 = a[a.machine_id == "PUMP-A101"]
    assert not a101.empty and a101["detected_at"].between(start, stop).any()
    assert "discharge pressure" in " ".join(a101["triggered_reasons"]).lower()
    a102 = a[a.machine_id == "PUMP-A102"]
    assert "outside its normal range" in " ".join(a102["triggered_reasons"])
    assert not a[(a.machine_id == "PUMP-A103")
                 & (a.detected_at > datetime(2026, 3, 2))].shape[0]


def test_job_output_equals_the_spark_pipeline_run_directly(admin, b_run, b_export, tmp_path):
    """The proof that the worker path is the Spark path: run Phase 1's detect() on the same
    normalised data outside the worker and compare every output row."""
    from spark_jobs.ingestion import get_spark_session
    from spark_jobs.long_format import detect, load_long_readings, load_machines

    mapping = SensorMapping.from_dict(B_MAPPING)
    data = normalize_sensor_readings(b_export, mapping)
    write_bundle(tmp_path, data, mapping)
    with admin.connect() as conn:
        rules = load_anomaly_rules(conn, TB)
    spark = get_spark_session("pytest-parity")
    exp_summary, exp_anoms = detect(load_long_readings(spark, str(tmp_path / "sensor_readings.csv")),
                                    load_machines(spark, str(tmp_path / "machines.csv")), rules)
    # Phase 1's own route: write Parquet, read it back (what load_postgres.py does).
    exp_anoms.write.mode("overwrite").parquet(str(tmp_path / "a.parquet"))
    exp_summary.write.mode("overwrite").parquet(str(tmp_path / "s.parquet"))
    exp_a = pd.read_parquet(tmp_path / "a.parquet").sort_values(["machine_id", "detected_at"]).reset_index(drop=True)
    got_a = anomalies(admin, TB)
    assert len(got_a) == len(exp_a) > 0
    pd.testing.assert_frame_equal(got_a, exp_a[list(got_a.columns)], check_dtype=False)

    exp_s = pd.read_parquet(tmp_path / "s.parquet").sort_values(["machine_id", "window_start"]).reset_index(drop=True)
    got_s = summary(admin, TB)
    assert len(got_s) == len(exp_s) > 0
    pd.testing.assert_frame_equal(got_s, exp_s[list(got_s.columns)], check_dtype=False)


def test_dry_run_preview_and_job_visibility(client, b_export):
    up = client.post("/uploads", json={"filename": r"C:\exports\b.csv"}, headers=_hdr(TB)).json()
    assert up["filename"] == "b.csv" and up["status"] == "created" and "storage_key" not in up
    uid = up["upload_id"]
    assert client.post(f"/uploads/{uid}/preview", headers=_hdr(TB)).status_code == 409   # no content yet
    client.post(f"/uploads/{uid}/content", content=_csv(b_export), headers=_hdr(TB, **CSV))
    p = client.post(f"/uploads/{uid}/preview", json={"rows": 3}, headers=_hdr(TB)).json()
    assert p["rows_returned"] == 3
    types = {c["name"]: c["detected_type"] for c in p["columns"]}
    assert types == {"AssetTag": "text", "ReadTime": "timestamp", "Area": "text",
                     "PressKPa": "number", "AmpsMilli": "number", "BrgTempF": "number"}
    r = client.post(f"/uploads/{uid}/mapping", json={"mapping": B_MAPPING}, headers=_hdr(TB)).json()
    assert r["valid"] and r["job"] is None and r["report"]["total_errors"] == 0
    assert client.get(f"/uploads/{uid}", headers=_hdr(TB)).json()["has_mapping"] is False   # not stored
    bad = client.post(f"/uploads/{uid}/mapping", headers=_hdr(TB),
                      json={"mapping": {**B_MAPPING, "sensors": {"x": {"source": "PressKPa", "unit": "furlongs"}}}})
    assert bad.status_code == 422 and any("unknown unit" in p for p in bad.json()["detail"]["problems"])
    for broken in ({**B_MAPPING, "machine_id": "NoSuchColumn"}, {**B_MAPPING, "production_line": "NoSuchArea"}):
        missing = client.post(f"/uploads/{uid}/mapping", headers=_hdr(TB), json={
            "mapping": broken, "confirm": True})
        assert missing.status_code == 422 and missing.json()["detail"]["missing_columns"]    # cannot apply


# ---------------------------------------------------------------------
# 2. Partial rejects
# ---------------------------------------------------------------------

def _bad_file() -> pd.DataFrame:
    base = datetime(2026, 4, 1)
    frames = []
    for tag in ("M-1", "M-2"):
        frames.append(pd.DataFrame({
            "AssetTag": tag, "ReadTime": [(base + timedelta(hours=i)).isoformat() for i in range(40)],
            "Area": "Line 1", "PressKPa": [400.0 + i for i in range(40)],
            "AmpsMilli": [25000.0 + i for i in range(40)], "BrgTempF": [150.0 + i / 10 for i in range(40)]}))
    df = pd.concat(frames, ignore_index=True).astype(str)
    df.loc[3, "PressKPa"] = "abc"                      # non_numeric (1 reading)
    df.loc[5, "ReadTime"] = "not-a-time"               # bad_timestamp
    df.loc[7, "AssetTag"] = ""                         # missing_machine_id
    df.loc[9] = df.loc[8]                              # exact duplicate -> 3 duplicate_row
    long_id = "X" * 70
    extra = df.iloc[[20, 21]].copy()
    extra["AssetTag"] = long_id                        # 2 rows x 3 sensors rejected as too long
    return pd.concat([df, extra], ignore_index=True)


def test_a_bad_file_loads_the_clean_rows_and_persists_every_reject(admin, wengine, client):
    out = submit(client, TA, _bad_file())
    drain(wengine)
    j = job(client, TA, out["job"]["job_id"])
    assert j["status"] == "succeeded", j
    assert j["rows_in"] == 82
    assert j["rows_loaded"] == 230 and n_readings(admin, TA) == 230     # 240 - 1 - 3 - 3 - 3
    assert j["rows_rejected"] == 1 + 1 + 1 + 3 + 6
    rows = client.get(f"/jobs/{j['job_id']}/rejected-rows?limit=100", headers=_hdr(TA)).json()
    by_code: dict[str, list] = {}
    for r in rows:
        by_code.setdefault(r["code"], []).append(r)
    assert {c: len(v) for c, v in by_code.items()} == {
        "non_numeric": 1, "bad_timestamp": 1, "missing_machine_id": 1, "duplicate_row": 3,
        "machine_id_too_long": 1}
    nn = by_code["non_numeric"][0]
    assert nn["row_number"] == 5 and nn["raw"]["PressKPa"] == "abc" and nn["raw"]["AssetTag"] == "M-1"
    assert by_code["bad_timestamp"][0]["row_number"] == 7
    assert by_code["machine_id_too_long"][0]["row_number"] is None
    assert not (table(admin, "SELECT 1 FROM machines WHERE tenant_id = :t AND length(machine_id) > 64", t=TA)).shape[0]


def test_a_file_with_nothing_usable_fails_permanently_then_a_fixed_file_retries_ok(
        admin, wengine, client, b_export):
    out = submit(client, TA, b_export.head(30))
    jid, uid = out["job"]["job_id"], out["upload_id"]
    storage = get_storage()
    key = table(admin, "SELECT storage_key FROM uploads WHERE upload_id = :u", u=uid)["storage_key"][0]
    storage.put(key, io.BytesIO(b"AssetTag,ReadTime,Area,PressKPa,AmpsMilli,BrgTempF\nP1,garbage,L,x,y,z\n"), tenant_id=TA)
    drain(wengine)
    j = job(client, TA, jid)
    assert j["status"] == "failed" and j["retry_count"] == 0           # permanent: not retried
    assert "no usable readings" in j["error"]
    assert client.get(f"/jobs/{jid}/rejected-rows", headers=_hdr(TA)).json()
    storage.put(key, io.BytesIO(_csv(b_export.head(30))), tenant_id=TA)   # the operator fixes the file
    r = client.post(f"/jobs/{jid}/retry", headers=_hdr(TA))
    assert r.status_code == 403                                         # Phase 4: an API key may not retry
    r = client.post(f"/jobs/{jid}/retry", headers=_admin_hdr(admin, TA))
    assert r.status_code == 202 and r.json()["status"] == "queued"
    drain(wengine)
    j = job(client, TA, jid)
    assert j["status"] == "succeeded" and j["retry_count"] == 1 and j["error"] is None
    assert client.get(f"/jobs/{jid}/rejected-rows", headers=_hdr(TA)).json() == []   # replaced by the retry
    assert client.post(f"/jobs/{jid}/retry", headers=_admin_hdr(admin, TA)).status_code == 409    # only failed jobs


# ---------------------------------------------------------------------
# 3. Idempotency and crash recovery
# ---------------------------------------------------------------------

def test_the_same_upload_twice_changes_nothing(admin, wengine, client, b_export, b_run):
    before = (n_readings(admin, TB), len(anomalies(admin, TB)), len(summary(admin, TB)))
    ck = table(admin, "SELECT last_window_end FROM ingestion_checkpoints WHERE tenant_id = :t", t=TB)
    out = submit(client, TB, b_export)                  # a second upload of the same file
    again = client.post(f"/uploads/{out['upload_id']}/mapping",           # ...and a second job on it
                        json={"mapping": B_MAPPING, "confirm": True}, headers=_hdr(TB))
    assert again.status_code == 409                     # first job still queued
    drain(wengine)
    assert job(client, TB, out["job"]["job_id"])["status"] == "succeeded"
    again = client.post(f"/uploads/{out['upload_id']}/mapping",
                        json={"mapping": B_MAPPING, "confirm": True}, headers=_hdr(TB))
    assert again.status_code == 200
    drain(wengine)
    assert (n_readings(admin, TB), len(anomalies(admin, TB)), len(summary(admin, TB))) == before
    assert table(admin, "SELECT last_window_end FROM ingestion_checkpoints WHERE tenant_id = :t",
                 t=TB).equals(ck)                       # checkpoint is monotonic, never rewound


def test_a_crash_mid_job_then_retry_does_not_double_load(admin, wengine, client, b_export, monkeypatch):
    out = submit(client, TS1, b_export)
    real = pipeline.stage_detect
    calls = {"n": 0}

    def crash_once(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("db-internal-host-7:5432 went away")     # must not leak to the tenant
        return real(*a, **kw)

    monkeypatch.setattr(pipeline, "stage_detect", crash_once)
    assert worker.process_one(wengine)
    j = job(client, TS1, out["job"]["job_id"])
    assert (j["status"], j["retry_count"]) == ("queued", 1)
    assert "db-internal-host-7" not in (j["error"] or "") and "ConnectionError" in j["error"]
    assert n_readings(admin, TS1) == 3 * 3 * N                  # the first attempt had already committed these
    drain(wengine)
    j = job(client, TS1, out["job"]["job_id"])
    assert j["status"] == "succeeded" and j["error"] is None
    assert n_readings(admin, TS1) == 3 * 3 * N
    whole = anomalies(admin, TS1)
    assert len(whole) == len(anomalies(admin, TB)) and not whole.duplicated(["machine_id", "detected_at"]).any()


def test_retries_are_bounded_then_the_job_fails_with_a_generic_error(wengine, client, b_export, monkeypatch):
    out = submit(client, TA, b_export.head(20))
    monkeypatch.setattr(pipeline, "stage_detect", lambda *a, **k: (_ for _ in ()).throw(OSError("secret-path")))
    drain(wengine)                                               # max_retries=3 -> 4 attempts
    j = job(client, TA, out["job"]["job_id"])
    assert j["status"] == "failed" and j["retry_count"] == j["max_retries"] == 3
    assert "secret-path" not in j["error"] and "gave up" in j["error"]


def test_a_job_whose_worker_died_is_reaped(admin, wengine, client, b_export):
    out = submit(client, TA, b_export.head(10))
    jid = out["job"]["job_id"]
    with admin.begin() as conn:
        conn.execute(text("UPDATE jobs SET status='running', stage='detect', "
                          "locked_at = (NOW() AT TIME ZONE 'UTC') - interval '1 hour' WHERE job_id = :j"), {"j": jid})
    assert worker.claim_job(wengine) is None                     # running jobs are not claimable
    assert worker.reap_stale(wengine, 300) == 1
    j = job(client, TA, jid)
    assert (j["status"], j["retry_count"]) == ("queued", 1) and "worker lost" in j["error"]
    with admin.begin() as conn:
        conn.execute(text("UPDATE jobs SET status='running', retry_count = max_retries, "
                          "locked_at = (NOW() AT TIME ZONE 'UTC') - interval '1 hour' WHERE job_id = :j"), {"j": jid})
    assert worker.reap_stale(wengine, 300) == 1
    assert job(client, TA, jid)["status"] == "failed"
    drain(wengine)                                               # nothing left to claim


# ---------------------------------------------------------------------
# 4. Two uploads continue each other's rolling baseline
# ---------------------------------------------------------------------

def test_two_uploads_give_the_same_detection_as_one(admin, wengine, client, b_export, b_run):
    """Split mid-fault, so the second file's z-scores depend on the first file's readings."""
    cut = 705
    idx = b_export.groupby("AssetTag").cumcount()
    first, second = b_export[idx < cut], b_export[idx >= cut]
    submit(client, TS2, first)
    drain(wengine)
    submit(client, TS2, second)
    drain(wengine)
    one, two = anomalies(admin, TB), anomalies(admin, TS2)
    assert len(one) > 0
    pd.testing.assert_frame_equal(one, two, check_dtype=False)
    pd.testing.assert_frame_equal(summary(admin, TB), summary(admin, TS2), check_dtype=False)


# ---------------------------------------------------------------------
# 5. Isolation, limits, auth
# ---------------------------------------------------------------------

def test_tenant_a_cannot_reach_tenant_bs_uploads_jobs_rejects_or_objects(admin, client, b_run):
    jid = b_run["job"]["job_id"]
    uid = b_run["upload_id"]
    for method, path, kw in (("get", f"/uploads/{uid}", {}), ("get", f"/jobs/{jid}", {}),
                             ("get", f"/jobs/{jid}/rejected-rows", {}), ("post", f"/jobs/{jid}/retry", {}),
                             ("post", f"/uploads/{uid}/preview", {}),
                             ("post", f"/uploads/{uid}/mapping", {"json": {"mapping": B_MAPPING, "confirm": True}}),
                             ("post", f"/uploads/{uid}/content", {"content": b"a,b\n1,2\n", "headers": CSV})):
        headers = (_admin_hdr(admin, TA) if path.endswith("/retry")      # retry needs an admin (Phase 4)
                   else _hdr(TA, **kw.pop("headers", {})))
        r = getattr(client, method)(path, headers=headers, **kw)
        assert r.status_code == 404, (path, r.status_code)
    assert uid not in {u["upload_id"] for u in client.get("/uploads", headers=_hdr(TA)).json()}
    assert jid not in {j["job_id"] for j in client.get("/jobs", headers=_hdr(TA)).json()}
    for body in (client.get("/uploads", headers=_hdr(TB)).text, client.get(f"/jobs/{jid}", headers=_hdr(TB)).text):
        assert "tenants/" not in body and "storage_key" not in body
    key = table(admin, "SELECT storage_key FROM uploads WHERE upload_id = :u", u=uid)["storage_key"][0]
    assert key.startswith(f"tenants/{TB}/uploads/{uid}/")
    storage = get_storage()
    assert storage.exists(key, tenant_id=TB)
    with pytest.raises(StorageKeyError):
        storage.open(key, tenant_id=TA)
    with pytest.raises(StorageKeyError):
        storage.exists(f"tenants/{TB}/uploads/../../{TA}/x", tenant_id=TB)


@pytest.mark.parametrize("role_url", ["APP_DATABASE_URL", "WORKER_DATABASE_URL"])
def test_postgres_rls_hides_other_tenants_rows_from_both_roles(admin, b_run, role_url):
    engine = install(create_engine(os.environ[role_url]))
    try:
        for tenant, expect_rows in ((TA, False), (TB, True)):
            with engine.begin() as conn:
                bind_tenant(conn, tenant)
                for t in ("uploads", "jobs", "rejected_rows"):
                    seen = {r[0] for r in conn.execute(text(f"SELECT DISTINCT tenant_id FROM {t}"))}
                    assert seen <= {tenant}, (t, seen)
                    if t != "rejected_rows" and expect_rows:
                        assert seen == {TB}
                with pytest.raises(Exception, match="row-level security|permission denied"):
                    with conn.begin_nested():
                        conn.execute(text("INSERT INTO uploads (upload_id, tenant_id, filename, content_type, "
                                          "storage_key) VALUES (gen_random_uuid(), :o, 'x', 'text/csv', 'k')"),
                                     {"o": TB if tenant == TA else TA})
    finally:
        engine.dispose()


def test_the_api_role_cannot_write_readings_but_the_worker_role_can_only_as_its_tenant(admin):
    app_engine = install(create_engine(os.environ["APP_DATABASE_URL"]))
    w = install(create_engine(os.environ["WORKER_DATABASE_URL"]))
    try:
        with app_engine.connect() as conn:
            assert not conn.execute(text("SELECT has_table_privilege('iia_app', 'sensor_readings', 'INSERT')")).scalar()
        with w.begin() as conn:
            bind_tenant(conn, TA)
            with pytest.raises(Exception, match="row-level security"):
                with conn.begin_nested():
                    conn.execute(text("INSERT INTO machines (tenant_id, machine_id, production_line) "
                                      "VALUES (:t, 'EVIL', 'L')"), {"t": TB})
    finally:
        app_engine.dispose()
        w.dispose()


def test_upload_validation_limits_and_auth(client):
    # anonymous (demo posture) requests may not upload
    assert client.post("/uploads", json={"filename": "a.csv"}).status_code == 401
    assert client.get("/jobs").status_code == 401
    assert client.post("/uploads", json={"filename": "a.csv"}, headers={"X-API-Key": "iia_wrong"}).status_code == 401
    assert client.post("/uploads", json={"filename": "a.csv", "content_type": "image/png"},
                       headers=_hdr(TA)).status_code == 415
    uid = client.post("/uploads", json={"filename": "a.csv"}, headers=_hdr(TA)).json()["upload_id"]
    assert client.post(f"/uploads/{uid}/content", content=b"a\n1\n",
                       headers=_hdr(TA, **{"Content-Type": "application/pdf"})).status_code == 415
    assert client.post(f"/uploads/{uid}/content", content=b"", headers=_hdr(TA, **CSV)).status_code == 400
    assert client.post(f"/uploads/{uid}/content", content=b"a,b\n\x00\x01\n", headers=_hdr(TA, **CSV)).status_code == 415
    assert client.post(f"/uploads/{uid}/mapping", headers=_hdr(TA),
                       json={"mapping": B_MAPPING}).status_code == 409      # no content yet
    assert client.post("/uploads/not-a-uuid/preview", headers=_hdr(TA)).status_code == 404
    with env(UPLOAD_MAX_BYTES="100"):
        r = client.post(f"/uploads/{uid}/content", content=b"x" * 101, headers=_hdr(TA, **CSV))
        assert r.status_code == 413
        r = client.post(f"/uploads/{uid}/content", content=iter([b"x" * 60, b"y" * 60]), headers=_hdr(TA, **CSV))
        assert r.status_code == 413                                          # chunked, no Content-Length
    assert client.get(f"/uploads/{uid}", headers=_hdr(TA)).json()["status"] == "created"   # nothing stored
    with env(UPLOAD_RATE_LIMIT_PER_MINUTE="3"):
        limiter.reset()
        codes = [client.get("/uploads", headers=_hdr(TA)).status_code for _ in range(5)]
        assert codes == [200, 200, 200, 429, 429]
        limiter.reset()
