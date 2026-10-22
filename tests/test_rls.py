"""
Database-level tenant isolation (Phase 2): Postgres row-level security.

Everything below talks to Postgres AS THE LEAST-PRIVILEGE `iia_app` ROLE
(APP_DATABASE_URL) - or, for write checks, as a throwaway non-superuser role
via SET ROLE - because a superuser/owner connection bypasses RLS and would
prove nothing. DATABASE_URL (the owner) is used only for fixtures. Skips
without both URLs. Provision the role first: scripts/provision_app_role.py.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError

from app.core.tenancy import generate_api_key, hash_api_key
from app.database import queries
from app.database.tenant_context import bind_tenant, bypasses_rls, install, lookup_tenant_by_key_hash
from app.main import _check_rls_role, app

load_dotenv()

A, B, C = "rls-a", "rls-b", "rls-c"          # C is inactive
A_KEY, B_KEY, C_KEY = generate_api_key(), generate_api_key(), generate_api_key()
READABLE = ("machines", "sensor_summary", "machine_anomalies", "maintenance_records",
            "sensor_registry", "sensor_readings", "tenant_settings")
ALL_TENANT_TABLES = READABLE + ("tenant_schema_mappings", "investigation_audit_log",
                                "ingestion_checkpoints")
WRITER = "iia_rls_writer"


@pytest.fixture(scope="module")
def admin():
    url = os.environ.get("DATABASE_URL")
    if not url or not os.environ.get("APP_DATABASE_URL"):
        pytest.skip(
            "DATABASE_URL and APP_DATABASE_URL (iia_app) are both required")
    engine = create_engine(url)
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def app_engine(admin):
    # One pooled connection, so every request reuses the same DB session.
    engine = install(create_engine(
        os.environ["APP_DATABASE_URL"], pool_size=1, max_overflow=0))
    yield engine
    engine.dispose()


def _cleanup(conn) -> None:
    conn.execute(text("DROP TABLE IF EXISTS sensor_readings_y2031m01"))
    for table in ("investigations", "investigation_audit_log", "tenant_settings", "sensor_readings", "sensor_registry",
                  "maintenance_records", "machine_anomalies", "sensor_summary", "machines"):
        conn.execute(text(f"DELETE FROM {table} WHERE tenant_id IN (:a, :b, :c)"), {
                     "a": A, "b": B, "c": C})
    conn.execute(text("DELETE FROM tenants WHERE tenant_id IN (:a, :b, :c)"), {
                 "a": A, "b": B, "c": C})


@pytest.fixture(scope="module", autouse=True)
def data(admin):
    with admin.begin() as conn:
        _cleanup(conn)
        conn.execute(text("""INSERT INTO tenants (tenant_id, name, api_key_hash, active) VALUES
            (:a, 'A', :ah, TRUE), (:b, 'B', :bh, TRUE), (:c, 'C', :ch, FALSE)"""),
                     {"a": A, "b": B, "c": C, "ah": hash_api_key(A_KEY),
                      "bh": hash_api_key(B_KEY), "ch": hash_api_key(C_KEY)})
        # same machine id in both tenants
        for t, mid in ((A, "R-1"), (B, "R-1"), (B, "R-2")):
            conn.execute(text("INSERT INTO machines (tenant_id, machine_id, production_line, type) "
                              "VALUES (:t, :m, :t, 'L')"), {"t": t, "m": mid})
            conn.execute(text("""INSERT INTO sensor_summary (tenant_id, machine_id, window_start, window_end,
                reading_count, anomalous_reading_count, max_anomaly_score, health_status)
                VALUES (:t, :m, '2026-03-01 00:00', '2026-03-01 01:00', 12, 0, 0, 'HEALTHY')"""),
                         {"t": t, "m": mid})
            conn.execute(text("""INSERT INTO machine_anomalies (tenant_id, machine_id, detected_at,
                anomaly_score, severity, triggered_reasons) VALUES (:t, :m, '2026-03-01 00:30', 3, 'HIGH', 'x')"""),
                         {"t": t, "m": mid})
            conn.execute(text("""INSERT INTO maintenance_records (tenant_id, machine_id, event_date, event_type,
                resolved) VALUES (:t, :m, '2026-02-01', 'inspection', TRUE)"""), {"t": t, "m": mid})
        for t in (A, B):
            conn.execute(text(
                "INSERT INTO sensor_registry (tenant_id, sensor_name, unit) VALUES (:t, 'temp', 'C')"), {"t": t})
            conn.execute(text(
                "INSERT INTO sensor_readings VALUES (:t, 'R-1', 'temp', '2026-03-01 00:00', 1.0)"), {"t": t})
            conn.execute(text(
                "INSERT INTO tenant_settings (tenant_id, config) VALUES (:t, '{}')"), {"t": t})
    yield
    with admin.begin() as conn:
        _cleanup(conn)


@contextmanager
def session(engine, tenant_id=None):
    with engine.connect() as conn:
        if tenant_id:
            bind_tenant(conn, tenant_id)
        yield conn


def _count(conn, sql="SELECT COUNT(*) FROM machines") -> int:
    return conn.execute(text(sql)).scalar()


# --- the role itself --------------------------------------------------------

def test_app_role_is_least_privilege(admin, app_engine):
    assert not bypasses_rls(app_engine)
    with admin.connect() as c:
        attrs = c.execute(text("SELECT rolsuper, rolbypassrls, rolcreatedb, rolcreaterole "
                               "FROM pg_roles WHERE rolname = 'iia_app'")).one()
        assert tuple(attrs) == (False, False, False, False)
        assert c.execute(text("SELECT COUNT(*) FROM pg_class c JOIN pg_roles r ON r.oid = c.relowner "
                              # owns nothing
                              "WHERE r.rolname = 'iia_app'")).scalar() == 0
        assert c.execute(text(
            "SELECT has_table_privilege('iia_app', 'sensor_readings_default', 'SELECT')")).scalar() is False
    with session(app_engine, A) as conn:
        with pytest.raises(DBAPIError, match="permission denied"):
            conn.execute(text("SELECT * FROM tenants"))


def test_every_tenant_table_has_forced_rls_and_a_policy(admin):
    with admin.connect() as c:
        flags = dict(c.execute(text("SELECT relname, relforcerowsecurity AND relrowsecurity FROM pg_class "
                                    "WHERE relname = ANY(:t)"), {"t": list(ALL_TENANT_TABLES)}).all())
        assert flags == {t: True for t in ALL_TENANT_TABLES}
        have = {r[0] for r in c.execute(
            text("SELECT tablename FROM pg_policies WHERE policyname = 'tenant_isolation'"))}
        assert set(ALL_TENANT_TABLES) <= have


# --- reads -------------------------------------------------------------------

@pytest.mark.parametrize("table", READABLE)
def test_query_without_tenant_where_sees_only_the_current_tenant(app_engine, table):
    for tenant in (A, B):
        with session(app_engine, tenant) as conn:
            tenants = {r[0] for r in conn.execute(
                text(f"SELECT DISTINCT tenant_id FROM {table}"))}
            assert tenants == {tenant}, table


def test_tenant_b_sees_its_own_two_machines_and_a_sees_one(app_engine):
    with session(app_engine, A) as conn:
        assert _count(conn) == 1
    with session(app_engine, B) as conn:
        assert _count(conn) == 2


@pytest.mark.parametrize("table", READABLE)
def test_no_context_means_zero_rows(app_engine, table):
    with session(app_engine) as conn:
        assert _count(conn, f"SELECT COUNT(*) FROM {table}") == 0


def test_unknown_or_empty_context_means_zero_rows(app_engine):
    with session(app_engine, "no-such-tenant") as conn:
        assert _count(conn) == 0
    with session(app_engine) as conn:
        conn.execute(text("SELECT set_config('app.tenant_id', '', true)"))
        assert _count(conn) == 0


# --- writes --------------------------------------------------------------------

@pytest.fixture()
def writer(admin):
    """A throwaway non-superuser role with DML grants, to prove the policies
    reject writes (iia_app itself has almost no write privileges)."""
    with admin.begin() as c:
        c.execute(text(f"DROP ROLE IF EXISTS {WRITER}"))
        c.execute(text(f"CREATE ROLE {WRITER} NOLOGIN"))
        c.execute(text(
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON machines, sensor_summary TO {WRITER}"))
        c.execute(
            text(f"GRANT USAGE ON SEQUENCE sensor_summary_id_seq TO {WRITER}"))

    @contextmanager
    def as_writer(tenant_id):
        with admin.connect() as conn:
            trans = conn.begin()
            conn.execute(text(f"SET LOCAL ROLE {WRITER}"))
            if tenant_id:
                conn.execute(text("SELECT set_config('app.tenant_id', :t, true)"), {
                             "t": tenant_id})
            try:
                yield conn
            finally:
                trans.rollback()

    yield as_writer
    with admin.begin() as c:
        c.execute(text(f"DROP OWNED BY {WRITER}"))
        c.execute(text(f"DROP ROLE {WRITER}"))


_INS = "INSERT INTO machines (tenant_id, machine_id, production_line, type) VALUES (:t, 'W-1', 'L', 'L')"


def test_cross_tenant_insert_is_rejected_and_own_insert_works(writer):
    with writer(A) as conn:
        # own tenant: fine
        conn.execute(text(_INS), {"t": A})
    with writer(A) as conn:
        with pytest.raises(DBAPIError, match="row-level security"):
            conn.execute(text(_INS), {"t": B})
    with writer(None) as conn:                                    # no context: fail closed
        with pytest.raises(DBAPIError, match="row-level security"):
            conn.execute(text(_INS), {"t": A})


def test_cross_tenant_update_and_delete_touch_nothing(writer):
    with writer(A) as conn:
        assert conn.execute(text("UPDATE machines SET production_line = 'HACK' WHERE tenant_id = :t"),
                            {"t": B}).rowcount == 0
        assert conn.execute(text("DELETE FROM machines WHERE tenant_id = :t"), {
                            "t": B}).rowcount == 0
        assert conn.execute(text("UPDATE machines SET production_line = 'ok' WHERE tenant_id = :t"),
                            {"t": A}).rowcount == 1


def test_update_cannot_move_a_row_to_another_tenant(writer):
    with writer(A) as conn:
        with pytest.raises(DBAPIError, match="row-level security"):
            conn.execute(text("UPDATE machines SET tenant_id = :b WHERE tenant_id = :a"), {
                         "a": A, "b": B})


def test_app_role_audit_insert_is_tenant_checked_and_append_only(app_engine):
    row = dict(request_id="r", endpoint="chat", machine_id=None, intent=None,
               risk_level=None, authenticated=True, question="q")
    with session(app_engine, A) as conn:
        queries.insert_investigation_audit(
            conn, tenant_id=A, **row)          # allowed
        conn.rollback()
        with pytest.raises(DBAPIError, match="row-level security"):
            queries.insert_investigation_audit(
                conn, tenant_id=B, **row)      # wrong tenant
    with session(app_engine, A) as conn:
        for sql in ("UPDATE investigation_audit_log SET question = 'x'", "DELETE FROM investigation_audit_log"):
            with pytest.raises(DBAPIError, match="permission denied"):
                conn.execute(text(sql))
            conn.rollback()


# --- pooling and commits -------------------------------------------------------

def test_pooled_connection_reuse_never_leaks_a_tenant(app_engine):
    with session(app_engine, A) as conn:
        pid = conn.execute(text("SELECT pg_backend_pid()")).scalar()
        assert _count(conn) == 1
    # same DB session, no context
    with session(app_engine) as conn:
        assert conn.execute(text("SELECT pg_backend_pid()")).scalar() == pid
        assert _count(conn) == 0
        assert conn.execute(text(
            "SELECT NULLIF(current_setting('app.tenant_id', true), '')")).scalar() is None
    # and the next tenant gets its own
    with session(app_engine, B) as conn:
        assert conn.execute(text("SELECT pg_backend_pid()")).scalar() == pid
        assert _count(conn) == 2


def test_context_survives_a_mid_request_commit(app_engine):
    """routes._audit() commits in the middle of a request; later queries on the
    same connection must still run as the tenant, not silently see zero rows."""
    with session(app_engine, A) as conn:
        assert _count(conn) == 1
        queries.insert_investigation_audit(
            conn, tenant_id=A, request_id="c", endpoint="chat", machine_id=None, intent=None,
            risk_level=None, authenticated=True, question="commit test")
        conn.commit()
        assert _count(conn) == 1
        conn.commit()
        assert conn.execute(
            text("SELECT current_setting('app.tenant_id', true)")).scalar() == A


def test_binding_after_the_auth_lookup_applies_inside_the_open_transaction(app_engine):
    with app_engine.connect() as conn:
        assert lookup_tenant_by_key_hash(conn, hash_api_key(A_KEY))[
            "tenant_id"] == A   # opens a transaction
        assert _count(conn) == 0
        bind_tenant(conn, A)
        assert _count(conn) == 1


# --- authentication lookup -------------------------------------------------------

def test_key_lookup_goes_through_the_definer_function(app_engine):
    with app_engine.connect() as conn:
        assert lookup_tenant_by_key_hash(conn, hash_api_key(A_KEY)) == {
            "tenant_id": A, "name": "A"}
        assert lookup_tenant_by_key_hash(
            conn, hash_api_key(C_KEY)) is None      # inactive
        assert lookup_tenant_by_key_hash(
            conn, hash_api_key("iia_wrong")) is None


# --- partitions ----------------------------------------------------------------

def test_partition_created_after_the_migration_is_isolated(admin, app_engine):
    from scripts.manage_partitions import create_partition
    create_partition(admin, "sensor_readings", 2031, 1)
    with admin.begin() as c:
        for t in (A, B):
            c.execute(text(
                "INSERT INTO sensor_readings VALUES (:t, 'R-1', 'temp', '2031-01-05 00:00', 9.0)"), {"t": t})
        flags = c.execute(text("SELECT relrowsecurity, relforcerowsecurity FROM pg_class "
                               "WHERE relname = 'sensor_readings_y2031m01'")).one()
        assert tuple(flags) == (True, True)
        assert c.execute(text(
            "SELECT COUNT(*) FROM pg_policies WHERE tablename = 'sensor_readings_y2031m01'")).scalar() == 1
    q = "SELECT tenant_id FROM sensor_readings WHERE ts >= '2031-01-01'"
    for tenant in (A, B):
        with session(app_engine, tenant) as conn:
            assert [r[0] for r in conn.execute(text(q))] == [tenant]
    # no grant on the partition itself
    with session(app_engine, A) as conn:
        with pytest.raises(DBAPIError, match="permission denied"):
            conn.execute(text("SELECT * FROM sensor_readings_y2031m01"))
    # even if someone grants it, the partition's own policy holds
    with admin.begin() as c:
        c.execute(text("GRANT SELECT ON sensor_readings_y2031m01 TO iia_app"))
    try:
        with session(app_engine, A) as conn:
            assert {r[0] for r in conn.execute(
                text("SELECT tenant_id FROM sensor_readings_y2031m01"))} == {A}
    finally:
        with admin.begin() as c:
            c.execute(text("REVOKE ALL ON sensor_readings_y2031m01 FROM iia_app"))


# --- the API end to end, as iia_app ----------------------------------------------

def test_http_requests_run_as_the_app_role_and_stay_isolated(admin):
    with TestClient(app) as client:
        assert not bypasses_rls(client.app.state.db_engine)
        ha, hb = {"X-API-Key": A_KEY}, {"X-API-Key": B_KEY}
        assert [m["machine_id"]
                for m in client.get("/machines", headers=ha).json()] == ["R-1"]
        assert [m["machine_id"] for m in client.get(
            "/machines", headers=hb).json()] == ["R-1", "R-2"]
        assert client.get("/machines/R-2", headers=ha).status_code == 404
        assert client.get("/machines/R-2/health",
                          headers=hb).status_code == 200
        r = client.post("/investigate", headers=ha,
                        json={"machine_id": "R-1", "question": "Why is R-1 at risk?"})
        assert r.status_code == 200
        # inactive tenant
        assert client.get(
            "/machines", headers={"X-API-Key": C_KEY}).status_code == 401
        # demo posture
        assert len(client.get("/machines").json()) == 18
    # audit row written by the app role, correct tenant
    with admin.connect() as c:
        rows = c.execute(text("SELECT tenant_id FROM investigation_audit_log WHERE tenant_id IN (:a, :b) "
                              "AND question = 'Why is R-1 at risk?'"),
                         {"a": A, "b": B}).all()
    assert [r[0] for r in rows] == [A]


# --- startup guard ----------------------------------------------------------------

def test_startup_guard_flags_a_role_that_bypasses_rls(admin, app_engine, caplog):
    assert bypasses_rls(admin)
    with pytest.raises(RuntimeError, match="bypasses row-level security"):
        _check_rls_role(admin, SimpleNamespace(rls_required=True))
    with caplog.at_level("WARNING"):
        _check_rls_role(admin, SimpleNamespace(rls_required=False))
    assert "bypasses row-level security" in caplog.text
    _check_rls_role(app_engine, SimpleNamespace(
        rls_required=True))       # least privilege: fine
