"""
Tenant isolation (production-readiness fixes 1-3): one company's data must
never be reachable with another company's credential.

Seeds a second tenant ("acme-test") that deliberately reuses the demo
tenant's machine id "M-01" with different data, then checks every layer:
the query functions, the static "every query is tenant-filtered" guard,
the HTTP API (including the adversarial case — tenant B's credential asking
for tenant A's machine id), and the auth modes.

Needs DATABASE_URL (skips cleanly without it), same as test_postgres.py.
"""

from __future__ import annotations

import inspect
import os
import re
from datetime import datetime

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from app.core.tenancy import DEFAULT_TENANT_ID as DEFAULT
from app.core.tenancy import generate_api_key, hash_api_key
from app.database import queries
from app.main import app
from tests.helpers import env

load_dotenv()

ACME = "acme-test"
OTHER = "globex-test"
ACME_KEY = generate_api_key()
OTHER_KEY = generate_api_key()


@pytest.fixture(scope="module")
def engine():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL not set")
    return create_engine(database_url)


def _cleanup(conn) -> None:
    for table in ("investigations", "investigation_audit_log", "ingestion_checkpoints", "tenant_schema_mappings",
                  "tenant_settings", "sensor_readings", "sensor_registry", "maintenance_records",
                  "machine_anomalies", "sensor_summary", "machines"):
        conn.execute(text(f"DELETE FROM {table} WHERE tenant_id IN (:a, :b)"),
                     {"a": ACME, "b": OTHER})
    conn.execute(text("DELETE FROM tenants WHERE tenant_id IN (:a, :b)"),
                 {"a": ACME, "b": OTHER})


@pytest.fixture(scope="module", autouse=True)
def two_extra_tenants(engine):
    """acme-test owns M-01 (same id as the demo fleet's, different data) and
    A-77; globex-test owns one machine and an INACTIVE credential check."""
    with engine.begin() as conn:
        _cleanup(conn)
        conn.execute(text("""
            INSERT INTO tenants (tenant_id, name, api_key_hash, active) VALUES
              (:a, 'Acme (test)', :ah, TRUE),
              (:b, 'Globex (test)', :bh, FALSE)
        """), {"a": ACME, "b": OTHER, "ah": hash_api_key(ACME_KEY), "bh": hash_api_key(OTHER_KEY)})
        conn.execute(text("""
            INSERT INTO machines (tenant_id, machine_id, production_line, type) VALUES
              (:a, 'M-01', 'ACME-1', 'H'), (:a, 'A-77', 'ACME-2', 'L'), (:b, 'G-01', 'GLX-1', 'M')
        """), {"a": ACME, "b": OTHER})
        conn.execute(text("""
            INSERT INTO sensor_summary
              (tenant_id, machine_id, window_start, window_end, reading_count,
               anomalous_reading_count, max_anomaly_score, health_status)
            VALUES (:a, 'M-01', '2026-03-01 00:00', '2026-03-01 01:00', 12, 0, 0, 'HEALTHY'),
                   (:a, 'A-77', '2026-03-01 00:00', '2026-03-01 01:00', 12, 9, 3, 'AT_RISK')
        """), {"a": ACME})
        conn.execute(text("""
            INSERT INTO machine_anomalies (tenant_id, machine_id, detected_at, anomaly_score, severity, triggered_reasons)
            VALUES (:a, 'A-77', '2026-03-01 00:30', 3, 'HIGH', 'acme-only reason')
        """), {"a": ACME})
        conn.execute(text("""
            INSERT INTO maintenance_records (tenant_id, machine_id, event_date, event_type, technician_notes, resolved)
            VALUES (:a, 'M-01', '2026-02-20', 'inspection', 'acme-only note', TRUE)
        """), {"a": ACME})
    yield
    with engine.begin() as conn:
        _cleanup(conn)


@pytest.fixture()
def conn(engine):
    with engine.connect() as connection:
        yield connection


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def _h(key: str) -> dict[str, str]:
    return {"X-API-Key": key}


# ---------------------------------------------------------------------
# Query layer
# ---------------------------------------------------------------------

def test_machine_lists_are_disjoint_per_tenant(conn):
    demo = {m["machine_id"]
            for m in queries.list_machines(conn, tenant_id=DEFAULT)}
    acme = {m["machine_id"]
            for m in queries.list_machines(conn, tenant_id=ACME)}
    assert len(demo) == 18 and acme == {"M-01", "A-77"}
    assert "A-77" not in demo and "M-04" not in acme


def test_same_machine_id_resolves_to_each_tenants_own_row(conn):
    assert queries.get_machine(
        conn, "M-01", tenant_id=DEFAULT)["production_line"] == "LINE-1"
    assert queries.get_machine(
        conn, "M-01", tenant_id=ACME)["production_line"] == "ACME-1"


def test_another_tenants_machine_id_is_simply_absent(conn):
    assert queries.get_machine(conn, "M-04", tenant_id=ACME) is None
    assert queries.get_machine(conn, "A-77", tenant_id=DEFAULT) is None
    assert queries.get_machine(
        conn, "A-77", tenant_id="no-such-tenant") is None


def test_evidence_queries_never_cross_tenants(conn):
    # acme's M-01 has one sensor window and no anomalies; the demo M-01 has hundreds.
    assert len(queries.get_sensor_summary(conn, "M-01", tenant_id=ACME)) == 1
    assert len(queries.get_sensor_summary(
        conn, "M-01", tenant_id=DEFAULT)) > 100
    assert queries.get_machine_anomalies(conn, "M-01", tenant_id=ACME) == []
    assert queries.get_machine_health(
        conn, "M-01", tenant_id=ACME)["health_status"] == "HEALTHY"
    assert queries.get_machine_anomalies(conn, "A-77", tenant_id=DEFAULT) == []
    assert [r["technician_notes"] for r in queries.get_maintenance_records(
        conn, "M-01", tenant_id=ACME)] == ["acme-only note"]
    assert all("acme-only" not in (r["technician_notes"] or "")
               for r in queries.get_maintenance_records(conn, "M-01", tenant_id=DEFAULT))


def test_fleet_wide_queries_only_see_their_own_tenant(conn):
    assert {r["machine_id"] for r in queries.get_fleet_status(
        conn, tenant_id=ACME)} == {"M-01", "A-77"}
    assert set(queries.get_fleet_sensor_trend(
        conn, tenant_id=ACME)) == {"M-01", "A-77"}
    assert queries.get_fleet_recent_high_anomalies(
        conn, tenant_id=ACME) == {"A-77"}
    demo_high = queries.get_fleet_recent_high_anomalies(
        conn, tenant_id=DEFAULT)
    assert "A-77" not in demo_high
    ranked = queries.rank_machines_for_inspection(
        conn, since=datetime(2026, 1, 1), tenant_id=ACME)
    assert [r["machine_id"] for r in ranked] == ["A-77"]


def test_query_functions_refuse_to_run_without_a_tenant(conn):
    with pytest.raises(TypeError):
        queries.list_machines(conn)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        queries.get_machine(conn, "M-01")  # type: ignore[call-arg]


def test_composite_foreign_key_blocks_cross_tenant_child_rows(engine):
    """A sensor row cannot reference a machine that belongs to another tenant:
    (globex-test, M-01) doesn't exist even though (default, M-01) does."""
    with pytest.raises(Exception) as exc_info:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO sensor_summary
                  (tenant_id, machine_id, window_start, window_end, reading_count, health_status)
                VALUES (:t, 'M-01', '2026-03-02', '2026-03-02 01:00', 1, 'HEALTHY')
            """), {"t": OTHER})
    assert "foreign key" in str(exc_info.value).lower()


# ---------------------------------------------------------------------
# Static guard: a new query can't forget the tenant filter
# ---------------------------------------------------------------------

TENANT_TABLES = ("uploads", "jobs", "rejected_rows", "machines", "sensor_summary", "machine_anomalies", "maintenance_records",
                 "sensor_registry", "sensor_readings", "tenant_settings", "tenant_schema_mappings",
                 "investigation_audit_log", "ingestion_checkpoints", "users", "audit_log",
                 "connectors", "connector_objects", "connector_runs", "investigations")
# Deliberately not tenant-scoped: public reference data / the credential lookup itself.
EXEMPT = {"get_ai4i_failure_mode_rates", "get_tenant_by_key_hash"}


def test_every_tenant_table_query_is_tenant_filtered():
    checked = 0
    for name, fn in inspect.getmembers(queries, inspect.isfunction):
        if fn.__module__ != queries.__name__ or name.startswith("_") or name in EXEMPT:
            continue
        params = inspect.signature(fn).parameters
        assert "tenant_id" in params, f"{name} has no tenant_id parameter"
        assert params["tenant_id"].kind is inspect.Parameter.KEYWORD_ONLY, name
        assert params["tenant_id"].default is inspect.Parameter.empty, (
            f"{name}: tenant_id must be required — a default would let a caller forget it")
        source = inspect.getsource(fn)
        table_refs = re.findall(
            r"\b(?:FROM|JOIN)\s+(?:%s)\b" % "|".join(TENANT_TABLES), source)
        predicates = re.findall(
            r"tenant_id\s*=\s*(?::tenant_id|\w+\.tenant_id)", source)
        assert len(predicates) >= len(table_refs), (
            f"{name}: {len(table_refs)} tenant-table reference(s) but only "
            f"{len(predicates)} tenant_id predicate(s)")
        checked += 1
    assert checked >= 15  # the guard must actually be inspecting the query layer


# ---------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------

def test_each_key_sees_only_its_own_fleet(client):
    acme = client.get("/machines", headers=_h(ACME_KEY))
    assert acme.status_code == 200
    assert {m["machine_id"] for m in acme.json()} == {"M-01", "A-77"}
    demo = client.get("/machines")  # demo posture, no key
    assert demo.status_code == 200 and len(demo.json()) == 18


def test_adversarial_other_tenants_machine_id_is_404_on_every_route(client):
    """Tenant acme's credential + the demo tenant's machine id (M-04, which
    really exists — for someone else) must look exactly like a missing machine."""
    h = _h(ACME_KEY)
    for path in ("/machines/M-04", "/machines/M-04/health", "/machines/M-04/anomalies",
                 "/machines/M-04/sensors"):
        assert client.get(path, headers=h).status_code == 404, path
    for path, key in (("/investigate", "question"), ("/chat", "message")):
        r = client.post(path, headers=h, json={
                        "machine_id": "M-04", key: "Why is M-04 failing?"})
        assert r.status_code == 404, path
        r = client.post(path, headers=h, json={
                        key: "Why is M-04 underperforming?"})
        assert r.status_code == 404, path  # named only in the text


def test_shared_machine_id_returns_the_callers_own_data(client):
    acme = client.get("/machines/M-01", headers=_h(ACME_KEY)).json()
    demo = client.get("/machines/M-01").json()
    assert acme["production_line"] == "ACME-1" and demo["production_line"] == "LINE-1"
    assert len(client.get("/machines/M-01/sensors",
               headers=_h(ACME_KEY)).json()) == 1


def test_investigation_evidence_and_fleet_scan_are_tenant_scoped(client):
    r = client.post("/investigate", headers=_h(ACME_KEY),
                    json={"machine_id": "A-77", "question": "Why is A-77 at risk?"})
    assert r.status_code == 200
    assert r.json()["risk_level"] == "HIGH"
    fleet = client.post("/investigate", headers=_h(ACME_KEY),
                        json={"question": "Which machines currently show abnormal behavior?"})
    assert fleet.status_code == 200
    ranking = fleet.json()["fleet_ranking"]
    assert {row["machine_id"] for row in ranking} <= {"M-01", "A-77"}


def test_documents_are_not_shared_across_tenants(client):
    """acme has ingested no documents, so it retrieves none — it does NOT
    fall back to the demo tenant's manuals."""
    r = client.post("/investigate", headers=_h(ACME_KEY),
                    json={"machine_id": "A-77", "question": "Why is A-77 at risk?"})
    assert r.status_code == 200
    assert r.json()["supporting_citations"] == []


# ---------------------------------------------------------------------
# Authentication modes
# ---------------------------------------------------------------------

def test_wrong_key_is_401_even_in_demo_posture(client):
    assert client.get(
        "/machines", headers=_h("iia_definitely-wrong")).status_code == 401


def test_inactive_tenant_key_is_rejected(client):
    assert client.get("/machines", headers=_h(OTHER_KEY)).status_code == 401


def test_auth_required_blocks_keyless_requests_on_every_data_route(client):
    with env(AUTH_REQUIRED="true", API_KEY=None):
        assert client.get("/machines").status_code == 401
        assert client.get("/machines/M-01/health").status_code == 401
        assert client.post(
            "/investigate", json={"question": "Which machines are abnormal?"}).status_code == 401
        assert client.get("/machines", headers=_h(ACME_KEY)).status_code == 200
        assert client.get("/").status_code == 200  # liveness stays open


def test_legacy_api_key_enforces_auth_and_only_reaches_the_demo_tenant(client):
    with env(API_KEY="legacy-bootstrap-key", AUTH_REQUIRED=None):
        # setting API_KEY turns enforcement on
        assert client.get("/machines").status_code == 401
        ok = client.get("/machines", headers=_h("legacy-bootstrap-key"))
        assert ok.status_code == 200 and len(
            ok.json()) == 18  # demo tenant, not acme
        assert client.get("/machines", headers=_h(ACME_KEY)
                          ).status_code == 200  # per-tenant keys still work


def test_401_body_does_not_reveal_whether_a_tenant_exists(client):
    a = client.get("/machines", headers=_h("iia_nope"))
    b = client.get("/machines", headers=_h(OTHER_KEY))
    assert a.json() == b.json()
