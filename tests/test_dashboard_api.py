"""Phase 6: dashboard endpoints (fleet summary, series, stored investigations, units,
registry editor). Real Postgres; under RLS when APP_DATABASE_URL is the iia_app role."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from app.api.rate_limit import limiter
from app.core.security import create_token, hash_password
from app.core.tenancy import generate_api_key, hash_api_key
from app.main import app
from tests.helpers import env

load_dotenv()

TA, TB = "dash-a-test", "dash-b-test"
KEY_A = generate_api_key()
SECRET = "phase6-test-secret-" + "0123456789abcdef" * 2
END = datetime(2026, 3, 10, 12, 0)          # newest reading of D-1 (a fixed date, not "now")


@pytest.fixture(scope="module")
def engine():
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    return create_engine(url)


def _cleanup(conn) -> None:
    for table in ("investigations", "audit_log", "sensor_readings", "machine_anomalies",
                  "sensor_summary", "sensor_registry", "users", "machines"):
        conn.execute(text(f"DELETE FROM {table} WHERE tenant_id IN (:a, :b)"), {"a": TA, "b": TB})
    conn.execute(text("DELETE FROM tenants WHERE tenant_id IN (:a, :b)"), {"a": TA, "b": TB})


@pytest.fixture(scope="module", autouse=True)
def seeded(engine):
    with engine.begin() as c:
        _cleanup(c)
        c.execute(text("INSERT INTO tenants (tenant_id, name, api_key_hash) VALUES "
                       "(:a, 'Dash A', :h), (:b, 'Dash B', :h2)"),
                  {"a": TA, "b": TB, "h": hash_api_key(KEY_A), "h2": hash_api_key(generate_api_key())})
        c.execute(text("INSERT INTO machines (tenant_id, machine_id, production_line, type) VALUES "
                       "(:a, 'D-1', 'L1', NULL), (:a, 'D-2', 'L1', NULL), (:a, 'D-3', 'L2', NULL), "
                       "(:b, 'D-1', 'LB', NULL)"), {"a": TA, "b": TB})
        c.execute(text("INSERT INTO sensor_registry (tenant_id, sensor_name, unit, normal_min, normal_max) "
                       "VALUES (:a, 'temp', 'C', 10, 90), (:a, 'press', 'bar', NULL, NULL), "
                       "(:b, 'temp', 'C', 0, 50)"), {"a": TA, "b": TB})
        # D-1: sustained 50% anomalous windows => HIGH; D-2: clean => LOW; D-3 has no windows => UNKNOWN
        for mid, anomalous, score, status in (("D-1", 6, 3, "AT_RISK"), ("D-2", 0, 0, "HEALTHY")):
            c.execute(text("""
                INSERT INTO sensor_summary (tenant_id, machine_id, window_start, window_end,
                    reading_count, anomalous_reading_count, max_anomaly_score, health_status)
                SELECT :t, :m, g, g + interval '1 hour', 12, :an, :sc, :st
                FROM generate_series(timestamp '2026-03-09 00:00', timestamp '2026-03-10 11:00',
                                     interval '1 hour') g
            """), {"t": TA, "m": mid, "an": anomalous, "sc": score, "st": status})
        c.execute(text("""
            INSERT INTO machine_anomalies (tenant_id, machine_id, detected_at, anomaly_score, severity,
                                           triggered_reasons)
            VALUES (:t, 'D-1', '2026-03-10 10:00', 3, 'HIGH', 'temp high')
        """), {"t": TA})
        # D-1 temp readings every 5 min for 10 days ending at END; one B reading on the same ids
        c.execute(text("""
            INSERT INTO sensor_readings (tenant_id, machine_id, sensor_name, ts, value)
            SELECT :t, 'D-1', 'temp', g, 50 + (extract(epoch FROM g)::bigint % 7)
            FROM generate_series(CAST(:e AS timestamp) - interval '10 days', CAST(:e AS timestamp),
                                 interval '5 minutes') g
        """), {"t": TA, "e": END})
        c.execute(text("INSERT INTO sensor_readings (tenant_id, machine_id, sensor_name, ts, value) "
                       "VALUES (:b, 'D-1', 'temp', :e, 999)"), {"b": TB, "e": END})
    yield
    with engine.begin() as c:
        _cleanup(c)


@pytest.fixture(scope="module", autouse=True)
def jwt_env():
    with env(JWT_SECRET=SECRET, JWT_TTL_MINUTES="60", API_KEY=None, AUTH_REQUIRED=None):
        yield


@pytest.fixture(autouse=True)
def _fresh_limiter():
    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def make_user(engine):
    def _make(role="viewer", tenant=TA):
        uid, email = str(uuid.uuid4()), f"{uuid.uuid4().hex[:8]}@example.com"
        with engine.begin() as c:
            c.execute(text("INSERT INTO users (id, tenant_id, email, password_hash, role) "
                           "VALUES (:i, :t, :e, :h, :r)"),
                      {"i": uid, "t": tenant, "e": email, "h": hash_password("pw-" + uid), "r": role})
        tok = create_token(user_id=uid, tenant_id=tenant, secret=SECRET, ttl_minutes=60)
        return uid, email, {"Authorization": f"Bearer {tok}"}
    return _make


# ---------------------------------------------------------------------
# Fleet summary
# ---------------------------------------------------------------------

def test_fleet_summary_risk_total_and_pagination(client, make_user):
    _, _, h = make_user("viewer")
    r = client.get("/fleet/summary", headers=h)
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 3 and [i["machine_id"] for i in body["items"]] == ["D-1", "D-2", "D-3"]
    risk = {i["machine_id"]: i["risk_level"] for i in body["items"]}
    assert risk == {"D-1": "HIGH", "D-2": "LOW", "D-3": "UNKNOWN"}
    d3 = body["items"][2]
    assert d3["window_start"] is None and d3["health_status"] is None
    page = client.get("/fleet/summary?limit=2&offset=2", headers=h).json()
    assert page["total"] == 3 and [i["machine_id"] for i in page["items"]] == ["D-3"]
    assert client.get("/fleet/summary?limit=201", headers=h).status_code == 422
    assert client.get("/fleet/summary?offset=-1", headers=h).status_code == 422


def test_fleet_summary_is_tenant_scoped(client, make_user):
    _, _, hb = make_user("viewer", TB)
    body = client.get("/fleet/summary", headers=hb).json()
    assert body["total"] == 1 and body["items"][0]["production_line"] == "LB"
    assert body["items"][0]["risk_level"] == "UNKNOWN"


# ---------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------

def test_series_defaults_to_seven_days_ending_at_the_newest_reading(client, make_user):
    _, _, h = make_user("viewer")
    r = client.get("/machines/D-1/series?sensor=temp", headers=h)
    assert r.status_code == 200
    s = r.json()
    assert s["unit"] == "C" and s["normal_min"] == 10 and s["normal_max"] == 90
    assert datetime.fromisoformat(s["until"]) == END
    assert datetime.fromisoformat(s["until"]) - datetime.fromisoformat(s["since"]) == timedelta(days=7)
    assert 0 < len(s["points"]) <= 300
    assert all(50 <= p["min"] <= p["avg"] <= p["max"] <= 56 for p in s["points"])   # never tenant B's 999


def test_series_respects_max_points_and_caps_the_range(client, make_user):
    _, _, h = make_user("viewer")
    s = client.get("/machines/D-1/series?sensor=temp&max_points=10", headers=h).json()
    assert len(s["points"]) <= 10 and s["bucket_seconds"] >= 60
    too_wide = client.get("/machines/D-1/series?sensor=temp&since=2026-01-01T00:00:00&until=2026-03-10T00:00:00",
                          headers=h)
    assert too_wide.status_code == 422
    backwards = client.get("/machines/D-1/series?sensor=temp&since=2026-03-10T00:00:00&until=2026-03-09T00:00:00",
                           headers=h)
    assert backwards.status_code == 422
    assert client.get("/machines/D-1/series?sensor=temp&max_points=5", headers=h).status_code == 422
    assert client.get("/machines/D-1/series?sensor=temp&max_points=1001", headers=h).status_code == 422


def test_series_unknowns_and_empty(client, make_user):
    _, _, h = make_user("viewer")
    assert client.get("/machines/NOPE/series?sensor=temp", headers=h).status_code == 404
    assert client.get("/machines/D-1/series?sensor=nope", headers=h).status_code == 404
    assert client.get("/machines/D-1/series", headers=h).status_code == 422
    empty = client.get("/machines/D-2/series?sensor=temp", headers=h)
    assert empty.status_code == 200 and empty.json()["points"] == []
    # a sensor declared for the tenant but never read for this machine is also empty, not an error
    assert client.get("/machines/D-1/series?sensor=press", headers=h).json()["points"] == []


def test_series_cannot_read_another_tenants_machine(client, make_user):
    _, _, hb = make_user("viewer", TB)
    # D-1 exists in both tenants; tenant B sees only its own single 999 reading
    s = client.get("/machines/D-1/series?sensor=temp", headers=hb).json()
    assert [p["max"] for p in s["points"]] == [999.0] and s["normal_max"] == 50
    assert client.get("/machines/D-2/series?sensor=temp", headers=hb).status_code == 404


# ---------------------------------------------------------------------
# Stored investigations
# ---------------------------------------------------------------------

def test_investigation_is_stored_listed_and_isolated(client, make_user, engine):
    uid, email, admin = make_user("admin")
    _, _, viewer = make_user("viewer")
    _, _, other = make_user("viewer", TB)
    r = client.post("/investigate", headers=admin,
                    json={"question": "Why is D-1 abnormal?", "machine_id": "D-1"})
    assert r.status_code == 200, r.text
    c = client.post("/chat", headers=viewer, json={"message": "Which machines look abnormal?"})
    assert c.status_code == 200, c.text

    rows = client.get("/investigations", headers=viewer).json()
    assert [x["endpoint"] for x in rows] == ["chat", "investigate"]            # newest first
    inv = next(x for x in rows if x["endpoint"] == "investigate")
    assert inv["created_by_label"] == email and inv["machine_id"] == "D-1"
    assert inv["risk_level"] == r.json()["risk_level"] == "HIGH"
    assert "result" not in inv                                                 # summary rows only
    detail = client.get(f"/investigations/{inv['id']}", headers=viewer).json()
    assert detail["result"]["narrative"] == r.json()["narrative"]
    assert detail["result"]["recommendation"] == r.json()["recommendation"]
    chat_row = next(x for x in rows if x["endpoint"] == "chat")
    chat_detail = client.get(f"/investigations/{chat_row['id']}", headers=viewer).json()
    assert chat_detail["result"]["recommendation"] == c.json()["reply"]
    with engine.connect() as conn:
        by = conn.execute(text("SELECT created_by::text FROM investigations WHERE id = :i"),
                          {"i": inv["id"]}).scalar()
    assert by == uid

    assert [x["id"] for x in client.get("/investigations?machine_id=D-1", headers=viewer).json()] == [inv["id"]]
    assert len(client.get("/investigations?limit=1", headers=viewer).json()) == 1
    assert len(client.get("/investigations?limit=1&offset=1", headers=viewer).json()) == 1
    assert client.get("/investigations?limit=101", headers=viewer).status_code == 422

    # tenant B sees none of it, and cannot fetch A's row by id
    assert client.get("/investigations", headers=other).json() == []
    assert client.get(f"/investigations/{inv['id']}", headers=other).status_code == 404
    assert client.get("/investigations/not-a-uuid", headers=viewer).status_code == 404
    assert client.get(f"/investigations/{uuid.uuid4()}", headers=viewer).status_code == 404


def test_an_oversized_result_is_truncated_not_dropped(client, make_user, monkeypatch):
    from app.api import routes
    from app.agents import graph
    real = graph.run_investigation

    def fat(*a, **kw):
        out = dict(real(*a, **kw))
        out["narrative"] = "x" * 200_000
        return out
    monkeypatch.setattr(routes, "run_investigation", fat)
    _, _, h = make_user("viewer")
    before = len(client.get("/investigations?limit=100", headers=h).json())
    assert client.post("/investigate", headers=h, json={"question": "Which machines look abnormal?"}).status_code == 200
    rows = client.get("/investigations?limit=100", headers=h).json()
    assert len(rows) == before + 1
    detail = client.get(f"/investigations/{rows[0]['id']}", headers=h).json()
    assert detail["result"]["truncated"] is True and len(detail["result"]["narrative"]) < 25_000


# ---------------------------------------------------------------------
# Units and registry editor
# ---------------------------------------------------------------------

def test_units_catalog(client, make_user):
    _, _, h = make_user("viewer")
    units = client.get("/units", headers=h).json()
    assert {"unit": "kPa", "dimension": "pressure"} in units and {"unit": "F", "dimension": "temperature"} in units


def test_admin_edits_a_sensor_and_it_is_audited(client, make_user, engine):
    _, _, admin = make_user("admin")
    r = client.post("/sensors/temp", headers=admin,
                    json={"normal_min": 5, "normal_max": 80, "z_score_threshold": 2.5, "enabled": False})
    assert r.status_code == 200
    assert r.json() == {"sensor_name": "temp", "unit": "C", "normal_min": 5.0, "normal_max": 80.0,
                        "z_score_threshold": 2.5, "enabled": False}
    assert {s["sensor_name"]: s for s in client.get("/sensors", headers=admin).json()}["temp"]["enabled"] is False
    cleared = client.post("/sensors/temp", headers=admin, json={"normal_min": None, "enabled": True}).json()
    assert cleared["normal_min"] is None and cleared["normal_max"] == 80.0 and cleared["enabled"] is True
    with engine.connect() as c:
        n = c.execute(text("SELECT COUNT(*) FROM audit_log WHERE tenant_id = :t AND "
                           "action = 'POST /sensors/{sensor_name}' AND outcome = 'success'"), {"t": TA}).scalar()
    assert n >= 2


def test_sensor_edit_validation(client, make_user):
    _, _, admin = make_user("admin")
    assert client.post("/sensors/temp", headers=admin, json={}).status_code == 422
    assert client.post("/sensors/temp", headers=admin, json={"unit": "K"}).status_code == 422      # extra field
    assert client.post("/sensors/temp", headers=admin, json={"normal_min": 100}).status_code == 422  # >= stored max
    assert client.post("/sensors/temp", headers=admin, json={"z_score_threshold": 0}).status_code == 422
    assert client.post("/sensors/temp", headers=admin, json={"enabled": None}).status_code == 422
    assert client.post("/sensors/nope", headers=admin, json={"enabled": True}).status_code == 404


def test_viewer_and_api_key_cannot_edit_and_tenants_are_separate(client, make_user):
    _, _, viewer = make_user("viewer")
    assert client.post("/sensors/temp", headers=viewer, json={"enabled": False}).status_code == 403
    assert client.post("/sensors/temp", headers={"X-API-Key": KEY_A}, json={"enabled": False}).status_code == 403
    _, _, admin_b = make_user("admin", TB)
    # 'press' exists only for tenant A; 'temp' for B is B's own row
    assert client.post("/sensors/press", headers=admin_b, json={"enabled": False}).status_code == 404
    b = client.post("/sensors/temp", headers=admin_b, json={"normal_max": 60}).json()
    assert b["normal_max"] == 60.0
    a = {s["sensor_name"]: s for s in client.get("/sensors", headers=viewer).json()}["temp"]
    assert a["normal_max"] == 80.0       # A's row untouched by B's edit


# ---------------------------------------------------------------------
# Row-level security on the new table, and what the API role may write
# ---------------------------------------------------------------------

def test_investigations_rls_is_forced_and_fails_closed(engine, client, make_user):
    _, _, admin = make_user("admin")
    assert client.post("/investigate", headers=admin,
                       json={"question": "Which machines look abnormal?"}).status_code == 200
    with engine.connect() as c:
        flags = c.execute(text("SELECT relrowsecurity, relforcerowsecurity FROM pg_class "
                               "WHERE relname = 'investigations'")).one()
        assert tuple(flags) == (True, True)
        assert c.execute(text("SELECT COUNT(*) FROM pg_policies WHERE tablename = 'investigations'")).scalar() == 1
    app_url = os.environ.get("APP_DATABASE_URL")
    if not app_url:
        pytest.skip("APP_DATABASE_URL not set: RLS behaviour needs the iia_app role")
    app_engine = create_engine(app_url)
    try:
        with app_engine.begin() as c:                       # no tenant bound: zero rows, writes refused
            assert c.execute(text("SELECT COUNT(*) FROM investigations")).scalar() == 0
        with app_engine.begin() as c:
            c.execute(text("SELECT set_config('app.tenant_id', :t, true)"), {"t": TB})
            assert c.execute(text("SELECT COUNT(*) FROM investigations WHERE tenant_id = :a"),
                             {"a": TA}).scalar() == 0
        with pytest.raises(Exception):                      # iia_app can neither update nor delete history
            with app_engine.begin() as c:
                c.execute(text("SELECT set_config('app.tenant_id', :t, true)"), {"t": TA})
                c.execute(text("DELETE FROM investigations"))
        with pytest.raises(Exception):                      # nor edit the unit or the name of a sensor
            with app_engine.begin() as c:
                c.execute(text("SELECT set_config('app.tenant_id', :t, true)"), {"t": TA})
                c.execute(text("UPDATE sensor_registry SET unit = 'K'"))
    finally:
        app_engine.dispose()
