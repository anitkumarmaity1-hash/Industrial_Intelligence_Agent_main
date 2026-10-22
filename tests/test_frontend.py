"""Phase 6: the Streamlit dashboard, driven with Streamlit's AppTest.

The dashboard's HTTP calls are routed into the real FastAPI app (TestClient) over a real
Postgres, so these tests exercise frontend -> API -> database, not a mock of the API.
Skips without DATABASE_URL.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

import pytest
import requests
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from streamlit.testing.v1 import AppTest

from app.api.rate_limit import limiter
from app.core.security import hash_password
from app.core.tenancy import generate_api_key, hash_api_key
from app.database import queries
from app.main import app
from tests.helpers import env

load_dotenv()

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
ENTRY = str(FRONTEND / "streamlit_app.py")
T, OTHER = "fe-test", "fe-other-test"
PW = "frontend test password 1"
SECRET = "phase6-frontend-secret-" + "0123456789abcdef" * 2
WIDE_CSV = (b"Asset,At,Line,Temp,Press\n"
            + b"".join(f"W-1,2026-04-01T00:{m:02d}:00,L1,{20 + m % 5},{150 + m}\n".encode() for m in range(0, 30)))


@pytest.fixture(scope="module")
def engine():
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    return create_engine(url)


def _cleanup(conn) -> None:
    for table in ("investigations", "audit_log", "rejected_rows", "jobs", "uploads", "sensor_readings",
                  "machine_anomalies", "sensor_summary", "sensor_registry", "users", "machines"):
        conn.execute(text(f"DELETE FROM {table} WHERE tenant_id IN (:a, :b)"), {"a": T, "b": OTHER})
    conn.execute(text("DELETE FROM tenants WHERE tenant_id IN (:a, :b)"), {"a": T, "b": OTHER})


@pytest.fixture(scope="module", autouse=True)
def seeded(engine):
    with engine.begin() as c:
        _cleanup(c)
        c.execute(text("INSERT INTO tenants (tenant_id, name, api_key_hash) VALUES (:a, 'FE', :h), (:b, 'FE other', :h2)"),
                  {"a": T, "b": OTHER, "h": hash_api_key(generate_api_key()), "h2": hash_api_key(generate_api_key())})
        c.execute(text("INSERT INTO machines (tenant_id, machine_id, production_line) VALUES "
                       "(:a, 'FE-1', 'Line 1'), (:b, 'OTHER-SECRET-1', 'Hidden')"), {"a": T, "b": OTHER})
        c.execute(text("INSERT INTO sensor_registry (tenant_id, sensor_name, unit, normal_min, normal_max) "
                       "VALUES (:a, 'temp', 'C', 10, 90)"), {"a": T})
        c.execute(text("""
            INSERT INTO sensor_summary (tenant_id, machine_id, window_start, window_end, reading_count,
                anomalous_reading_count, max_anomaly_score, health_status)
            SELECT :t, 'FE-1', g, g + interval '1 hour', 12, 6, 3, 'AT_RISK'
            FROM generate_series(timestamp '2026-03-09 00:00', timestamp '2026-03-10 11:00', interval '1 hour') g
        """), {"t": T})
        c.execute(text("""
            INSERT INTO sensor_readings (tenant_id, machine_id, sensor_name, ts, value)
            SELECT :t, 'FE-1', 'temp', g, 40 + (extract(epoch FROM g)::bigint % 9)
            FROM generate_series(timestamp '2026-03-08 00:00', timestamp '2026-03-10 11:00', interval '10 minutes') g
        """), {"t": T})
        for role in ("admin", "viewer"):
            c.execute(text("INSERT INTO users (id, tenant_id, email, password_hash, role) VALUES (:i, :t, :e, :h, :r)"),
                      {"i": str(uuid.uuid4()), "t": T, "e": f"{role}@fe.example", "h": hash_password(PW), "r": role})
    yield
    with engine.begin() as c:
        _cleanup(c)


@pytest.fixture(scope="module", autouse=True)
def _env(tmp_path_factory):
    with env(JWT_SECRET=SECRET, JWT_TTL_MINUTES="60", API_KEY=None, AUTH_REQUIRED=None,
             STORAGE_LOCAL_DIR=str(tmp_path_factory.mktemp("storage")), STORAGE_BACKEND="local"):
        yield


@pytest.fixture(autouse=True)
def bridge(monkeypatch):
    """Route the dashboard's `requests.request` calls into the FastAPI app."""
    import sys
    if str(FRONTEND) not in sys.path:
        sys.path.insert(0, str(FRONTEND))
    import api_client
    limiter.reset()
    with TestClient(app) as client:
        def fake(method, url, headers=None, timeout=None, params=None, json=None, data=None, **_):
            return client.request(method, re.sub(r"^https?://[^/]+", "", url), headers=headers,
                                  params=params, json=json, content=data)
        monkeypatch.setattr(requests, "request", fake)
        api_client._cached_get.clear()
        yield client
    api_client._cached_get.clear()
    limiter.reset()


def _login(at: AppTest, role: str, password: str = PW) -> AppTest:
    at.text_input[0].set_value(T)
    at.text_input[1].set_value(f"{role}@fe.example")
    at.text_input[2].set_value(password)
    next(b for b in at.button if b.label == "Sign in").click()
    return at.run()


def _signed_in_token(bridge, role: str) -> str:
    r = bridge.post("/auth/login", json={"tenant": T, "email": f"{role}@fe.example", "password": PW})
    assert r.status_code == 200
    return r.json()["access_token"]


def _view(bridge, module: str, role: str, after: str = "") -> AppTest:
    """Render ONE view as a signed-in `role` (so a test can drive a page the nav would show)."""
    src = (f"import sys\nsys.path.insert(0, {str(FRONTEND)!r})\nimport api_client\nfrom views import {module}\n"
           f"{module}.render()\n{after}")
    at = AppTest.from_string(src, default_timeout=60)
    at.session_state["token"] = _signed_in_token(bridge, role)
    at.session_state["role"] = role
    at.session_state["email"] = f"{role}@fe.example"
    at.session_state["tenant"] = T
    return at.run()


# ---------------------------------------------------------------------
# Login, logout, expiry
# ---------------------------------------------------------------------

def test_login_flow_and_logout():
    at = AppTest.from_file(ENTRY, default_timeout=60).run()
    assert not at.exception and at.title[0].value.endswith("Industrial Intelligence Agent")
    assert [b.label for b in at.button] == ["Sign in"]
    at = _login(at, "admin", "wrong password")
    assert not at.exception and at.error[0].value == "Invalid credentials."
    assert not at.session_state["token"] if "token" in at.session_state else True
    at = _login(at, "admin")
    assert not at.exception
    assert at.title[0].value == "Fleet health"
    assert "admin@fe.example" in at.sidebar.markdown[0].value and T in at.sidebar.markdown[0].value
    next(b for b in at.sidebar.button if b.label == "Sign out").click()
    at.run()
    assert at.title[0].value.endswith("Industrial Intelligence Agent") and "token" not in at.session_state


def test_an_expired_or_invalid_token_returns_to_sign_in_with_a_notice():
    at = AppTest.from_file(ENTRY, default_timeout=60)
    at.session_state["token"] = "not-a-real-token"
    at.session_state["role"] = "admin"
    at.run()
    assert not at.exception
    assert "token" not in at.session_state
    assert any("session expired" in i.value.lower() for i in at.info)
    assert [b.label for b in at.button] == ["Sign in"]


# ---------------------------------------------------------------------
# Viewer vs admin
# ---------------------------------------------------------------------

def test_pages_by_role_are_declared_in_one_table(bridge):
    import navigation
    assert navigation.titles_for("viewer") == ["Fleet health", "Investigations", "Jobs"]
    assert navigation.titles_for("admin") == ["Fleet health", "Investigations", "Upload data", "Jobs", "Admin"]
    assert navigation.titles_for(None) == navigation.titles_for("viewer")


def test_viewer_sees_no_write_controls(bridge, engine):
    for module in ("upload", "admin"):               # even if a viewer reaches the page, it is inert
        at = _view(bridge, module, "viewer")
        assert not at.exception and at.warning and not at.file_uploader and not at.button
    # a failed job: admin gets Retry, viewer does not
    with engine.begin() as c:
        up = queries.create_upload(c, str(uuid.uuid4()), "f.csv", "text/csv", "k", tenant_id=T)
        job = {"job_id": str(queries.create_job(c, str(uuid.uuid4()), up["upload_id"], 3, tenant_id=T)["job_id"])}
        queries.finish_job(c, job["job_id"], status="failed", error="boom", tenant_id=T)
    labels = lambda at: [b.label for b in at.button]   # noqa: E731
    viewer = _view(bridge, "jobs", "viewer")
    viewer.selectbox[1].set_value(job["job_id"]).run()
    assert not viewer.exception and "Retry this job" not in labels(viewer)
    admin = _view(bridge, "jobs", "admin")
    admin.selectbox[1].set_value(job["job_id"]).run()
    assert "Retry this job" in labels(admin)
    next(b for b in admin.button if b.label == "Retry this job").click()
    admin.run()
    assert not admin.exception and any("Queued again" in s.value for s in admin.success)


def test_viewer_can_read_fleet_and_investigate(bridge):
    at = _view(bridge, "fleet", "viewer")
    assert not at.exception and at.title[0].value == "Fleet health"
    assert "FE-1" in str(at.dataframe[0].value) and "HIGH" in str(at.dataframe[0].value)
    assert not any("OTHER-SECRET-1" in str(d.value) for d in at.dataframe)       # no other tenant's machine
    assert not at.error
    inv = _view(bridge, "investigate", "viewer")
    inv.text_area[0].set_value("Which machines look abnormal?")
    next(b for b in inv.button if b.label == "Investigate").click()
    inv.run()
    assert not inv.exception and not inv.error
    inv.run()
    assert any("Latest result" in m.value for m in inv.markdown)
    hist = _view(bridge, "investigate", "viewer")
    assert hist.expander, "the stored investigation appears in the history"
    next(b for b in hist.button if b.label == "Show result").click()
    hist.run()
    assert not hist.exception and any("Intent" in m.value for m in hist.markdown)


def test_admin_page_edits_a_sensor_and_lists_users(bridge):
    at = _view(bridge, "admin", "admin")
    assert not at.exception and not at.error
    at.checkbox(key="rz_on_temp").set_value(True)
    at.run()
    at.number_input(key="rz_temp").set_value(2.5)
    next(b for b in at.button if b.label == "Save sensor").click()
    at.run()
    assert not at.exception and any("Saved temp" in s.value for s in at.success)
    got = {s["sensor_name"]: s for s in bridge.get("/sensors", headers={"Authorization": "Bearer " + _signed_in_token(bridge, "viewer")}).json()}
    assert got["temp"]["z_score_threshold"] == 2.5


# ---------------------------------------------------------------------
# Upload wizard, happy path
# ---------------------------------------------------------------------

def test_upload_wizard_happy_path(bridge, engine):
    at = _view(bridge, "upload", "admin")
    assert not at.exception and at.file_uploader
    at.file_uploader[0].set_value(("wide.csv", WIDE_CSV, "text/csv"))
    at.run()
    next(b for b in at.button if b.label == "Upload file").click()
    at.run()
    assert not at.exception and not at.error
    wiz = at.session_state["wizard"]
    assert wiz["step"] == 1 and wiz["filename"] == "wide.csv"
    assert {c["name"]: c["detected_type"] for c in wiz["preview"]["columns"]}["Temp"] == "number"

    next(b for b in at.button if b.label == "Next: map columns").click()
    at.run()
    at.selectbox(key="map_machine").set_value("Asset")
    at.run()
    assert at.selectbox(key="map_ts").value == "At"            # the timestamp column was detected
    at.selectbox(key="map_unit_Temp").set_value("C")
    at.selectbox(key="map_unit_Press").set_value("kPa")
    at.run()
    at.selectbox(key="map_target_Press_kPa").set_value("bar")
    at.text_input(key="map_hi_Temp").set_value("90")
    at.run()
    next(b for b in at.button if b.label == "Validate on a sample").click()
    at.run()
    assert not at.exception and not at.error
    dry = at.session_state["wizard"]["dry_run"]
    assert dry["valid"] and dry["report"]["readings_accepted"] == 60 and dry["report"]["total_errors"] == 0
    mapping = at.session_state["wizard"]["mapping"]
    assert mapping["sensors"]["press"] == {"source": "Press", "unit": "kPa", "target_unit": "bar"}
    assert mapping["sensors"]["temp"]["normal_max"] == 90.0

    next(b for b in at.button if b.label == "Confirm and start the job").click()
    at.run()
    assert not at.exception and any("queued" in s.value for s in at.success)
    job = at.session_state["wizard"]["job"]
    with engine.connect() as c:
        row = c.execute(text("SELECT j.status, u.filename, u.mapping IS NOT NULL FROM jobs j JOIN uploads u "
                             "ON u.upload_id = j.upload_id WHERE j.job_id = :j"), {"j": job["job_id"]}).one()
    assert tuple(row) == ("queued", "wide.csv", True)


def test_upload_wizard_reports_a_bad_mapping_and_blocks_submit(bridge):
    at = _view(bridge, "upload", "admin")
    at.file_uploader[0].set_value(("wide.csv", WIDE_CSV, "text/csv"))
    at.run()
    next(b for b in at.button if b.label == "Upload file").click()
    at.run()
    next(b for b in at.button if b.label == "Next: map columns").click()
    at.run()
    at.selectbox(key="map_machine").set_value("Asset")
    at.run()
    at.selectbox(key="map_unit_Temp").set_value("kPa")           # pressure unit on a column, then target in temperature
    at.run()
    at.selectbox(key="map_target_Temp_kPa").set_value("bar")
    at.selectbox(key="map_unit_Press").set_value("C")
    at.run()
    at.selectbox(key="map_target_Press_C").set_value("K")
    at.run()
    next(b for b in at.button if b.label == "Validate on a sample").click()
    at.run()
    assert not at.exception
    # compatible units were chosen (kPa->bar, C->K), so this validates; now break it with pasted JSON
    at.session_state["wizard"]["step"] = 2
    at.run()
    at.text_area(key="map_json").set_value('{"dataset_kind": "sensor_readings", "format": "wide"}')
    next(b for b in at.button if b.label == "Validate on a sample").click()
    at.run()
    assert not at.exception and at.error, "the API's mapping problems are shown, not swallowed"
    assert at.session_state["wizard"]["step"] == 2             # did not advance


def test_a_failed_upload_call_shows_the_api_error(bridge, monkeypatch):
    at = _view(bridge, "upload", "admin")
    at.file_uploader[0].set_value(("wide.csv", WIDE_CSV, "text/csv"))
    at.run()
    limiter.reset()
    with env(UPLOAD_MAX_BYTES="10"):
        next(b for b in at.button if b.label == "Upload file").click()
        at.run()
    assert not at.exception and at.error and at.session_state["wizard"]["step"] == 0
