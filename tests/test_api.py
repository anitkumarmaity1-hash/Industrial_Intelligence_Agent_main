"""
Phase 4 tests: the FastAPI layer.

Same DATABASE_URL-gating pattern as tests/test_postgres.py — the
database-backed tests skip cleanly without a live Postgres instance, so
this file still runs (and still asserts something) in CI/offline. The
few pure-dependency tests at the bottom need neither DB nor app.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv
from fastapi import HTTPException
from fastapi.testclient import TestClient

from sqlalchemy.exc import OperationalError

from app.api.dependencies import get_conn, get_db_engine
from app.main import app

load_dotenv()

# Same known ground truth as test_postgres.py's six injected scenarios.
SCENARIO_MACHINES = {"M-01", "M-02", "M-03", "M-04", "M-05", "M-06"}
HEALTHY_MACHINES = {f"M-{i:02d}" for i in range(7, 19)}


@pytest.fixture(scope="module")
def client():
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------

def test_root_is_reachable_without_a_database():
    """The landing endpoint touches no dependency, so it must work even
    with no DATABASE_URL at all — confirms lifespan doesn't crash startup."""
    with TestClient(app) as test_client:
        response = test_client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "industrial-intelligence-agent"


# ---------------------------------------------------------------------
# GET /machines, GET /machines/{id}
# ---------------------------------------------------------------------

def test_list_machines_returns_all_18(client):
    response = client.get("/machines")
    assert response.status_code == 200
    machines = response.json()
    assert len(machines) == 18
    assert {m["machine_id"]
            for m in machines} == SCENARIO_MACHINES | HEALTHY_MACHINES


def test_get_machine_known_id(client):
    response = client.get("/machines/M-01")
    assert response.status_code == 200
    body = response.json()
    assert body["machine_id"] == "M-01"
    assert body["production_line"] == "LINE-1"
    # Never fabricated per Phase 1/2 — must stay null, not defaulted.
    assert body["name"] is None


def test_get_machine_unknown_id_is_404(client):
    response = client.get("/machines/M-99")
    assert response.status_code == 404


# ---------------------------------------------------------------------
# GET /machines/rank-for-inspection
# ---------------------------------------------------------------------

def test_rank_for_inspection_is_not_swallowed_by_machine_id_route(client):
    """Regression guard for the registration-order dependency this route
    documents in its own docstring: if it were ever moved after
    GET /machines/{machine_id}, "rank-for-inspection" would 404 as an
    unknown machine_id instead of running this endpoint."""
    response = client.get(
        "/machines/rank-for-inspection", params={"since": "2026-01-01T00:00:00"})
    assert response.status_code == 200


def test_rank_for_inspection_surfaces_sustained_scenario_machines(client):
    """Same ground truth as test_postgres.py's direct query_layer test,
    now confirmed reachable through the API — P2 cleanup: this query
    existed and was tested since Phase 2 but had no caller at all."""
    response = client.get(
        "/machines/rank-for-inspection",
        params={"since": "2026-01-01T00:00:00", "top_n": 5},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 5
    assert body[0]["machine_id"] == "M-06"
    assert "anomaly_count" in body[0]
    assert "high_severity_count" in body[0]


# ---------------------------------------------------------------------
# GET /machines/{id}/health
# ---------------------------------------------------------------------

def test_get_health_for_unresolved_osf_machine_is_not_healthy(client):
    """M-04 has an unrepaired OSF scenario — mirrors test_postgres.py's
    assertion at the API layer."""
    response = client.get("/machines/M-04/health")
    assert response.status_code == 200
    assert response.json()["health_status"] in ("WATCH", "AT_RISK")


def test_get_health_unknown_machine_is_404(client):
    response = client.get("/machines/M-99/health")
    assert response.status_code == 404


def test_get_health_risk_level_agrees_with_investigate(client):
    """Audit F8 (finding 3) regression: the Overview tab's badge (this
    endpoint) and the Investigate tab's badge (/investigate) used to come
    from two different rules and could disagree (M-04 shown as WATCH on
    one, HIGH on the other, for the same underlying data). Both now call
    app.agents.risk.assess_risk over the same default evidence windows,
    so for the same machine at the same moment they must agree exactly.
    """
    health = client.get("/machines/M-04/health")
    assert health.status_code == 200
    assert "risk_level" in health.json()

    investigate = client.post(
        "/investigate", json={"machine_id": "M-04", "question": "Why is M-04 underperforming?"})
    assert investigate.status_code == 200

    assert health.json()["risk_level"] == investigate.json()["risk_level"]


def test_get_health_includes_measured_metrics(client):
    """Audit F11 regression: the Overview tab's "key metrics" row used to
    have nothing to show but window_start/window_end. get_machine_health
    now also selects avg_defect_rate and tool_wear_min from sensor_summary."""
    response = client.get("/machines/M-04/health")
    assert response.status_code == 200
    body = response.json()
    assert body["avg_defect_rate"] is not None
    assert body["tool_wear_min"] is not None


# ---------------------------------------------------------------------
# GET /machines/{id}/sensors
# ---------------------------------------------------------------------

def test_get_sensors_returns_hourly_trend(client):
    """Audit F11: backs the Streamlit trend chart. Same underlying query
    (queries.get_sensor_summary) the agent's evidence-gathering tools
    already used internally, now exposed as its own endpoint."""
    response = client.get("/machines/M-04/sensors", params={"limit": 24})
    assert response.status_code == 200
    points = response.json()
    assert 0 < len(points) <= 24
    assert {"window_start", "window_end", "avg_defect_rate",
            "tool_wear_min", "health_status"} <= points[0].keys()


def test_get_sensors_unknown_machine_is_404(client):
    response = client.get("/machines/M-99/sensors")
    assert response.status_code == 404


# ---------------------------------------------------------------------
# GET /machines/{id}/anomalies
# ---------------------------------------------------------------------

def test_get_anomalies_for_scenario_machine(client):
    response = client.get("/machines/M-01/anomalies")
    assert response.status_code == 200
    anomalies = response.json()
    assert len(anomalies) > 0
    assert all(a["severity"] in ("MEDIUM", "HIGH") for a in anomalies)


def test_get_anomalies_severity_filter_is_applied(client):
    response = client.get("/machines/M-01/anomalies",
                          params={"severity": "HIGH"})
    assert response.status_code == 200
    assert all(a["severity"] == "HIGH" for a in response.json())


def test_get_anomalies_rejects_invalid_severity(client):
    response = client.get("/machines/M-01/anomalies",
                          params={"severity": "LOW"})
    assert response.status_code == 422  # not a valid Severity literal


def test_get_anomalies_unknown_machine_is_404(client):
    response = client.get("/machines/M-99/anomalies")
    assert response.status_code == 404


# ---------------------------------------------------------------------
# POST /investigate, POST /chat
# ---------------------------------------------------------------------

def test_investigate_returns_a_real_report_for_a_scenario_machine(client):
    """M-04 has an unresolved OSF scenario (same ground truth as
    test_postgres.py) — the agent should come back with a non-trivial
    risk assessment and a populated narrative, not a stub."""
    response = client.post(
        "/investigate", json={"machine_id": "M-04", "question": "Why is M-04 underperforming?"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "machine_investigation"
    assert body["risk_level"] in ("MEDIUM", "HIGH")
    assert body["narrative"].startswith("## Machine")
    assert "M-04" in body["narrative"]


def test_investigate_still_validates_machine_id(client):
    """An unknown machine_id must still 404 before the graph ever runs."""
    response = client.post(
        "/investigate", json={"machine_id": "M-99", "question": "Why?"})
    assert response.status_code == 404


def test_investigate_without_machine_id_runs_fleet_scan(client):
    """No machine_id routes to the fleet-wide branch instead of erroring."""
    response = client.post(
        "/investigate", json={"question": "Which machines currently show abnormal behavior?"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "fleet_scan"
    assert body["fleet_ranking"] is not None
    assert len(body["fleet_ranking"]) == 18


def test_investigate_rejects_oversized_question(client):
    """Audit F10: question previously had no max_length — a 200 KB body
    used to sail through as a 200, and each character in a Gemini-backed
    request is billed cost/latency."""
    response = client.post(
        "/investigate", json={"machine_id": "M-04", "question": "x" * 2001})
    assert response.status_code == 422


# Phase 9 fix: /chat ran a real investigation and returned a populated
# report all along at the graph level — the endpoint was the only thing
# stubbed to a 501. These mirror the /investigate coverage above, through
# /chat's request/response shape (message/reply instead of question/
# recommendation).

def test_chat_returns_a_real_report_for_a_scenario_machine(client):
    response = client.post(
        "/chat", json={"machine_id": "M-04", "message": "Why is M-04 underperforming?"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "machine_investigation"
    assert body["risk_level"] in ("MEDIUM", "HIGH")
    assert body["narrative"].startswith("## Machine")
    assert body["reply"]  # non-empty — this is the field a chat UI shows


def test_chat_still_validates_machine_id(client):
    response = client.post(
        "/chat", json={"machine_id": "M-99", "message": "Why?"})
    assert response.status_code == 404


def test_chat_without_machine_id_runs_fleet_scan(client):
    response = client.post(
        "/chat", json={"message": "Which machines need inspection?"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "fleet_scan"


# ---------------------------------------------------------------------
# get_db_engine dependency — no DB or app needed
# ---------------------------------------------------------------------

class _FakeState:
    db_engine = None


class _FakeRequest:
    class app:
        state = _FakeState()


def test_get_db_engine_raises_503_when_unconfigured():
    with pytest.raises(HTTPException) as exc_info:
        get_db_engine(_FakeRequest())  # type: ignore[arg-type]
    assert exc_info.value.status_code == 503


class _FailingEngine:
    """Stands in for a configured-but-unreachable Engine — e.g. the wrong
    password, or Postgres down. Raises only when connect() is actually
    called, mirroring get_db_engine (well-formed URL) succeeding while the
    real socket/auth attempt (get_conn) is what fails."""

    def connect(self):
        raise OperationalError("connect", {}, Exception(
            "password authentication failed"))


class _FailingEngineState:
    db_engine = _FailingEngine()


class _FailingEngineRequest:
    class app:
        state = _FailingEngineState()


def test_get_conn_raises_503_on_connect_failure_not_a_raw_exception():
    """Pins the fix for a bad DB password/outage surfacing as a clean 503
    instead of an unhandled SQLAlchemyError with a raw driver traceback."""
    with pytest.raises(HTTPException) as exc_info:
        next(get_conn(_FailingEngineRequest()))  # type: ignore[arg-type]
    assert exc_info.value.status_code == 503


def test_get_conn_503_detail_does_not_leak_driver_exception_text():
    """Audit F10 regression: the 503 detail used to interpolate the raw
    SQLAlchemyError, which leaked infrastructure detail (DB host/port,
    auth failure specifics) straight into the HTTP response body. The
    caller now gets a generic message; the real detail still reaches logs
    (app.api.dependencies logs it server-side before raising)."""
    with pytest.raises(HTTPException) as exc_info:
        next(get_conn(_FailingEngineRequest()))  # type: ignore[arg-type]
    assert "password" not in exc_info.value.detail
    assert "localhost" not in exc_info.value.detail


