"""
Production-hardening fixes 5-8: CORS, rate limiting, LLM timeout/retry,
planner failure degradation, and the graph's hard step ceiling.
"""

from __future__ import annotations

import importlib
import os
from datetime import datetime
from unittest.mock import patch

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from langgraph.errors import GraphRecursionError

from app.agents.graph import run_investigation
from app.api.rate_limit import SlidingWindowRateLimiter
from app.core.tenancy import generate_api_key
from app.main import app
from tests.helpers import env

load_dotenv()

needs_db = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")


# ---------------------------------------------------------------------
# 5. CORS
# ---------------------------------------------------------------------

def _preflight(client: TestClient, origin: str):
    return client.options(
        "/machines",
        headers={"Origin": origin, "Access-Control-Request-Method": "GET",
                 "Access-Control-Request-Headers": "X-API-Key"},
    )


def test_no_cors_headers_by_default():
    import app.main as main_module

    with env(CORS_ALLOW_ORIGINS=None):
        reloaded = importlib.reload(main_module)
        with TestClient(reloaded.app) as client:
            r = _preflight(client, "https://evil.example")
            assert "access-control-allow-origin" not in r.headers


def test_cors_allow_list_admits_only_listed_origins():
    import app.main as main_module

    try:
        with env(CORS_ALLOW_ORIGINS="https://ops.example.com, http://localhost:8501"):
            reloaded = importlib.reload(main_module)
            with TestClient(reloaded.app) as client:
                ok = _preflight(client, "https://ops.example.com")
                assert ok.headers["access-control-allow-origin"] == "https://ops.example.com"
                assert "x-api-key" in ok.headers["access-control-allow-headers"].lower()
                assert "access-control-allow-credentials" not in ok.headers
                bad = _preflight(client, "https://evil.example")
                assert "access-control-allow-origin" not in bad.headers
    finally:
        with env(CORS_ALLOW_ORIGINS=None):
            importlib.reload(main_module)


# ---------------------------------------------------------------------
# 6. Rate limiting
# ---------------------------------------------------------------------

def test_sliding_window_limiter_allows_then_blocks_then_recovers():
    now = [1000.0]
    limiter = SlidingWindowRateLimiter(clock=lambda: now[0])
    assert limiter.check("k", 2, 60) is None
    assert limiter.check("k", 2, 60) is None
    wait = limiter.check("k", 2, 60)
    assert wait is not None and 0 < wait <= 60
    assert limiter.check("other", 2, 60) is None       # keys are independent
    now[0] += 61
    assert limiter.check("k", 2, 60) is None            # window slid past the old hits


def test_limit_zero_disables_limiting():
    limiter = SlidingWindowRateLimiter()
    assert all(limiter.check("k", 0, 60) is None for _ in range(1000))


@needs_db
def test_llm_endpoints_return_429_with_retry_after_and_reads_are_not_limited():
    body = {"question": "Which machines currently show abnormal behavior?"}
    with env(RATE_LIMIT_PER_MINUTE="2"), TestClient(app) as client:
        assert client.post("/investigate", json=body).status_code == 200
        assert client.post("/chat", json={"message": body["question"]}).status_code == 200  # shared budget
        blocked = client.post("/investigate", json=body)
        assert blocked.status_code == 429
        assert int(blocked.headers["retry-after"]) >= 1
        assert client.get("/machines").status_code == 200  # only the billed endpoints are throttled


@needs_db
def test_rate_limit_is_per_tenant_not_global():
    from sqlalchemy import create_engine, text

    from app.core.tenancy import hash_api_key

    key = generate_api_key()
    engine = create_engine(os.environ["DATABASE_URL"])
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM investigations WHERE tenant_id = 'ratelimit-test'"))
        conn.execute(text("DELETE FROM investigation_audit_log WHERE tenant_id = 'ratelimit-test'"))
        conn.execute(text("DELETE FROM tenants WHERE tenant_id = 'ratelimit-test'"))
        conn.execute(text("INSERT INTO tenants (tenant_id, name, api_key_hash) VALUES "
                          "('ratelimit-test', 'RL', :h)"), {"h": hash_api_key(key)})
    body = {"question": "Which machines currently show abnormal behavior?"}
    try:
        with env(RATE_LIMIT_PER_MINUTE="1"), TestClient(app) as client:
            assert client.post("/investigate", json=body).status_code == 200            # demo tenant, uses its 1
            assert client.post("/investigate", json=body).status_code == 429            # demo tenant exhausted
            other = client.post("/investigate", json=body, headers={"X-API-Key": key})  # different tenant unaffected
            assert other.status_code == 200
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM investigations WHERE tenant_id = 'ratelimit-test'"))
            conn.execute(text("DELETE FROM investigation_audit_log WHERE tenant_id = 'ratelimit-test'"))
            conn.execute(text("DELETE FROM tenants WHERE tenant_id = 'ratelimit-test'"))


# ---------------------------------------------------------------------
# 7. LLM timeout + retry
# ---------------------------------------------------------------------

def test_genai_clients_are_built_with_the_configured_timeout_and_retry(monkeypatch):
    genai = pytest.importorskip("google.genai")
    captured: dict = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(genai, "Client", FakeClient)
    from app.core.genai_client import build_genai_client

    with env(LLM_TIMEOUT_SECONDS="12.5", LLM_MAX_ATTEMPTS="3"):
        build_genai_client("proj", "global")
    assert captured["vertexai"] is True
    assert captured["project"] == "proj" and captured["location"] == "global"
    options = captured["http_options"]
    assert options.timeout == 12500  # milliseconds
    assert options.retry_options.attempts == 3


def test_every_llm_call_site_uses_the_shared_client_builder():
    """Regression guard: synthesis, planner and embedder must not go back to
    a bare genai.Client(...) with no timeout."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "app"
    offenders = [
        str(path.relative_to(root)) for path in root.rglob("*.py")
        if path.name != "genai_client.py" and "genai.Client(" in path.read_text()
    ]
    assert offenders == []


class _DeadPlannerLLM:
    """A planner whose Vertex call times out on every attempt."""

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        raise TimeoutError("vertex deadline exceeded")


def test_a_dead_planner_llm_degrades_the_investigation_instead_of_500ing():
    pytest.importorskip("langchain_core")
    health = {"machine_id": "M-04", "window_start": datetime(2026, 1, 1, 10),
              "window_end": datetime(2026, 1, 1, 11), "health_status": "AT_RISK",
              "max_anomaly_score": 92, "anomalous_reading_count": 40, "reading_count": 60}
    anomaly = {"detected_at": datetime(2026, 1, 1, 10, 30), "anomaly_score": 92,
               "severity": "HIGH", "triggered_reasons": "OSF"}

    class NoDocs:
        backend_name = "test"

        def search(self, query, top_k=5, filters=None):
            return []

    with patch("app.database.queries.get_machine_health", return_value=health), \
            patch("app.database.queries.get_sensor_summary", return_value=[health]), \
            patch("app.database.queries.get_machine_anomalies", return_value=[anomaly]), \
            patch("app.database.queries.get_maintenance_records", return_value=[]), \
            patch("app.database.queries.get_ai4i_failure_mode_rates", return_value={}):
        result = run_investigation(
            "Why is M-04 failing?", "M-04", conn=object(), retriever=NoDocs(),
            planner_llm=_DeadPlannerLLM(),
        )
    assert result["risk_level"] == "HIGH"          # mandatory evidence was unaffected
    assert result["recommendation"]
    assert any("planner unavailable" in e.lower() for e in result["errors"])


# ---------------------------------------------------------------------
# 8. Hard step ceiling
# ---------------------------------------------------------------------

def _run_simple():
    health = {"machine_id": "M-04", "window_start": datetime(2026, 1, 1, 10),
              "window_end": datetime(2026, 1, 1, 11), "health_status": "HEALTHY",
              "max_anomaly_score": 0, "anomalous_reading_count": 0, "reading_count": 60}
    with patch("app.database.queries.get_machine_health", return_value=health), \
            patch("app.database.queries.get_sensor_summary", return_value=[health]), \
            patch("app.database.queries.get_machine_anomalies", return_value=[]), \
            patch("app.database.queries.get_maintenance_records", return_value=[]), \
            patch("app.database.queries.get_ai4i_failure_mode_rates", return_value={}):
        return run_investigation("status?", "M-04", conn=object(), retriever=None)


def test_graph_recursion_limit_comes_from_settings_and_stops_a_runaway():
    with env(GRAPH_RECURSION_LIMIT="3"):
        with pytest.raises(GraphRecursionError):
            _run_simple()
    with env(GRAPH_RECURSION_LIMIT=None):
        assert _run_simple()["risk_level"] == "LOW"  # default ceiling leaves plenty of headroom


def test_planner_round_cap_is_enforced_in_state():
    """The audit HTML said no planner iteration cap exists; one does
    (planner.MAX_PLANNER_TOOL_ROUNDS, checked in nodes.plan_supplementary_evidence).
    Pin it so it can't quietly disappear."""
    pytest.importorskip("langchain_core")
    from app.agents import nodes, planner

    calls = {"n": 0}

    class LoopingLLM:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            calls["n"] += 1
            from langchain_core.messages import AIMessage
            return AIMessage(content="", tool_calls=[
                {"name": "get_historical_evidence", "args": {}, "id": f"c{calls['n']}"}])

    node = nodes.plan_supplementary_evidence(LoopingLLM(), [])
    state: dict = {"machine_id": "M-04", "user_query": "q", "planner_rounds": planner.MAX_PLANNER_TOOL_ROUNDS}
    out = node(state)
    assert out["planner_done"] is True and calls["n"] == 0
