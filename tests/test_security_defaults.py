"""Phase 1, task 4: secure-by-default auth, /healthz, /readyz, compose creds."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app.core.config import database_is_remote, get_settings
from app.main import app
from tests.helpers import env

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("url,remote", [
    (None, False), ("", False),
    ("postgresql://u:p@localhost:5432/db", False),
    ("postgresql://u:p@127.0.0.1/db", False), ("postgresql://u:p@127.0.1.1/db", False),
    ("postgresql://u:p@[::1]:5432/db", False), ("postgresql:///db", False),
    # docker-compose service name
    ("postgresql://u:p@postgres:5432/db", True),
    ("postgresql://u:p@db.internal.example.com/db", True),
    ("postgresql://u:p@10.0.0.5/db",
     True), ("postgresql://u:p@LOCALHOST.evil.com/db", True),
])
def test_database_is_remote(url, remote):
    assert database_is_remote(url) is remote


@pytest.mark.parametrize("db,auth,api_key,expected", [
    # local demo posture unchanged
    ("postgresql://u:p@localhost/db", None, None, False),
    ("postgresql://u:p@db.example.com/db", None, None, True),   # NEW default
    ("postgresql://u:p@db.example.com/db", "false",
     None, False),   # explicit opt-out still honoured
    ("postgresql://u:p@localhost/db", "true", None, True),
    ("postgresql://u:p@localhost/db", None, "k", True),        # legacy rule intact
])
def test_auth_required_default(db, auth, api_key, expected):
    with env(DATABASE_URL=db, AUTH_REQUIRED=auth, API_KEY=api_key):
        assert get_settings().auth_required is expected


def test_healthz_needs_nothing():
    with env(DATABASE_URL=None, AUTH_REQUIRED="true"):
        with TestClient(app) as c:
            r = c.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_readyz_ready_with_database_and_503_without():
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    with env(AUTH_REQUIRED="true"):
        with TestClient(app) as c:
            ok = c.get("/readyz")
    assert ok.status_code == 200 and ok.json() == {"status": "ready"}
    with env(DATABASE_URL=None, APP_DATABASE_URL=None):
        with TestClient(app) as c:
            assert c.get("/readyz").status_code == 503
    bad = "postgresql://nobody:secret@127.0.0.1:1/none"
    with env(DATABASE_URL=bad, APP_DATABASE_URL=bad):
        with TestClient(app) as c:
            r = c.get("/readyz")
    assert r.status_code == 503 and r.json(
    )["detail"] == "database unavailable"
    assert "secret" not in r.text and "127.0.0.1" not in r.text       # nothing leaks


def test_compose_has_no_hardcoded_database_credentials():
    raw = (ROOT / "docker-compose.yml").read_text()
    assert "postgres:postgres" not in raw
    cfg = yaml.safe_load(raw)["services"]
    pw = cfg["postgres"]["environment"]["POSTGRES_PASSWORD"]
    # required, no default
    assert pw.startswith("${POSTGRES_PASSWORD:?"), pw
    api_env = cfg["api"]["environment"]
    # Phase 2: the API gets only the least-privilege role's URL, never owner credentials.
    assert "APP_DB_PASSWORD:?" in api_env["APP_DATABASE_URL"]
    assert "DATABASE_URL" not in api_env and "POSTGRES_PASSWORD" not in str(
        api_env)
    assert "APP_DB_PASSWORD:?" in cfg["postgres"]["environment"]["APP_DB_PASSWORD"]
    assert cfg["api"]["environment"]["API_KEY"].startswith("${API_KEY:?")
    assert cfg["postgres"]["ports"] == ["127.0.0.1:5434:5432"]
