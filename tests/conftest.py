"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from app.api.rate_limit import limiter


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """The rate limiter is process-wide state; without this, request counts
    from one test would leak into the next and cause spurious 429s."""
    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture(autouse=True)
def _isolate_auth_env(monkeypatch):
    """app.core.config calls load_dotenv() at import, so a developer's local
    .env (API_KEY for docker compose, AUTH_REQUIRED) would silently switch
    auth on and break tests that assert the zero-config demo posture.
    Tests that care set these explicitly (tests' `env(...)` helper)."""
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    from app.core.config import get_settings
    # Settings is cached; drop what .env put in it
    get_settings(refresh=True)
    yield
    monkeypatch.undo()
    get_settings(refresh=True)
