"""HTTP client for the dashboard (Phase 6). The dashboard talks to the API ONLY, with the
signed-in user's JWT; it never opens a database connection and never names a tenant.

State lives in st.session_state (one browser session): token, role, tenant, email.
Read calls are cached for a short time with a key that includes a hash of the token, so
one user's cached data is never served to another (and logging out clears the cache).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
CACHE_TTL_SECONDS = 20


class ApiError(Exception):
    """A failed API call, with a message that is safe to show the user."""

    def __init__(self, message: str, status: int | None = None, detail: Any = None):
        super().__init__(message)
        self.status = status
        self.detail = detail


def base_url() -> str:
    return st.session_state.get("api_base_url", API_BASE_URL).rstrip("/")


def token() -> str | None:
    return st.session_state.get("token")


def is_admin() -> bool:
    return st.session_state.get("role") == "admin"


def _detail_text(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        msg = str(detail.get("message", "Request rejected."))
        problems = detail.get("problems") or detail.get("missing_columns")
        return msg + ("\n- " + "\n- ".join(map(str, problems)) if problems else "")
    if isinstance(detail, list):       # FastAPI validation errors
        return "; ".join(str(d.get("msg", d)) if isinstance(d, dict) else str(d) for d in detail)
    return "Request rejected."


def _request(method: str, path: str, tok: str | None, timeout: int = 30, **kwargs) -> Any:
    headers = dict(kwargs.pop("headers", {}) or {})
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    url = f"{base_url()}{path}"
    try:
        resp = requests.request(method, url, headers=headers, timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError as exc:
        raise ApiError(f"Can't reach the API at {base_url()}.") from exc
    except requests.exceptions.Timeout as exc:
        raise ApiError("The request timed out.") from exc
    if resp.status_code < 400:
        return None if resp.status_code == 204 or not resp.content else resp.json()
    try:
        detail = resp.json().get("detail")
    except Exception:  # noqa: BLE001
        detail = None
    if resp.status_code == 401:
        raise ApiError("Your session is not valid (expired or signed out).", 401, detail)
    if resp.status_code == 403:
        raise ApiError("Your role does not allow this action.", 403, detail)
    if resp.status_code == 429:
        raise ApiError("Rate limit reached - wait a moment and try again "
                       f"(Retry-After: {resp.headers.get('Retry-After', '?')}s).", 429, detail)
    if resp.status_code == 503:
        raise ApiError("The service is unavailable right now. Try again shortly.", 503, detail)
    raise ApiError(_detail_text(detail) if detail else f"Request failed ({resp.status_code}).",
                   resp.status_code, detail)


def _token_key(tok: str | None) -> str:
    return hashlib.sha256((tok or "").encode()).hexdigest()[:16]


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _cached_get(base: str, token_key: str, path: str, params_json: str, _tok: str | None) -> Any:
    # token_key (a hash) is the cache key; the raw token is passed with a leading underscore,
    # which Streamlit excludes from hashing. Exceptions are never cached.
    return _request("GET", path, _tok, params=json.loads(params_json) or None)


def _expired() -> None:
    """The API said 401: drop the session and go back to the sign-in page."""
    sign_out(message="Your session expired. Please sign in again.")
    st.rerun()


def get(path: str, params: dict | None = None, cached: bool = True) -> Any:
    tok = token()
    try:
        if cached:
            return _cached_get(base_url(), _token_key(tok), path,
                               json.dumps(params or {}, sort_keys=True, default=str), tok)
        return _request("GET", path, tok, params=params)
    except ApiError as exc:
        if exc.status == 401 and tok:
            _expired()
        raise


def post(path: str, payload: dict | None = None, timeout: int = 30, **kwargs) -> Any:
    tok = token()
    try:
        if payload is not None:
            kwargs["json"] = payload
        return _request("POST", path, tok, timeout=timeout, **kwargs)
    except ApiError as exc:
        if exc.status == 401 and tok:
            _expired()
        raise
    finally:
        _cached_get.clear()        # a write may change what the reads show


def sign_in(tenant: str, email: str, password: str) -> None:
    out = _request("POST", "/auth/login", None,
                   json={"tenant": tenant.strip(), "email": email.strip(), "password": password})
    st.session_state.update(token=out["access_token"], role=out["role"],
                            tenant=tenant.strip().lower(), email=email.strip().lower(), notice=None)
    _cached_get.clear()


def sign_out(message: str | None = None) -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    _cached_get.clear()
    if message:
        st.session_state["notice"] = message
