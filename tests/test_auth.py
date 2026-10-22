"""Phase 4: users, roles, login.

Runs against a real Postgres (skips without DATABASE_URL). When APP_DATABASE_URL points at
the least-privilege iia_app role (as in CI) every request below runs under RLS.
"""

from __future__ import annotations

import base64
import importlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import jwt
import pytest
from dotenv import load_dotenv
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

from app.api.rate_limit import limiter
from app.core.permissions import (
    PUBLIC_ROUTES, ROLE_MATRIX, ROUTE_PERMISSIONS, WRITE_PERMISSIONS, Permission, role_allows,
)
from app.core.security import create_token, hash_password
from app.core.tenancy import generate_api_key, hash_api_key
from app.main import app
from tests.helpers import env

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
TA, TB = "auth-a-test", "auth-b-test"
KEY_A, KEY_B = generate_api_key(), generate_api_key()
SECRET = "phase4-test-secret-" + "0123456789abcdef" * 2
PW = "correct horse battery staple"
JSON = {"content-type": "application/json"}


@pytest.fixture(scope="module")
def engine():
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    return create_engine(url)


def _cleanup(conn) -> None:
    for table in ("investigations", "audit_log", "jobs", "uploads", "users", "machines"):
        conn.execute(text(f"DELETE FROM {table} WHERE tenant_id IN (:a, :b)"), {
                     "a": TA, "b": TB})
    conn.execute(
        text("DELETE FROM uploads WHERE filename = 'phase4-legacy.csv'"))
    conn.execute(text("DELETE FROM tenants WHERE tenant_id IN (:a, :b)"), {
                 "a": TA, "b": TB})


@pytest.fixture(scope="module", autouse=True)
def tenants(engine):
    with engine.begin() as c:
        _cleanup(c)
        c.execute(text("INSERT INTO tenants (tenant_id, name, api_key_hash) VALUES "
                       "(:a, 'Auth A', :ha), (:b, 'Auth B', :hb)"),
                  {"a": TA, "b": TB, "ha": hash_api_key(KEY_A), "hb": hash_api_key(KEY_B)})
        c.execute(text("INSERT INTO machines (tenant_id, machine_id, production_line, type) VALUES "
                       "(:a, 'A-1', 'LA', 'L'), (:b, 'B-1', 'LB', 'L')"), {"a": TA, "b": TB})
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
    """Insert a user straight through the owner connection; returns (id, email, bearer headers)."""
    def _make(tenant=TA, role="viewer", active=True, password=PW):
        uid, email = str(uuid.uuid4()), f"{uuid.uuid4().hex[:10]}@example.com"
        with engine.begin() as c:
            c.execute(text("INSERT INTO users (id, tenant_id, email, password_hash, role, is_active) "
                           "VALUES (:i, :t, :e, :h, :r, :a)"),
                      {"i": uid, "t": tenant, "e": email, "h": hash_password(password), "r": role, "a": active})
        return uid, email, bearer(uid, tenant)
    return _make


def bearer(uid: str, tenant: str, ttl: int = 60, secret: str = SECRET) -> dict[str, str]:
    return {"Authorization": "Bearer " + create_token(
        user_id=uid, tenant_id=tenant, secret=secret, ttl_minutes=ttl)}


def api(key: str) -> dict[str, str]:
    return {"X-API-Key": key}


def login(client, tenant, email, password=PW):
    return client.post("/auth/login", json={"tenant": tenant, "email": email, "password": password})


def audit_rows(engine, tenant, action=None):
    with engine.connect() as c:
        rows = c.execute(text(
            "SELECT action, outcome, request_id, principal_kind, actor, detail FROM audit_log "
            "WHERE tenant_id = :t AND (CAST(:a AS text) IS NULL OR action = :a) ORDER BY id"),
            {"t": tenant, "a": action}).mappings().all()
    return [dict(r) for r in rows]


def _fill(path: str) -> str:
    out = path
    for name in ("machine_id", "upload_id", "job_id", "user_id"):
        out = out.replace("{" + name + "}", "M-01" if name ==
                          "machine_id" else str(uuid.uuid4()))
    return out


# ---------------------------------------------------------------------
# Every route declares a permission (the test that fails on a new, unguarded route)
# ---------------------------------------------------------------------

def test_every_registered_route_declares_a_permission():
    seen = set()
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue                      # /docs, /redoc, /openapi.json
        for method in route.methods - {"HEAD", "OPTIONS"}:
            key = (method, route.path)
            seen.add(key)
            assert key in ROUTE_PERMISSIONS or key in PUBLIC_ROUTES, (
                f"{key} has no permission in app/core/permissions.py "
                "(add it to ROUTE_PERMISSIONS, or to PUBLIC_ROUTES if it truly needs no credential)")
            assert not (key in ROUTE_PERMISSIONS and key in PUBLIC_ROUTES), key
    stale = (set(ROUTE_PERMISSIONS) | set(PUBLIC_ROUTES)) - seen
    assert not stale, f"declared but not registered: {sorted(stale)}"


def test_every_guarded_route_runs_the_enforcement_dependency():
    from app.api.dependencies import enforce_route_permission

    def calls(dep):
        yield dep.call
        for sub in dep.dependencies:
            yield from calls(sub)

    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in route.methods - {"HEAD", "OPTIONS"}:
            if (method, route.path) in ROUTE_PERMISSIONS:
                assert enforce_route_permission in set(
                    calls(route.dependant)), (method, route.path)


def test_an_undeclared_route_is_refused_not_allowed(client, make_user, monkeypatch):
    _, _, admin = make_user(role="admin")
    assert client.get("/sensors", headers=admin).status_code == 200
    monkeypatch.delitem(ROUTE_PERMISSIONS, ("GET", "/sensors"))
    # fails closed, even for an admin
    assert client.get("/sensors", headers=admin).status_code == 403


def test_role_matrix_shape():
    assert role_allows("admin", Permission.USERS_MANAGE) and not role_allows(
        "viewer", Permission.USERS_MANAGE)
    assert ROLE_MATRIX["admin"] == frozenset(Permission)
    for perm in WRITE_PERMISSIONS:
        assert not role_allows("viewer", perm)
    for perm in (Permission.USERS_MANAGE, Permission.JOBS_RETRY, Permission.REGISTRY_WRITE,
                 Permission.CONNECTORS_WRITE):
        assert not role_allows("api_key", perm)
    assert role_allows("api_key", Permission.UPLOADS_WRITE) and role_allows(
        "api_key", Permission.MAPPING_WRITE)
    assert not role_allows(None, Permission.DATA_READ) and not role_allows(
        "nope", Permission.DATA_READ)


# ---------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------

def test_login_returns_a_working_short_lived_token(client, make_user, engine):
    _, email, _ = make_user(role="viewer")
    # email is case-insensitive
    r = login(client, TA, email.upper())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["token_type"] == "bearer" and body["role"] == "viewer" and body["expires_in"] == 3600
    claims = jwt.decode(body["access_token"], SECRET,
                        algorithms=["HS256"], issuer="iia")
    assert claims["tid"] == TA and claims["exp"] - \
        claims["iat"] == 3600 and "role" not in claims
    machines = client.get(
        "/machines", headers={"Authorization": "Bearer " + body["access_token"]})
    assert machines.status_code == 200 and {
        m["machine_id"] for m in machines.json()} == {"A-1"}
    with engine.connect() as c:
        assert c.execute(text("SELECT last_login_at FROM users WHERE email = :e"), {
                         "e": email.lower()}).scalar()


def test_failed_logins_are_indistinguishable(client, make_user, monkeypatch):
    _, ok_email, _ = make_user()
    _, off_email, _ = make_user(active=False)
    import app.api.auth as auth_module
    calls = []
    real = auth_module.verify_password
    monkeypatch.setattr(auth_module, "verify_password",
                        lambda h, p: calls.append(1) or real(h, p))
    attempts = [
        (TA, ok_email, "wrong password"),            # wrong password
        (TA, "nobody@example.com", PW),               # unknown user
        # right user, wrong tenant
        (TB, ok_email, PW),
        ("no-such-tenant", ok_email, PW),             # unknown tenant
        # deactivated, right password
        (TA, off_email, PW),
        (TA, "not-an-email", PW),                     # malformed email
        ("../etc", ok_email, PW),                     # malformed tenant
    ]
    responses = [login(client, t, e, p) for t, e, p in attempts]
    assert {r.status_code for r in responses} == {401}
    assert len({r.text for r in responses}) == 1, [r.text for r in responses]
    assert len({r.headers["www-authenticate"] for r in responses}) == 1
    # Same work for every shape: exactly one argon2 verify per attempt, even when no such account exists.
    assert len(calls) == len(attempts)


def test_login_is_throttled_per_account(client, make_user):
    _, email, _ = make_user()
    codes = [login(client, TA, email, "bad").status_code for _ in range(12)]
    assert codes[:10] == [401] * 10 and codes[10:] == [429, 429]
    r = login(client, TA, email, "bad")
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert login(client, TA, "other@example.com",
                 "bad").status_code == 401   # other accounts unaffected


def test_login_503_when_jwt_is_not_configured(client, make_user):
    _, email, _ = make_user()
    with env(JWT_SECRET=None):
        assert login(client, TA, email).status_code == 503
    with env(JWT_SECRET="too-short"):
        assert login(client, TA, email).status_code == 503
    with env(JWT_TTL_MINUTES=None):
        assert login(client, TA, email).status_code == 503


def test_password_hashes_are_argon2id_and_never_returned_or_selectable(client, make_user, engine):
    uid, email, h = make_user(role="admin")
    with engine.connect() as c:
        stored = c.execute(text("SELECT password_hash FROM users WHERE id = :i"), {
                           "i": uid}).scalar()
    assert stored.startswith("$argon2id$") and PW not in stored
    listing = client.get("/users", headers=h)
    assert listing.status_code == 200 and "argon2" not in listing.text and "password" not in listing.text
    app_url = os.environ.get("APP_DATABASE_URL")
    if not app_url:
        pytest.skip(
            "APP_DATABASE_URL not set (the column-grant check needs the iia_app role)")
    app_engine = create_engine(app_url)
    try:
        with app_engine.connect() as c:
            with pytest.raises(ProgrammingError, match="permission denied"):
                c.execute(text("SELECT password_hash FROM users"))
        with app_engine.connect() as c:
            c.execute(
                text("SELECT set_config('app.tenant_id', :t, false)"), {"t": TA})
            emails = {r[0] for r in c.execute(text("SELECT email FROM users"))}
            assert email in emails
            c.execute(
                text("SELECT set_config('app.tenant_id', :t, false)"), {"t": TB})
            assert email not in {r[0] for r in c.execute(
                text("SELECT email FROM users"))}
    finally:
        app_engine.dispose()


# ---------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------

def test_bad_tokens_are_rejected_with_one_generic_401(client, make_user):
    uid, _, good = make_user()
    token = good["Authorization"].split()[1]
    head, payload, sig = token.split(".")
    forged = json.loads(base64.urlsafe_b64decode(payload + "=="))
    forged["tid"] = TB
    forged_payload = base64.urlsafe_b64encode(
        json.dumps(forged).encode()).rstrip(b"=").decode()
    claims = {"sub": uid, "tid": TA, "iss": "iia",
              "iat": 1, "exp": 4102444800, "jti": "x"}
    cases = {
        "expired": bearer(uid, TA, ttl=-5),
        "wrong secret": bearer(uid, TA, secret="another-secret-that-is-long-enough-xxxxxxxx"),
        "tampered payload": {"Authorization": f"Bearer {head}.{forged_payload}.{sig}"},
        "tampered signature": {"Authorization": f"Bearer {head}.{payload}.{sig[:-3]}AAA"},
        "alg none": {"Authorization": "Bearer " + jwt.encode(claims, key=None, algorithm="none")},
        "wrong tenant (valid signature)": bearer(uid, TB),
        "unknown user": bearer(str(uuid.uuid4()), TA),
        "garbage": {"Authorization": "Bearer not.a.token"},
        "wrong scheme": {"Authorization": "Basic " + token},
        "no token": {"Authorization": "Bearer"},
        "non-uuid subject": {"Authorization": "Bearer " + jwt.encode({**claims, "sub": "x"}, SECRET, algorithm="HS256")},
        "other issuer": {"Authorization": "Bearer " + jwt.encode({**claims, "iss": "evil"}, SECRET, algorithm="HS256")},
    }
    bodies = set()
    for name, headers in cases.items():
        r = client.get("/machines", headers=headers)
        assert r.status_code == 401, name
        bodies.add(r.text)
    assert len(bodies) == 1
    assert client.get("/machines", headers=good).status_code == 200


def test_a_bearer_token_never_falls_back_to_the_demo_tenant(client):
    # Demo posture (no AUTH_REQUIRED): no credential is fine, a bad one is not.
    assert client.get("/machines").status_code == 200
    assert client.get(
        "/machines", headers={"Authorization": "Bearer junk"}).status_code == 401
    with env(JWT_SECRET=None):
        assert client.get(
            "/machines", headers={"Authorization": "Bearer junk"}).status_code == 401


def test_deactivation_and_role_change_apply_to_the_next_request(client, make_user):
    admin_id, _, admin = make_user(role="admin")
    other_admin = make_user(role="admin")
    victim_id, _, victim = make_user(role="admin")
    assert client.get("/users", headers=victim).status_code == 200
    # demote -> the SAME token loses admin rights at once
    assert client.post(f"/users/{victim_id}/role", headers=admin,
                       json={"role": "viewer"}).status_code == 200
    assert client.get("/users", headers=victim).status_code == 403
    assert client.get("/machines", headers=victim).status_code == 200
    # deactivate -> the same token stops working at once
    assert client.post(f"/users/{victim_id}/active", headers=admin,
                       json={"active": False}).status_code == 200
    assert client.get("/machines", headers=victim).status_code == 401
    assert client.post(f"/users/{victim_id}/active",
                       headers=admin, json={"active": True}).status_code == 200
    assert client.get("/machines", headers=victim).status_code == 200
    assert other_admin[0]


def test_a_deactivated_tenant_invalidates_its_tokens(client, make_user, engine):
    _, _, h = make_user(tenant=TB)
    assert client.get("/machines", headers=h).status_code == 200
    with engine.begin() as c:
        c.execute(
            text("UPDATE tenants SET active = FALSE WHERE tenant_id = :t"), {"t": TB})
    try:
        assert client.get("/machines", headers=h).status_code == 401
    finally:
        with engine.begin() as c:
            c.execute(
                text("UPDATE tenants SET active = TRUE WHERE tenant_id = :t"), {"t": TB})


# ---------------------------------------------------------------------
# Role enforcement on every route
# ---------------------------------------------------------------------

def test_viewer_gets_403_on_every_write_route(client, make_user):
    _, _, viewer = make_user(role="viewer")
    checked = 0
    for (method, path), perm in ROUTE_PERMISSIONS.items():
        if role_allows("viewer", perm):
            continue
        r = client.request(method, _fill(path), headers=viewer,
                           json={} if method == "POST" else None)
        assert r.status_code == 403, (method, path, r.status_code, r.text)
        checked += 1
    assert checked >= 8
    # ... and can read
    for path in ("/machines", "/sensors", "/uploads", "/jobs", "/machines/A-1/health"):
        assert client.get(path, headers=viewer).status_code in (200, 404), path
    assert client.get("/machines", headers=viewer).status_code == 200
    assert client.get("/uploads", headers=viewer).status_code == 200
    assert client.get("/jobs", headers=viewer).status_code == 200
    r = client.post("/investigate", headers=viewer,
                    json={"question": "Which machines look abnormal?"})
    assert r.status_code == 200


def test_admin_passes_authorization_on_every_route(client, make_user):
    _, _, admin = make_user(role="admin")
    for (method, path) in ROUTE_PERMISSIONS:
        r = client.request(method, _fill(path), headers=admin,
                           json={} if method == "POST" else None)
        assert r.status_code not in (
            401, 403), (method, path, r.status_code, r.text)


def test_admin_runs_the_upload_flow(client, make_user):
    _, _, admin = make_user(role="admin")
    r = client.post("/uploads", headers=admin,
                    json={"filename": "a.csv", "content_type": "text/csv"})
    assert r.status_code == 201
    assert client.get(
        f"/uploads/{r.json()['upload_id']}", headers=admin).status_code == 200
    assert client.get("/jobs", headers=admin).status_code == 200
    # authorised, no such job
    assert client.post(f"/jobs/{uuid.uuid4()}/retry",
                       headers=admin).status_code == 404


def test_api_key_has_machine_scope_only(client):
    h = api(KEY_A)
    assert client.get("/machines", headers=h).status_code == 200
    assert client.post("/uploads", headers=h,
                       json={"filename": "k.csv", "content_type": "text/csv"}).status_code == 201
    assert client.get("/jobs", headers=h).status_code == 200
    # mapping is allowed (404 = got past authorization, there is no such upload)
    assert client.post(f"/uploads/{uuid.uuid4()}/mapping", headers=h,
                       json={"mapping": {}}).status_code == 404
    # everything beyond machine scope is refused
    assert client.post(f"/jobs/{uuid.uuid4()}/retry",
                       headers=h).status_code == 403
    assert client.get("/users", headers=h).status_code == 403
    assert client.post("/users", headers=h, json={"email": "x@example.com", "password": PW,
                                                  "role": "admin"}).status_code == 403
    assert client.post(f"/users/{uuid.uuid4()}/role",
                       headers=h, json={"role": "admin"}).status_code == 403


def test_legacy_api_key_is_an_api_key_principal_on_the_demo_tenant(client, engine):
    with env(API_KEY="legacy-phase4-key", AUTH_REQUIRED=None):
        h = api("legacy-phase4-key")
        assert len(client.get("/machines", headers=h).json()) == 18
        assert client.post("/uploads", headers=h,
                           json={"filename": "phase4-legacy.csv", "content_type": "text/csv"}).status_code == 201
        assert client.get("/users", headers=h).status_code == 403
        assert client.post(
            f"/jobs/{uuid.uuid4()}/retry", headers=h).status_code == 403
    with engine.begin() as c:
        c.execute(text("DELETE FROM uploads WHERE filename = 'phase4-legacy.csv'"))


# ---------------------------------------------------------------------
# The five legacy auth cases + AUTH_REQUIRED default
# ---------------------------------------------------------------------

def test_the_five_legacy_auth_cases_still_behave_the_same(client):
    # 4. no key, AUTH_REQUIRED off -> demo tenant; reads open, writes 401 (as uploads always were)
    with env(API_KEY=None, AUTH_REQUIRED=None):
        assert len(client.get("/machines").json()) == 18
        assert client.post(
            "/uploads", json={"filename": "x.csv"}).status_code == 401
        assert client.get("/uploads").status_code == 401
        # 3. a wrong key is a 401 even here (never a quiet downgrade to demo)
        assert client.get(
            "/machines", headers=api("iia_wrong")).status_code == 401
        # 2. a tenant key -> that tenant
        assert {m["machine_id"] for m in client.get(
            "/machines", headers=api(KEY_A)).json()} == {"A-1"}
    # 5. no key, AUTH_REQUIRED on -> 401 on every data route
    with env(API_KEY=None, AUTH_REQUIRED="true"):
        assert client.get("/machines").status_code == 401
        assert client.post(
            "/investigate", json={"question": "Which machines look abnormal?"}).status_code == 401
        assert client.get(
            "/healthz").status_code == 200 and client.get("/").status_code == 200
    # 1. legacy API_KEY -> demo tenant, and setting it turns enforcement on
    with env(API_KEY="legacy-bootstrap", AUTH_REQUIRED=None):
        assert client.get("/machines").status_code == 401
        assert len(client.get(
            "/machines", headers=api("legacy-bootstrap")).json()) == 18


def test_a_bearer_token_beats_an_api_key_header(client, make_user):
    _, _, h = make_user(tenant=TA)
    both = {**h, **api(KEY_B)}
    assert {m["machine_id"]
            for m in client.get("/machines", headers=both).json()} == {"A-1"}


# ---------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------

def test_admin_creates_lists_and_the_new_user_can_log_in(client, make_user):
    _, _, admin = make_user(role="admin")
    email = f"New.{uuid.uuid4().hex[:6]}@Example.com"
    r = client.post("/users", headers=admin,
                    json={"email": email, "password": PW, "role": "viewer"})
    assert r.status_code == 201, r.text
    assert r.json()["email"] == email.lower() and r.json()[
        "role"] == "viewer" and r.json()["is_active"] is True
    assert email.lower() in {u["email"]
                             for u in client.get("/users", headers=admin).json()}
    assert login(client, TA, email).status_code == 200
    # duplicate in the same tenant -> 409; the same address in another tenant is a different account
    assert client.post("/users", headers=admin,
                       json={"email": email, "password": PW, "role": "viewer"}).status_code == 409
    _, _, admin_b = make_user(tenant=TB, role="admin")
    assert client.post("/users", headers=admin_b,
                       json={"email": email, "password": PW, "role": "viewer"}).status_code == 201


def test_user_creation_validates_input(client, make_user):
    _, _, admin = make_user(role="admin")
    ok = {"email": "v@example.com", "password": PW, "role": "viewer"}
    assert client.post("/users", headers=admin,
                       json={**ok, "password": "short"}).status_code == 422
    assert client.post("/users", headers=admin,
                       json={**ok, "email": "nope"}).status_code == 422
    assert client.post("/users", headers=admin,
                       json={**ok, "role": "root"}).status_code == 422
    assert client.post("/users", headers=admin,
                       json={**ok, "role": "api_key"}).status_code == 422


def test_user_management_cannot_cross_tenants(client, make_user):
    _, _, admin_a = make_user(tenant=TA, role="admin")
    b_id, b_email, _ = make_user(tenant=TB, role="viewer")
    assert b_email not in {u["email"]
                           for u in client.get("/users", headers=admin_a).json()}
    assert client.post(f"/users/{b_id}/role", headers=admin_a,
                       json={"role": "admin"}).status_code == 404
    assert client.post(f"/users/{b_id}/active", headers=admin_a,
                       json={"active": False}).status_code == 404
    assert client.post(f"/users/{uuid.uuid4()}/role",
                       headers=admin_a, json={"role": "admin"}).status_code == 404
    assert client.post("/users/not-a-uuid/role", headers=admin_a,
                       json={"role": "admin"}).status_code == 404
    assert login(client, TB, b_email).status_code == 200      # untouched


def test_the_last_active_admin_cannot_be_demoted_or_deactivated(client, engine):
    tid = "auth-last-admin"
    with engine.begin() as c:
        c.execute(
            text("DELETE FROM audit_log WHERE tenant_id = :t"), {"t": tid})
        c.execute(text("DELETE FROM users WHERE tenant_id = :t"), {"t": tid})
        c.execute(text("DELETE FROM tenants WHERE tenant_id = :t"), {"t": tid})
        c.execute(text("INSERT INTO tenants (tenant_id, name, api_key_hash) VALUES (:t, 'LA', :h)"),
                  {"t": tid, "h": hash_api_key(generate_api_key())})
        for u in ("a1", "a2"):
            c.execute(text("INSERT INTO users (id, tenant_id, email, password_hash, role) VALUES "
                           "(:i, :t, :e, 'x', 'admin')"), {"i": str(uuid.uuid4()), "t": tid, "e": f"{u}@example.com"})
        ids = {r[0]: str(r[1]) for r in c.execute(
            text("SELECT email, id FROM users WHERE tenant_id = :t"), {"t": tid})}
    try:
        h1, h2 = bearer(ids["a1@example.com"],
                        tid), bearer(ids["a2@example.com"], tid)
        a1, a2 = ids["a1@example.com"], ids["a2@example.com"]
        # two admins: one may be demoted ...
        assert client.post(
            f"/users/{a2}/role", headers=h1, json={"role": "viewer"}).status_code == 200
        # ... but now a1 is the last: neither demote nor deactivate (even by themselves)
        assert client.post(
            f"/users/{a1}/role", headers=h1, json={"role": "viewer"}).status_code == 409
        assert client.post(
            f"/users/{a1}/active", headers=h1, json={"active": False}).status_code == 409
        # promoting a2 again makes demoting a1 legal; deactivating a2 (now the last) is not
        assert client.post(
            f"/users/{a2}/role", headers=h1, json={"role": "admin"}).status_code == 200
        assert client.post(
            f"/users/{a1}/active", headers=h2, json={"active": False}).status_code == 200
        assert client.post(
            f"/users/{a2}/active", headers=h2, json={"active": False}).status_code == 409
        assert client.post(
            f"/users/{a2}/role", headers=h2, json={"role": "viewer"}).status_code == 409
    finally:
        with engine.begin() as c:
            c.execute(
                text("DELETE FROM audit_log WHERE tenant_id = :t"), {"t": tid})
            c.execute(
                text("DELETE FROM users WHERE tenant_id = :t"), {"t": tid})
            c.execute(
                text("DELETE FROM tenants WHERE tenant_id = :t"), {"t": tid})


# ---------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------

def test_audit_records_login_denials_role_changes_and_writes(client, make_user, engine):
    admin_id, admin_email, admin = make_user(role="admin")
    viewer_id, viewer_email, viewer = make_user(role="viewer")

    bad = login(client, TA, viewer_email, "nope-nope-nope")
    ok = login(client, TA, viewer_email)
    assert bad.status_code == 401 and ok.status_code == 200
    denied = client.post("/uploads", headers=viewer,
                         json={"filename": "d.csv", "content_type": "text/csv"})
    assert denied.status_code == 403
    changed = client.post(
        f"/users/{viewer_id}/role", headers=admin, json={"role": "admin"})
    assert changed.status_code == 200
    created = client.post("/uploads", headers=admin,
                          json={"filename": "w.csv", "content_type": "text/csv"})
    assert created.status_code == 201

    rows = audit_rows(engine, TA)
    def by(a, o, actor=None): return [r for r in rows if r["action"] == a and r["outcome"] == o
                                      and (actor is None or r["actor"] == actor)]
    fail = by("auth.login", "failure", viewer_email)
    assert fail and fail[-1]["request_id"] == bad.headers["x-request-id"]
    assert fail[-1]["detail"] == {"reason": "bad_credentials"}
    assert by("auth.login", "success",
              viewer_email)[-1]["request_id"] == ok.headers["x-request-id"]
    den = [r for r in by("permission.denied", "denied")
           if r["request_id"] == denied.headers["x-request-id"]]
    assert den and den[0]["detail"]["required"] == "uploads:write" and den[0]["detail"]["role"] == "viewer"
    assert den[0]["principal_kind"] == "jwt" and den[0]["actor"] == viewer_email
    rc = by("user.role_change", "success")
    assert rc[-1]["detail"] == {"from": "viewer",
                                "to": "admin"} and rc[-1]["actor"] == admin_email
    w = [r for r in by("POST /uploads", "success")
         if r["request_id"] == created.headers["x-request-id"]]
    assert w and w[0]["principal_kind"] == "jwt"
    # an API key's write is attributed to the key, not a person
    k = client.post("/uploads", headers=api(KEY_A),
                    json={"filename": "k.csv", "content_type": "text/csv"})
    kr = [r for r in audit_rows(
        engine, TA, "POST /uploads") if r["request_id"] == k.headers["x-request-id"]]
    assert kr and kr[0]["principal_kind"] == "api_key" and kr[0]["actor"] == "api_key"
    # failed writes are audited too (an upload that does not exist)
    miss = client.post(f"/jobs/{uuid.uuid4()}/retry", headers=admin)
    assert miss.status_code == 404
    fr = [r for r in audit_rows(engine, TA, "POST /jobs/{job_id}/retry")
          if r["request_id"] == miss.headers["x-request-id"]]
    assert fr and fr[0]["outcome"] == "failure" and fr[0]["detail"] == {
        "status": 404}


def test_a_login_for_an_unknown_tenant_writes_no_audit_row(client, engine):
    before = {t: len(audit_rows(engine, t)) for t in (TA, TB)}
    assert login(client, "ghost-tenant", "x@example.com").status_code == 401
    assert {t: len(audit_rows(engine, t)) for t in (TA, TB)} == before


# ---------------------------------------------------------------------
# CORS, CLI
# ---------------------------------------------------------------------

def test_cors_allows_authorization_without_credentials():
    import app.main as main_module
    try:
        with env(CORS_ALLOW_ORIGINS="https://ops.example.com"):
            reloaded = importlib.reload(main_module)
            with TestClient(reloaded.app) as c:
                r = c.options("/users", headers={
                    "Origin": "https://ops.example.com", "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Authorization, Content-Type"})
                assert r.status_code == 200
                allowed = r.headers["access-control-allow-headers"].lower()
                assert "authorization" in allowed and "content-type" in allowed
                assert set(r.headers["access-control-allow-methods"].replace(
                    " ", "").split(",")) == {"GET", "POST", "OPTIONS"}
                assert "access-control-allow-credentials" not in r.headers
                bad = c.options("/users", headers={
                    "Origin": "https://ops.example.com", "Access-Control-Request-Method": "DELETE"})
                assert bad.status_code == 400
    finally:
        with env(CORS_ALLOW_ORIGINS=None):
            importlib.reload(main_module)


def _cli(*args, stdin="", db=None):
    e = {**os.environ, "DATABASE_URL": db or os.environ["DATABASE_URL"]}
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "manage_users.py"), *args],
                          input=stdin, capture_output=True, text=True, env=e, cwd=ROOT)


def test_cli_creates_the_first_admin_without_the_password_on_argv(client, engine):
    email = f"cli.{uuid.uuid4().hex[:6]}@example.com"
    short = _cli("create-admin", TA, email,
                 "--password-stdin", stdin="short\n")
    assert short.returncode == 1 and "at least" in short.stderr + short.stdout
    ok = _cli("create-admin", TA, email.upper(),
              "--password-stdin", stdin=PW + "\n")
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert PW not in ok.stdout + ok.stderr
    assert _cli("create-admin", TA, email, "--password-stdin",
                stdin=PW + "\n").returncode == 1   # exists
    assert _cli("create-admin", "no-such-tenant", email,
                "--password-stdin", stdin=PW + "\n").returncode == 1
    r = login(client, TA, email)
    assert r.status_code == 200 and r.json()["role"] == "admin"
    new_pw = "a completely different pass"
    assert _cli("set-password", TA, email, "--password-stdin",
                stdin=new_pw + "\n").returncode == 0
    assert login(client, TA, email, PW).status_code == 401
    assert login(client, TA, email, new_pw).status_code == 200
    assert email in _cli("list", TA).stderr + _cli("list", TA).stdout


def test_compose_requires_the_jwt_settings_with_no_defaults():
    import yaml
    api_env = yaml.safe_load(
        (ROOT / "docker-compose.yml").read_text())["services"]["api"]["environment"]
    assert api_env["JWT_SECRET"].startswith("${JWT_SECRET:?")
    assert api_env["JWT_TTL_MINUTES"].startswith("${JWT_TTL_MINUTES:?")
