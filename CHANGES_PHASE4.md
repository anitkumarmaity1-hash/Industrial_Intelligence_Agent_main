# Phase 4 changes - users, roles, login

## What changed
- **Migration 0008** (+ `sql/schema.sql`): `users` (argon2id hash, role admin|viewer, `UNIQUE (tenant_id, email)`, lower-case email check), `audit_log` (general audit trail; `investigation_audit_log` untouched), SECURITY DEFINER `auth_lookup_user(tenant, email)` (login) and `auth_check_user(tenant, user_id)` (per-request token check). `iia_app` gets column-level SELECT on `users` **without** `password_hash`, INSERT, UPDATE only on `role/is_active/last_login_at`, INSERT on `audit_log`. RLS: `audit_log` FORCEd; `users` ENABLEd not FORCEd (like `jobs`: the definer lookup runs as owner before a tenant is bound). Downgrade refuses if users/audit rows exist.
- **Principal** (`app/core/principal.py`, extends `TenantContext`, so existing routes are unchanged). `get_principal` replaces the body of `get_tenant`; `get_tenant` stays as a thin wrapper. Legacy cases: legacy `API_KEY` / tenant key -> kind `api_key`, role `api_key`; wrong key -> 401; no key + `AUTH_REQUIRED` off -> kind `demo`, role `viewer`, unauthenticated (reads only, any write -> the legacy 401); no key + on -> 401. `Authorization: Bearer` -> kind `jwt`, role read from the DB on every request; a bad token is 401, never a demo fallback; a Bearer token wins over `X-API-Key`.
- **Role matrix as code** `app/core/permissions.py`: `ROLE_MATRIX` + `ROUTE_PERMISSIONS` (every route) + `PUBLIC_ROUTES`. One router-level dependency `enforce_route_permission` (wired in `app/main.py`; `routes.py`/`uploads.py` are unchanged); an undeclared route is refused 403. A test fails if any registered route is neither declared nor public.
- **Login** `POST /auth/login` (tenant slug + email + password), `app/api/auth.py`: JWT HS256 (`app/core/security.py`), `JWT_SECRET` (>= 32 chars) and `JWT_TTL_MINUTES` required, no defaults (unset -> login 503). No refresh token. Generic 401 for every failure, one argon2 verify per attempt (dummy hash for unknown accounts), throttle 10/min per (tenant, email) and 30/min per IP.
- **User management** (admin only, own tenant only, all POST so CORS gains only the `Authorization` header): `GET/POST /users`, `POST /users/{id}/role`, `POST /users/{id}/active`; last active admin cannot be demoted/deactivated (row lock).
- **CLI** `scripts/manage_users.py`: `create-admin`, `list`, `set-password` (password via hidden prompt or `--password-stdin`, never argv). `set-password` was my addition for lockout recovery.
- **Audit** `app/core/audit.py`: login success/failure, permission denials, role changes, user create/(de)activate, and every write route (success, or failure + status) with principal, tenant, request id.
- **Dependencies**: `argon2-cffi==25.1.0`, `PyJWT==2.15.1` in `requirements.txt` and `requirements-runtime.txt`. **Config**: `JWT_SECRET`, `JWT_TTL_MINUTES` in compose (required) and `.env.example`.

## Behaviour changes to know about
- The legacy `API_KEY` and tenant keys can no longer `POST /jobs/{id}/retry` (403): machine scope = read + upload + mapping.
- `POST /uploads*` without a key now returns `401 "Missing or invalid API key."` (was `"Uploads require an API key."`; same status).
- `/uploads/{id}/preview` (raw rows) needs `uploads:write`: not viewers.

## Verified (actually run; local Postgres 16, Java 21, Python 3.13, `pytest -m "not dense and not load"`)
- **Baseline** (your Phase 3 folder, untouched, run first): 413 passed, 1 failed, 10 skipped, 18 deselected.
- **After Phase 4: 443 passed, 1 failed, 10 skipped, 18 deselected** (+30: 29 in `tests/test_auth.py`, 1 `test_0008_*`).
- The 1 failure is identical before and after and is not from Phase 4: `test_security_defaults.py::test_compose_has_no_hardcoded_database_credentials` expects the DB port `127.0.0.1:5433:5432` but your `docker-compose.yml` publishes `5434`. I left your compose port alone; change the test or the port.
- Skipped (10, same as before): `google.genai` / `pinecone` not installed.
- Negative controls (code temporarily broken, tests went red, then restored): undeclared route; viewer given write permission; deactivation check removed.
- Covered: viewer 403 on every write route; admin passes every route; API key scope; expired/tampered/alg-none/wrong-secret/wrong-tenant/unknown-user tokens all give the same 401; deactivated user and deactivated tenant rejected on the next request; role change effective immediately; failed-login bodies identical and exactly one argon2 verify each; cross-tenant user management is 404; last-admin rule; `password_hash` unreadable by `iia_app`; audit rows incl. request ids; CORS allows `Authorization`, no credentials, only GET/POST/OPTIONS; CLI; the five legacy cases; migration 0008 up/down/guard; `schema.sql` == `alembic upgrade head`.
- Existing tests edited: `test_uploads_pipeline.py` (job retries now use an admin Bearer token, plus an assertion that a tenant key gets 403; cleanup also removes users/audit_log), `test_tenant_isolation.py` (table list), `test_migrations.py` (new test only).

## NOT verified
`docker compose up` and the compose `JWT_*` wiring beyond YAML parsing; GitHub Actions; `-m dense`, `-m load`; Python 3.12 (ran 3.13); a non-superuser table owner; timing equality of login responses beyond the one-verify-per-attempt check; behaviour behind a reverse proxy; the Streamlit frontend (unchanged, API key only).

## Known limitations
See README "Limitations" (Phase 4 bullet): no password reset / email verification / SSO / MFA / token revocation / refresh token / login UI / audit read endpoint; login throttle counts attempts and is per process; tenant ids can be probed by timing; reserved permissions `registry:write` and `connectors:write` have no routes yet.
